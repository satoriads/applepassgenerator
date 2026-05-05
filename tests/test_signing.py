"""
Tests for the manifest signing path in ApplePass._create_signature_crypto.

Apple Wallet pkpass requires:
  - Detached PKCS7 (CMS) SignedData over manifest.json
  - SHA1 digest (no, this is not negotiable as of 2026)
  - DER encoding
  - Signer's WWDR intermediate cert embedded in the SignedData certificates set

These tests generate a self-signed throwaway PKI at runtime so no real Apple
credentials are touched. They assert structural shape via asn1crypto and
round-trip verifiability via the openssl CLI (ships with macOS and most
Linux distros, including the GitHub Actions runners).
"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import tempfile

import pytest
from asn1crypto import cms
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from applepassgenerator.models import ApplePass


SHA1_OID = "1.3.14.3.2.26"
PASSWORD = "test-password"


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    """
    Build a throwaway 3-cert chain: root CA -> WWDR-shaped intermediate -> leaf.

    Writes them to PEM files under a tempdir, returns the directory and paths.
    The leaf private key is encrypted with PASSWORD to mirror how Apple ships
    the developer certificate key.
    """
    tmpdir = tmp_path_factory.mktemp("pki")

    # Root CA
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    root_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Test Root CA"),
    ])
    root_cert = (
        x509.CertificateBuilder()
        .subject_name(root_subject)
        .issuer_name(root_subject)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650)
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(root_key, hashes.SHA256())
    )

    # WWDR-shaped intermediate
    wwdr_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    wwdr_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Test WWDR Intermediate"),
    ])
    wwdr_cert = (
        x509.CertificateBuilder()
        .subject_name(wwdr_subject)
        .issuer_name(root_subject)
        .public_key(wwdr_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650)
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(root_key, hashes.SHA256())
    )

    # Leaf signing cert (the "Pass Type ID" cert)
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Test Pass Type ID"),
    ])
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_subject)
        .issuer_name(wwdr_subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(wwdr_key, hashes.SHA256())
    )

    # Write the files
    root_pem = tmpdir / "root.pem"
    wwdr_pem = tmpdir / "wwdr.pem"
    leaf_pem = tmpdir / "leaf.pem"
    leaf_key_pem = tmpdir / "leaf_key.pem"

    root_pem.write_bytes(root_cert.public_bytes(serialization.Encoding.PEM))
    wwdr_pem.write_bytes(wwdr_cert.public_bytes(serialization.Encoding.PEM))
    leaf_pem.write_bytes(leaf_cert.public_bytes(serialization.Encoding.PEM))
    leaf_key_pem.write_bytes(
        leaf_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(
                PASSWORD.encode("utf-8")
            ),
        )
    )

    return {
        "tmpdir": tmpdir,
        "root_pem": str(root_pem),
        "wwdr_pem": str(wwdr_pem),
        "leaf_pem": str(leaf_pem),
        "leaf_key_pem": str(leaf_key_pem),
    }


@pytest.fixture
def signature_and_manifest(pki):
    """Sign a representative manifest and return (signature_bytes, manifest_str)."""
    manifest = '{"pass.json":"deadbeef","icon.png":"cafef00d"}'
    pass_obj = ApplePass.__new__(ApplePass)  # bypass __init__; we only need _create_signature_crypto
    sig = pass_obj._create_signature_crypto(
        manifest=manifest,
        certificate=pki["leaf_pem"],
        key=pki["leaf_key_pem"],
        wwdr_certificate=pki["wwdr_pem"],
        password=PASSWORD,
    )
    return sig, manifest


def test_signature_is_signed_data_content_type(signature_and_manifest):
    """Output must be a CMS ContentInfo of type signed_data."""
    sig, _ = signature_and_manifest
    content_info = cms.ContentInfo.load(sig)
    assert content_info["content_type"].native == "signed_data"


def test_signature_is_detached(signature_and_manifest):
    """encap_content_info must be type 'data' with NO embedded content (detached)."""
    sig, _ = signature_and_manifest
    signed_data = cms.ContentInfo.load(sig)["content"]
    eci = signed_data["encap_content_info"]
    assert eci["content_type"].native == "data"
    # Detached: the OCTET STRING content field is absent / void
    content = eci["content"]
    assert content.native is None, (
        "Signature should be detached (no eContent), but content was set"
    )


def test_signature_uses_sha1_digest(signature_and_manifest):
    """Both digest_algorithms set and the signer's digest_algorithm must be SHA1."""
    sig, _ = signature_and_manifest
    signed_data = cms.ContentInfo.load(sig)["content"]

    digest_oids = {da["algorithm"].dotted for da in signed_data["digest_algorithms"]}
    assert SHA1_OID in digest_oids, f"Expected SHA1 OID in {digest_oids}"

    signer_infos = signed_data["signer_infos"]
    assert len(signer_infos) == 1, "Expected exactly one SignerInfo"
    assert signer_infos[0]["digest_algorithm"]["algorithm"].dotted == SHA1_OID


def test_signature_embeds_both_certs(signature_and_manifest):
    """SignedData certificates field must contain BOTH the signing leaf and WWDR."""
    sig, _ = signature_and_manifest
    signed_data = cms.ContentInfo.load(sig)["content"]
    certs = signed_data["certificates"]
    assert len(certs) == 2, f"Expected 2 certs (signer + WWDR), got {len(certs)}"


def test_signer_identifier_matches_leaf(signature_and_manifest, pki):
    """SignerInfo issuer/serial must identify the leaf cert we signed with."""
    sig, _ = signature_and_manifest
    leaf_bytes = open(pki["leaf_pem"], "rb").read()
    leaf = x509.load_pem_x509_certificate(leaf_bytes)

    signed_data = cms.ContentInfo.load(sig)["content"]
    signer_info = signed_data["signer_infos"][0]
    sid = signer_info["sid"]
    assert sid.name == "issuer_and_serial_number"
    assert sid.chosen["serial_number"].native == leaf.serial_number


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")
def test_signature_round_trip_verifies(signature_and_manifest, pki, tmp_path):
    """openssl smime -verify must accept the signature against the test root."""
    sig, manifest = signature_and_manifest

    sig_path = tmp_path / "signature"
    manifest_path = tmp_path / "manifest"
    sig_path.write_bytes(sig)
    manifest_path.write_bytes(manifest.encode("utf-8"))

    result = subprocess.run(
        [
            "openssl", "smime", "-verify",
            "-in", str(sig_path),
            "-content", str(manifest_path),
            "-inform", "DER",
            "-binary",
            "-CAfile", pki["root_pem"],
            "-purpose", "any",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"openssl verify failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert b"Verification successful" in result.stderr


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")
def test_tampered_manifest_fails_verification(signature_and_manifest, pki, tmp_path):
    """Flipping one byte of the manifest must cause verification to fail."""
    sig, manifest = signature_and_manifest

    tampered = bytearray(manifest.encode("utf-8"))
    tampered[0] ^= 0x01
    sig_path = tmp_path / "signature"
    manifest_path = tmp_path / "manifest"
    sig_path.write_bytes(sig)
    manifest_path.write_bytes(bytes(tampered))

    result = subprocess.run(
        [
            "openssl", "smime", "-verify",
            "-in", str(sig_path),
            "-content", str(manifest_path),
            "-inform", "DER",
            "-binary",
            "-CAfile", pki["root_pem"],
            "-purpose", "any",
        ],
        capture_output=True,
    )
    assert result.returncode != 0, "Verification must fail on tampered manifest"


def test_signed_attributes_present(signature_and_manifest):
    """attrs=True is critical for iOS strictness; assert signed attrs exist."""
    sig, _ = signature_and_manifest
    signed_data = cms.ContentInfo.load(sig)["content"]
    signer_info = signed_data["signer_infos"][0]
    signed_attrs = signer_info["signed_attrs"]
    assert signed_attrs is not None and len(signed_attrs) > 0, (
        "Expected signed attributes (signing-time, content-type, message-digest)"
    )
    attr_oids = {a["type"].dotted for a in signed_attrs}
    # CMS message-digest OID must be present
    assert "1.2.840.113549.1.9.4" in attr_oids
    # CMS content-type OID must be present
    assert "1.2.840.113549.1.9.3" in attr_oids
