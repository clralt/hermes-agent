from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest
from unittest.mock import patch

from hermes_cli.audit_verdict import (
    AuditVerdictError,
    _verify_ed25519_signature,
    verify_signed_audit_verdict,
)


# RFC 8032 section 7.1, test vector 1. This is a published verification
# vector; no private audit key is created or retained by this test suite.
_RFC_PUBLIC = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)
_RFC_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)


def _verdict() -> dict:
    return {
        "schema_version": 2,
        "verdict_id": "verdict-1",
        "task_id": "HERMES-RESUMABLE-EXECUTION",
        "authority_task_id": "close-1",
        "audit_task_id": "audit-1",
        "status": "CLEAN",
        "candidate_tree": "a" * 40,
        "contract_sha256": "b" * 64,
        "evidence_manifest_sha256": "c" * 64,
        "dossier_sha256": "d" * 64,
        "audit_checkpoint_sha256": "e" * 64,
        "reviewer": {
            "identity": "desktop-independent-auditor",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
        },
        "producer_identity": "worker-1",
        "created_at": "2026-08-10T00:00:00Z",
        "expires_at": "2026-08-11T00:00:00Z",
        "signer_key_id": "desktop-key-1",
    }


def _trust_store() -> dict:
    return {
        "schema_version": 1,
        "keys": [{
            "key_id": "desktop-key-1",
            "algorithm": "ed25519",
            "public_key_base64": base64.b64encode(_RFC_PUBLIC).decode("ascii"),
            "allowed_reviewer_identities": ["desktop-independent-auditor"],
            "allowed_models": ["gpt-5.6-sol"],
            "allowed_reasoning_efforts": ["high"],
            "allowed_task_prefixes": ["audit-"],
            "not_before": "2026-08-01T00:00:00Z",
            "not_after": "2026-09-01T00:00:00Z",
            "revoked": False,
        }],
    }


def test_published_ed25519_verification_vector_passes_without_private_key():
    _verify_ed25519_signature(_RFC_PUBLIC, _RFC_SIGNATURE, b"")


def test_unsigned_clean_verdict_fails_closed():
    verdict = _verdict()
    with pytest.raises(AuditVerdictError, match="no detached signature"):
        verify_signed_audit_verdict(
            verdict,
            _trust_store(),
            expected={"authority_task_id": "close-1", "status": "CLEAN"},
            now=datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )


def test_wrong_candidate_binding_fails_before_signature_can_grant_authority():
    verdict = _verdict()
    verdict["signature"] = base64.b64encode(_RFC_SIGNATURE).decode("ascii")
    with pytest.raises(AuditVerdictError, match="candidate_tree does not match"):
        verify_signed_audit_verdict(
            verdict,
            _trust_store(),
            expected={"candidate_tree": "f" * 40, "status": "CLEAN"},
            now=datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )


def test_untrusted_signer_and_stale_verdict_fail_closed():
    verdict = _verdict()
    verdict["signature"] = base64.b64encode(_RFC_SIGNATURE).decode("ascii")
    verdict["signer_key_id"] = "unknown"
    with pytest.raises(AuditVerdictError, match="not currently trusted"):
        verify_signed_audit_verdict(
            verdict,
            _trust_store(), expected={},
            now=datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )

    verdict["signer_key_id"] = "desktop-key-1"
    with pytest.raises(AuditVerdictError, match="not currently fresh"):
        verify_signed_audit_verdict(
            verdict,
            _trust_store(), expected={},
            now=datetime(2026, 8, 12, tzinfo=timezone.utc),
        )


def test_closed_envelope_and_current_key_validity_fail_closed():
    verdict = _verdict()
    verdict["signature"] = base64.b64encode(_RFC_SIGNATURE).decode("ascii")
    verdict["unknown_semantic_authority"] = True
    with pytest.raises(AuditVerdictError, match="unknown fields"):
        verify_signed_audit_verdict(
            verdict, _trust_store(), expected={},
            now=datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )

    verdict.pop("unknown_semantic_authority")
    store = _trust_store()
    store["keys"][0]["not_after"] = "2026-08-10T11:00:00Z"
    with pytest.raises(AuditVerdictError, match="signer is not currently valid"):
        verify_signed_audit_verdict(
            verdict, store, expected={},
            now=datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )


def test_schema_shaped_verdict_reaches_crypto_and_binds_nested_reviewer():
    verdict = _verdict()
    verdict["signature"] = base64.b64encode(_RFC_SIGNATURE).decode("ascii")
    expected = {
        "task_id": "HERMES-RESUMABLE-EXECUTION",
        "authority_task_id": "close-1",
        "audit_checkpoint_sha256": "e" * 64,
        "producer_identity": "worker-1",
        "reviewer": verdict["reviewer"],
        "status": "CLEAN",
    }
    with patch("hermes_cli.audit_verdict._verify_ed25519_signature") as crypto:
        receipt = verify_signed_audit_verdict(
            verdict, _trust_store(), expected=expected,
            now=datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )
    crypto.assert_called_once()
    assert receipt["audit_task_id"] == "audit-1"
    assert receipt["authority_task_id"] == "close-1"
