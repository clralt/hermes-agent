"""Verification-only trust boundary for independently signed audit verdicts.

This host never signs verdicts and never needs private audit material.  A desktop
signer produces a detached Ed25519 signature over canonical JSON; the governed
closer verifies it against an operator-controlled public trust store.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class AuditVerdictError(ValueError):
    """A purported audit verdict is untrusted, stale, replayed, or misbound."""


def _canonical_json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AuditVerdictError(f"signed verdict {field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AuditVerdictError(f"signed verdict {field} is invalid") from exc
    return parsed.astimezone(timezone.utc)


def _decode_base64(value: Any, field: str, expected_bytes: int) -> bytes:
    try:
        decoded = base64.b64decode(str(value), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AuditVerdictError(f"signed verdict {field} is not valid base64") from exc
    if len(decoded) != expected_bytes:
        raise AuditVerdictError(
            f"signed verdict {field} must decode to {expected_bytes} bytes"
        )
    return decoded


def _verify_ed25519_signature(public_key: bytes, signature: bytes, message: bytes) -> None:
    """Verify a detached Ed25519 signature; signing is intentionally absent."""
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError) as exc:
        raise AuditVerdictError("signed verdict signature verification failed") from exc


def verify_signed_audit_verdict(
    verdict: Mapping[str, Any],
    trust_store: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify signature, signer scope, freshness, and exact authority bindings."""
    if not isinstance(verdict, Mapping):
        raise AuditVerdictError("signed audit verdict must be an object")
    if verdict.get("schema_version") != 2:
        raise AuditVerdictError("signed audit verdict schema_version must be 2")
    signature_value = verdict.get("signature")
    if not signature_value:
        raise AuditVerdictError("signed audit verdict has no detached signature")
    signer_key_id = str(verdict.get("signer_key_id") or "")
    keys = trust_store.get("keys") if isinstance(trust_store, Mapping) else None
    if trust_store.get("schema_version") != 1 or not isinstance(keys, list):
        raise AuditVerdictError("audit trust store is invalid")
    key = next(
        (item for item in keys if isinstance(item, Mapping)
         and item.get("key_id") == signer_key_id),
        None,
    )
    if key is None or key.get("algorithm") != "ed25519" or key.get("revoked") is True:
        raise AuditVerdictError("signed verdict signer is not currently trusted")

    allowed_fields = {
        "schema_version", "verdict_id", "task_id", "authority_task_id",
        "audit_task_id", "status", "candidate_tree", "contract_sha256",
        "evidence_manifest_sha256", "dossier_sha256",
        "audit_checkpoint_sha256", "reviewer", "producer_identity",
        "created_at", "expires_at", "signer_key_id", "signature",
    }
    unknown = sorted(set(verdict) - allowed_fields)
    if unknown:
        raise AuditVerdictError(
            "signed audit verdict has unknown fields: " + ", ".join(unknown)
        )
    required = allowed_fields - {"signature"}
    missing = sorted(field for field in required if not verdict.get(field))
    if missing:
        raise AuditVerdictError(
            "signed audit verdict is incomplete: " + ", ".join(missing)
        )
    reviewer = verdict.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {
        "identity", "model", "reasoning_effort",
    } or not all(reviewer.values()):
        raise AuditVerdictError("signed audit verdict reviewer binding is invalid")
    if verdict.get("status") not in {"CLEAN", "FAIL"}:
        raise AuditVerdictError("signed audit verdict status must be CLEAN or FAIL")
    for field, value in expected.items():
        if verdict.get(field) != value:
            raise AuditVerdictError(f"signed audit verdict {field} does not match authority binding")

    allowed_reviewers = key.get("allowed_reviewer_identities") or []
    allowed_models = key.get("allowed_models") or []
    allowed_efforts = key.get("allowed_reasoning_efforts") or []
    task_prefixes = key.get("allowed_task_prefixes") or []
    if allowed_reviewers and reviewer["identity"] not in allowed_reviewers:
        raise AuditVerdictError("signed verdict reviewer identity is outside signer scope")
    if allowed_models and reviewer["model"] not in allowed_models:
        raise AuditVerdictError("signed verdict model is outside signer scope")
    if allowed_efforts and reviewer["reasoning_effort"] not in allowed_efforts:
        raise AuditVerdictError("signed verdict reasoning effort is outside signer scope")
    if task_prefixes and not any(
        str(verdict["audit_task_id"]).startswith(str(prefix)) for prefix in task_prefixes
    ):
        raise AuditVerdictError("signed verdict task is outside signer scope")

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created = _parse_utc(verdict["created_at"], "created_at")
    expires = _parse_utc(verdict["expires_at"], "expires_at")
    not_before = _parse_utc(key["not_before"], "signer not_before") if key.get("not_before") else None
    not_after = _parse_utc(key["not_after"], "signer not_after") if key.get("not_after") else None
    if created > current or expires <= current or expires <= created:
        raise AuditVerdictError("signed audit verdict is not currently fresh")
    if (not_before and (created < not_before or current < not_before)) or (
        not_after and (created >= not_after or current >= not_after)
    ):
        raise AuditVerdictError("signed verdict signer is not currently valid")

    public_key = _decode_base64(key.get("public_key_base64"), "public key", 32)
    signature = _decode_base64(signature_value, "signature", 64)
    unsigned = dict(verdict)
    unsigned.pop("signature", None)
    payload = _canonical_json_bytes(unsigned)
    _verify_ed25519_signature(public_key, signature, payload)
    signed_bytes = _canonical_json_bytes(dict(verdict))
    return {
        "verdict_id": verdict["verdict_id"],
        "verdict_sha256": hashlib.sha256(signed_bytes).hexdigest(),
        "signer_key_id": signer_key_id,
        "status": verdict["status"],
        "candidate_tree": verdict["candidate_tree"],
        "authority_task_id": verdict["authority_task_id"],
        "audit_task_id": verdict["audit_task_id"],
        "expires_at": verdict["expires_at"],
    }
