"""Shared provider failure classification for the gateway implementations.

Embedding and rerank talk to the same providers over the same transport, so a
429 or a connect timeout means the same thing and must be reported the same way
— above all the retryable flag, which decides whether an ingest job retries or
gives up. Keeping one table here is what stops the two gateways from drifting
into disagreeing about that.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpcore
import httpx

from rag_service.providers.credentials import ProviderCredentialUnavailableError
from rag_service.providers.network_policy import ProviderNetworkPolicyError


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """A provider outcome, kept free of anything the caller supplied."""

    code: str
    message: str
    retryable: bool


def status_failure(status_code: int, *, input_rejected_message: str) -> ProviderFailure | None:
    """Classify an HTTP status, returning None when the response succeeded.

    `input_rejected_message` is the only caller-specific part: an embedding call
    and a rerank call reject different things, and the message reaches operators.
    """
    if type(status_code) is not int:
        return ProviderFailure("PROVIDER_RESPONSE_INVALID", "Provider response is invalid", False)
    if 200 <= status_code < 300:
        return None
    if status_code == 429:
        return ProviderFailure("PROVIDER_RATE_LIMITED", "Provider rate limited", True)
    if 500 <= status_code <= 599:
        return ProviderFailure("PROVIDER_UNAVAILABLE", "Provider unavailable", True)
    if status_code in {401, 403}:
        return ProviderFailure(
            "PROVIDER_AUTHENTICATION_FAILED", "Provider authentication failed", False
        )
    if status_code == 404:
        return ProviderFailure("PROVIDER_MODEL_NOT_FOUND", "Provider model not found", False)
    if status_code in {400, 413, 422}:
        return ProviderFailure("PROVIDER_INPUT_REJECTED", input_rejected_message, False)
    if 300 <= status_code <= 399:
        return ProviderFailure("PROVIDER_REDIRECT_REJECTED", "Provider redirect rejected", False)
    return ProviderFailure("PROVIDER_REQUEST_REJECTED", "Provider request rejected", False)


def transport_failure(error: Exception) -> ProviderFailure:
    """Classify a transport-level exception.

    Anything unrecognised is reported as retryable on purpose: an unknown
    network fault is far more often transient than permanent, and the caller's
    attempt ceiling bounds how long that optimism can cost.
    """
    if isinstance(error, ProviderNetworkPolicyError):
        return ProviderFailure("PROVIDER_ENDPOINT_REJECTED", "Provider endpoint rejected", False)
    if isinstance(error, ValueError):
        return ProviderFailure("PROVIDER_RESPONSE_INVALID", "Provider response is invalid", False)
    if isinstance(
        error,
        (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpcore.TimeoutException,
        ),
    ):
        return ProviderFailure("PROVIDER_TIMEOUT", "Provider request timed out", True)
    if isinstance(
        error,
        (
            httpx.ConnectError,
            httpx.NetworkError,
            httpcore.ConnectError,
            httpcore.NetworkError,
        ),
    ):
        return ProviderFailure("PROVIDER_UNAVAILABLE", "Provider unavailable", True)
    if isinstance(error, ProviderCredentialUnavailableError):
        return ProviderFailure(
            "PROVIDER_CREDENTIAL_UNAVAILABLE", "Provider credential unavailable", False
        )
    return ProviderFailure("PROVIDER_UNAVAILABLE", "Provider unavailable", True)


__all__ = ["ProviderFailure", "status_failure", "transport_failure"]
