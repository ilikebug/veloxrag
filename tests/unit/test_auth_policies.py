from dataclasses import FrozenInstanceError
from enum import StrEnum
from typing import Any
from uuid import UUID

import pytest

from rag_service.api.errors import BusinessError
from rag_service.auth.codec import KeyKind
from rag_service.auth.policies import (
    AdminPrincipal,
    AgentPrincipal,
    Capability,
    materialize_principal,
    require_admin,
    require_capability,
    require_document_read,
    require_knowledge_base_access,
)
from rag_service.db.models.auth import ApiKey

KEY_ID = UUID("10000000-0000-0000-0000-000000000001")
KB_ID = UUID("20000000-0000-0000-0000-000000000001")
NEW_KB_ID = UUID("20000000-0000-0000-0000-000000000002")
QUERY_PROFILE_ID = UUID("30000000-0000-0000-0000-000000000001")
PUBLIC_ID = "cHVibGljLWlkLXNvdXJjZQ"


class FalseyCapabilityList(list[str]):
    def __bool__(self) -> bool:
        return False


class CapabilityStringSubclass(str):
    pass


class KeyTypeStringSubclass(str):
    pass


class ForeignCapability(StrEnum):
    RETRIEVE = "retrieve"


class ExplodingCapabilityValue:
    def __hash__(self) -> int:
        raise RuntimeError("sensitive capability hash failure")

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("sensitive capability equality failure")

    def __repr__(self) -> str:
        return "sensitive-exploding-capability"


def _api_key(
    *,
    key_type: str,
    public_id: str = PUBLIC_ID,
    capabilities: Any = None,
    raw_file_read: Any = False,
    requests_per_minute: Any = None,
    max_concurrency: Any = None,
) -> ApiKey:
    values: dict[str, Any] = {
        "id": KEY_ID,
        "public_id": public_id,
        "key_type": key_type,
        "capabilities": capabilities,
        "raw_file_read": raw_file_read,
        "requests_per_minute": requests_per_minute,
        "max_concurrency": max_concurrency,
    }
    return ApiKey(**values)


def _agent(
    *capabilities: Capability,
    knowledge_base_ids: frozenset[UUID] = frozenset({KB_ID}),
) -> AgentPrincipal:
    return AgentPrincipal(
        key_id=KEY_ID,
        public_id=PUBLIC_ID,
        capabilities=frozenset(capabilities),
        knowledge_base_ids=knowledge_base_ids,
        query_profile_ids=frozenset({QUERY_PROFILE_ID}),
        default_query_profile_id=QUERY_PROFILE_ID,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


def test_capabilities_are_exact_and_separate_from_raw_file_permission() -> None:
    assert {capability.value for capability in Capability} == {
        "ingest",
        "retrieve",
        "answer",
        "manage",
    }
    assert "raw_file:read" not in Capability


def test_admin_row_with_empty_agent_policy_materializes_admin_principal() -> None:
    row = _api_key(
        key_type="admin",
        capabilities=[],
        raw_file_read=False,
        requests_per_minute=None,
        max_concurrency=None,
    )

    principal = materialize_principal(row)

    assert principal == AdminPrincipal(key_id=KEY_ID, public_id=PUBLIC_ID)
    assert principal.key_type == "admin"
    assert not hasattr(principal, "capabilities")
    with pytest.raises(FrozenInstanceError):
        principal.public_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "policy_values",
    [
        pytest.param({"capabilities": None}, id="capabilities-null"),
        pytest.param({"capabilities": "sensitive-capabilities"}, id="capabilities-not-list"),
        pytest.param({"capabilities": FalseyCapabilityList()}, id="capabilities-empty-subclass"),
        pytest.param(
            {"capabilities": FalseyCapabilityList(["manage"])},
            id="capabilities-falsey-nonempty-subclass",
        ),
        pytest.param({"capabilities": ["manage"]}, id="capabilities-known-nonempty"),
        pytest.param(
            {"capabilities": ["sensitive-admin-capability"]},
            id="capabilities-unknown-nonempty",
        ),
        pytest.param({"raw_file_read": "sensitive-raw-file-read"}, id="raw-file-not-bool"),
        pytest.param({"raw_file_read": True}, id="raw-file-enabled"),
        pytest.param({"requests_per_minute": "sensitive-rpm"}, id="rpm-wrong-type"),
        pytest.param({"requests_per_minute": 60}, id="rpm-present"),
        pytest.param({"max_concurrency": "sensitive-concurrency"}, id="concurrency-wrong-type"),
        pytest.param({"max_concurrency": 4}, id="concurrency-present"),
        pytest.param(
            {
                "capabilities": ["manage"],
                "raw_file_read": True,
                "requests_per_minute": 60,
                "max_concurrency": 4,
            },
            id="all-agent-policy-fields-present",
        ),
    ],
)
def test_admin_row_rejects_malformed_persisted_agent_policy_safely(
    policy_values: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "capabilities": [],
        "raw_file_read": False,
        "requests_per_minute": None,
        "max_concurrency": None,
    }
    values.update(policy_values)
    row = _api_key(key_type="admin", **values)

    with pytest.raises(BusinessError) as exc_info:
        materialize_principal(row)

    error = exc_info.value
    assert error == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    assert error.__context__ is None
    assert error.__cause__ is None
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("rag_service"):
            locals_repr = repr(traceback.tb_frame.f_locals)
            assert "sensitive" not in locals_repr
            assert repr(policy_values) not in locals_repr
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    ("scope_values", "markers"),
    [
        pytest.param(
            {"knowledge_base_ids": frozenset({KB_ID})},
            (str(KB_ID),),
            id="kb-scope-nonempty",
        ),
        pytest.param(
            {"knowledge_base_ids": ["sensitive-kb-scope"]},
            ("sensitive-kb-scope",),
            id="kb-scope-malformed",
        ),
        pytest.param(
            {"query_profile_ids": frozenset({QUERY_PROFILE_ID})},
            (str(QUERY_PROFILE_ID),),
            id="query-profile-scope-nonempty",
        ),
        pytest.param(
            {"query_profile_ids": ["sensitive-query-profile-scope"]},
            ("sensitive-query-profile-scope",),
            id="query-profile-scope-malformed",
        ),
        pytest.param(
            {"default_query_profile_id": QUERY_PROFILE_ID},
            (str(QUERY_PROFILE_ID),),
            id="default-query-profile-present",
        ),
        pytest.param(
            {"default_query_profile_id": "sensitive-default-query-profile"},
            ("sensitive-default-query-profile",),
            id="default-query-profile-malformed",
        ),
        pytest.param(
            {
                "knowledge_base_ids": frozenset({KB_ID}),
                "query_profile_ids": frozenset({QUERY_PROFILE_ID}),
                "default_query_profile_id": QUERY_PROFILE_ID,
            },
            (str(KB_ID), str(QUERY_PROFILE_ID)),
            id="all-external-agent-scope-fields-present",
        ),
    ],
)
def test_admin_row_rejects_and_redacts_external_agent_scopes(
    scope_values: dict[str, Any],
    markers: tuple[str, ...],
) -> None:
    row = _api_key(key_type="admin", capabilities=[])

    with pytest.raises(BusinessError) as exc_info:
        materialize_principal(row, **scope_values)

    error = exc_info.value
    assert error == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    assert error.__context__ is None
    assert error.__cause__ is None
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("rag_service"):
            locals_repr = repr(traceback.tb_frame.f_locals)
            for marker in markers:
                assert marker not in locals_repr
        traceback = traceback.tb_next


def test_agent_row_materializes_only_explicit_policy_and_scope_values() -> None:
    row = _api_key(
        key_type="agent",
        capabilities=["retrieve", "manage"],
        raw_file_read=True,
        requests_per_minute=120,
        max_concurrency=8,
    )

    principal = materialize_principal(
        row,
        knowledge_base_ids=frozenset({KB_ID}),
        query_profile_ids=frozenset({QUERY_PROFILE_ID}),
        default_query_profile_id=QUERY_PROFILE_ID,
    )

    assert principal == AgentPrincipal(
        key_id=KEY_ID,
        public_id=PUBLIC_ID,
        capabilities=frozenset({Capability.RETRIEVE, Capability.MANAGE}),
        knowledge_base_ids=frozenset({KB_ID}),
        query_profile_ids=frozenset({QUERY_PROFILE_ID}),
        default_query_profile_id=QUERY_PROFILE_ID,
        raw_file_read=True,
        requests_per_minute=120,
        max_concurrency=8,
    )
    assert principal.key_type == "agent"


@pytest.mark.parametrize(
    ("raw_capabilities", "marker"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("retrieve", None, id="string-not-list"),
        pytest.param(("retrieve",), None, id="tuple-not-list"),
        pytest.param(FalseyCapabilityList(["retrieve"]), None, id="list-subclass"),
        pytest.param([Capability.RETRIEVE], None, id="enum-element"),
        pytest.param([CapabilityStringSubclass("retrieve")], None, id="string-subclass-element"),
        pytest.param([ForeignCapability.RETRIEVE], None, id="foreign-enum-element"),
        pytest.param(["retrieve", "retrieve"], None, id="duplicate"),
        pytest.param(
            ["ingest", "retrieve", "answer", "manage", "retrieve"],
            None,
            id="over-bound",
        ),
        pytest.param(
            [ExplodingCapabilityValue()] * 5,
            "sensitive capability",
            id="over-bound-before-element-conversion",
        ),
        pytest.param(
            [ExplodingCapabilityValue()],
            "sensitive capability",
            id="malicious-non-string-element",
        ),
        pytest.param(["sensitive-unknown-capability"], "sensitive-unknown", id="unknown"),
    ],
)
def test_agent_row_rejects_malformed_persisted_capabilities_safely(
    raw_capabilities: Any,
    marker: str | None,
) -> None:
    row = _api_key(
        key_type="agent",
        capabilities=raw_capabilities,
        requests_per_minute=60,
        max_concurrency=4,
    )

    with pytest.raises(BusinessError) as exc_info:
        materialize_principal(row)

    error = exc_info.value
    assert error == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    assert error.__context__ is None
    assert error.__cause__ is None
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("rag_service"):
            locals_repr = repr(traceback.tb_frame.f_locals)
            if marker is not None:
                assert marker not in locals_repr
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    "capabilities",
    [
        [],
        ["ingest", "retrieve", "answer", "manage"],
    ],
)
def test_agent_row_accepts_bounded_unique_builtin_capability_strings(
    capabilities: list[str],
) -> None:
    row = _api_key(
        key_type="agent",
        capabilities=capabilities,
        requests_per_minute=60,
        max_concurrency=4,
    )

    principal = materialize_principal(row)

    assert isinstance(principal, AgentPrincipal)
    assert principal.capabilities == frozenset(Capability(value) for value in capabilities)


@pytest.mark.parametrize(
    "key_type",
    [KeyKind.ADMIN, KeyKind.AGENT, KeyTypeStringSubclass("admin"), KeyTypeStringSubclass("agent")],
)
def test_materialization_requires_exact_builtin_key_type(key_type: object) -> None:
    row = _api_key(key_type=key_type, capabilities=[])  # type: ignore[arg-type]

    with pytest.raises(BusinessError) as exc_info:
        materialize_principal(row)

    assert exc_info.value == BusinessError(500, "INTERNAL_ERROR", "Internal server error")


@pytest.mark.parametrize(
    "row",
    [
        _api_key(
            key_type="agent",
            capabilities=["unknown-sensitive-capability"],
            requests_per_minute=60,
            max_concurrency=4,
        ),
        _api_key(
            key_type="agent",
            capabilities=["retrieve"],
            requests_per_minute=None,
            max_concurrency=4,
        ),
        _api_key(key_type="unexpected-sensitive-kind"),
        _api_key(
            key_type="admin",
            public_id="noncanonical-sensitive-id",
            capabilities=[],
        ),
        _api_key(
            key_type="admin",
            public_id="unsafe-sensitive-id\n",
            capabilities=[],
        ),
        _api_key(
            key_type="agent",
            capabilities=["retrieve"],
            requests_per_minute=10_001,
            max_concurrency=4,
        ),
        _api_key(
            key_type="agent",
            capabilities=["retrieve"],
            requests_per_minute=60,
            max_concurrency=1_001,
        ),
    ],
)
def test_materialization_rejects_malformed_database_policy_with_safe_error(row: ApiKey) -> None:
    with pytest.raises(BusinessError) as exc_info:
        materialize_principal(row)

    assert exc_info.value == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    assert "sensitive" not in str(exc_info.value)


def test_materialization_rejects_default_query_profile_outside_explicit_scope() -> None:
    row = _api_key(
        key_type="agent",
        capabilities=["retrieve"],
        requests_per_minute=60,
        max_concurrency=4,
    )

    with pytest.raises(BusinessError) as exc_info:
        materialize_principal(
            row,
            query_profile_ids=frozenset(),
            default_query_profile_id=QUERY_PROFILE_ID,
        )

    assert exc_info.value == BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def test_empty_scope_denies_every_kb_and_new_kbs_are_not_implicitly_visible() -> None:
    empty_scope = _agent(Capability.RETRIEVE, knowledge_base_ids=frozenset())
    existing_scope = _agent(Capability.RETRIEVE)

    for principal, knowledge_base_id in (
        (empty_scope, KB_ID),
        (empty_scope, NEW_KB_ID),
        (existing_scope, NEW_KB_ID),
    ):
        with pytest.raises(BusinessError) as exc_info:
            require_knowledge_base_access(
                principal,
                knowledge_base_id,
                resource_exists=True,
            )
        assert exc_info.value == BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")

    assert (
        require_knowledge_base_access(existing_scope, KB_ID, resource_exists=True) is existing_scope
    )


def test_missing_and_out_of_scope_kb_resources_have_identical_non_enumerating_failure() -> None:
    principal = _agent(Capability.RETRIEVE)

    with pytest.raises(BusinessError) as missing:
        require_knowledge_base_access(principal, KB_ID, resource_exists=False)
    with pytest.raises(BusinessError) as out_of_scope:
        require_knowledge_base_access(principal, NEW_KB_ID, resource_exists=True)

    expected = BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
    assert missing.value == expected
    assert out_of_scope.value == expected
    assert str(KB_ID) not in str(missing.value)
    assert str(NEW_KB_ID) not in str(out_of_scope.value)


def test_admin_routes_require_admin_principal_and_manage_never_promotes_agent() -> None:
    admin = AdminPrincipal(key_id=KEY_ID, public_id=PUBLIC_ID)
    managing_agent = _agent(Capability.MANAGE)

    assert require_admin(admin) is admin
    with pytest.raises(BusinessError) as exc_info:
        require_admin(managing_agent)

    assert exc_info.value == BusinessError(
        403,
        "INSUFFICIENT_CAPABILITY",
        "Insufficient capability",
    )


def test_capability_policy_is_agent_specific_and_denies_missing_capability() -> None:
    retrieving_agent = _agent(Capability.RETRIEVE)
    admin = AdminPrincipal(key_id=KEY_ID, public_id=PUBLIC_ID)

    assert require_capability(retrieving_agent, Capability.RETRIEVE) is retrieving_agent
    for principal in (retrieving_agent, admin):
        with pytest.raises(BusinessError) as exc_info:
            require_capability(principal, Capability.ANSWER)
        assert exc_info.value == BusinessError(
            403,
            "INSUFFICIENT_CAPABILITY",
            "Insufficient capability",
        )


@pytest.mark.parametrize(
    "invalid_capability",
    [
        "retrieve",
        CapabilityStringSubclass("retrieve"),
        ForeignCapability.RETRIEVE,
    ],
)
def test_capability_policy_requires_exact_capability_type(invalid_capability: object) -> None:
    principal = _agent(Capability.RETRIEVE)

    with pytest.raises(BusinessError) as exc_info:
        require_capability(principal, invalid_capability)  # type: ignore[arg-type]

    assert exc_info.value == BusinessError(
        403,
        "INSUFFICIENT_CAPABILITY",
        "Insufficient capability",
    )


def test_invalid_capability_value_is_redacted_from_policy_traceback() -> None:
    marker = "sensitive-invalid-capability"
    principals: tuple[AdminPrincipal | AgentPrincipal, ...] = (
        _agent(Capability.RETRIEVE),
        AdminPrincipal(key_id=KEY_ID, public_id=PUBLIC_ID),
    )

    for principal in principals:
        with pytest.raises(BusinessError) as exc_info:
            require_capability(principal, marker)  # type: ignore[arg-type]

        traceback = exc_info.value.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            if module_name == "rag_service.auth.policies":
                assert marker not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next


def test_policy_helpers_reject_string_backed_principal_values() -> None:
    string_capability_agent = AgentPrincipal(
        key_id=KEY_ID,
        public_id=PUBLIC_ID,
        capabilities=frozenset({"retrieve"}),  # type: ignore[arg-type]
        knowledge_base_ids=frozenset({KB_ID}),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )
    enum_key_type_admin = AdminPrincipal(
        key_id=KEY_ID,
        public_id=PUBLIC_ID,
        key_type=KeyKind.ADMIN,  # type: ignore[arg-type]
    )
    enum_key_type_agent = AgentPrincipal(
        key_id=KEY_ID,
        public_id=PUBLIC_ID,
        capabilities=frozenset({Capability.RETRIEVE}),
        knowledge_base_ids=frozenset({KB_ID}),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
        key_type=KeyKind.AGENT,  # type: ignore[arg-type]
    )
    subclass_key_type_admin = AdminPrincipal(
        key_id=KEY_ID,
        public_id=PUBLIC_ID,
        key_type=KeyTypeStringSubclass("admin"),  # type: ignore[arg-type]
    )
    subclass_key_type_agent = AgentPrincipal(
        key_id=KEY_ID,
        public_id=PUBLIC_ID,
        capabilities=frozenset({Capability.RETRIEVE}),
        knowledge_base_ids=frozenset({KB_ID}),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
        key_type=KeyTypeStringSubclass("agent"),  # type: ignore[arg-type]
    )

    for operation in (
        lambda: require_capability(string_capability_agent, Capability.RETRIEVE),
        lambda: require_document_read(
            string_capability_agent,
            KB_ID,
            parent_knowledge_base_exists=True,
        ),
        lambda: require_admin(enum_key_type_admin),
        lambda: require_capability(enum_key_type_agent, Capability.RETRIEVE),
        lambda: require_admin(subclass_key_type_admin),
        lambda: require_capability(subclass_key_type_agent, Capability.RETRIEVE),
    ):
        with pytest.raises(BusinessError) as exc_info:
            operation()
        assert exc_info.value == BusinessError(
            403,
            "INSUFFICIENT_CAPABILITY",
            "Insufficient capability",
        )


@pytest.mark.parametrize("resource_exists", ["false", 1, object()])
def test_kb_access_requires_exact_true_resource_existence(resource_exists: object) -> None:
    with pytest.raises(BusinessError) as exc_info:
        require_knowledge_base_access(
            _agent(Capability.RETRIEVE),
            KB_ID,
            resource_exists=resource_exists,  # type: ignore[arg-type]
        )

    assert exc_info.value == BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


@pytest.mark.parametrize("parent_exists", ["false", 1, object()])
def test_document_read_requires_exact_true_parent_existence(parent_exists: object) -> None:
    with pytest.raises(BusinessError) as exc_info:
        require_document_read(
            _agent(Capability.RETRIEVE),
            KB_ID,
            parent_knowledge_base_exists=parent_exists,  # type: ignore[arg-type]
        )

    assert exc_info.value == BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


@pytest.mark.parametrize(
    "capability",
    [Capability.MANAGE, Capability.INGEST, Capability.RETRIEVE],
)
def test_document_read_accepts_any_allowed_capability_after_parent_scope_passes(
    capability: Capability,
) -> None:
    principal = _agent(capability)

    assert require_document_read(principal, KB_ID, parent_knowledge_base_exists=True) is principal


def test_document_read_checks_parent_existence_and_scope_before_capability() -> None:
    answer_only = _agent(Capability.ANSWER)

    for knowledge_base_id, exists in ((KB_ID, False), (NEW_KB_ID, True)):
        with pytest.raises(BusinessError) as hidden:
            require_document_read(
                answer_only,
                knowledge_base_id,
                parent_knowledge_base_exists=exists,
            )
        assert hidden.value == BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")

    with pytest.raises(BusinessError) as forbidden:
        require_document_read(answer_only, KB_ID, parent_knowledge_base_exists=True)
    assert forbidden.value == BusinessError(
        403,
        "INSUFFICIENT_CAPABILITY",
        "Insufficient capability",
    )


def test_document_read_does_not_treat_admin_as_an_agent_policy_bypass() -> None:
    admin = AdminPrincipal(key_id=KEY_ID, public_id=PUBLIC_ID)

    with pytest.raises(BusinessError) as exc_info:
        require_document_read(admin, KB_ID, parent_knowledge_base_exists=True)

    assert exc_info.value == BusinessError(
        403,
        "INSUFFICIENT_CAPABILITY",
        "Insufficient capability",
    )
