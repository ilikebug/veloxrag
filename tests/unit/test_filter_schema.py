from collections.abc import Callable
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from rag_service.api.errors import BusinessError
from rag_service.metadata.schemas import (
    FilterSchemaField,
    FilterSchemaReplacement,
    SafeFilterSchema,
    canonicalize_filter_schema,
)


def _field(
    *,
    name: str = "department",
    source_path: str = "attributes.department",
    field_type: str = "keyword",
    operators: tuple[str, ...] = ("in", "eq"),
) -> dict[str, object]:
    return {
        "name": name,
        "source_path": source_path,
        "type": field_type,
        "operators": operators,
    }


def _uuid_factory(*values: str) -> Callable[[], UUID]:
    identifiers = iter(UUID(value) for value in values)
    return identifiers.__next__


class _ExplodingList(list[object]):
    iterated: bool = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iterated = True
        raise AssertionError("oversized-input-must-not-be-iterated")


def test_filter_schema_canonicalizes_operators_generates_internal_ids_and_retains_them() -> None:
    command = FilterSchemaReplacement.model_validate(
        {
            "fields": [
                _field(),
                _field(
                    name="createdAt",
                    source_path="system.created_at",
                    field_type="datetime",
                    operators=("lte", "eq", "gte"),
                ),
            ]
        }
    )
    stored = canonicalize_filter_schema(
        command,
        {"fields": []},
        id_factory=_uuid_factory(
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        ),
    )

    assert stored == {
        "fields": [
            {
                "name": "department",
                "source_path": "attributes.department",
                "type": "keyword",
                "operators": ["eq", "in"],
                "field_id": "fld_ERERERERQRGBEREREREREQ",
                "payload_path": "metadata.f_11111111111141118111111111111111",
            },
            {
                "name": "createdAt",
                "source_path": "system.created_at",
                "type": "datetime",
                "operators": ["eq", "gte", "lte"],
                "field_id": "fld_IiIiIiIiQiKCIiIiIiIiIg",
                "payload_path": "metadata.f_22222222222242228222222222222222",
            },
        ]
    }

    replacement = FilterSchemaReplacement.model_validate(
        {
            "fields": [
                _field(operators=("eq",)),
                _field(
                    name="priority",
                    source_path="attributes.priority",
                    field_type="integer",
                    operators=("lte", "gte", "eq"),
                ),
            ]
        }
    )
    replaced = canonicalize_filter_schema(
        replacement,
        stored,
        id_factory=_uuid_factory("33333333-3333-4333-8333-333333333333"),
    )
    stored_fields = cast(list[dict[str, object]], stored["fields"])
    assert replaced["fields"] == [
        {
            **stored_fields[0],
            "operators": ["eq"],
        },
        {
            "name": "priority",
            "source_path": "attributes.priority",
            "type": "integer",
            "operators": ["eq", "gte", "lte"],
            "field_id": "fld_MzMzMzMzQzODMzMzMzMzMw",
            "payload_path": "metadata.f_33333333333343338333333333333333",
        },
    ]

    safe = SafeFilterSchema(
        fields=replacement.fields,
        resource_revision=2,
        mutation_revision=1,
        filter_schema_revision=1,
        etag='"kb:11111111-1111-4111-8111-111111111111:r2"',
    )
    document = safe.model_dump(mode="json")
    assert set(document) == {
        "fields",
        "resource_revision",
        "mutation_revision",
        "filter_schema_revision",
        "etag",
    }
    assert all(
        set(field) == {"name", "source_path", "type", "operators"} for field in document["fields"]
    )


@pytest.mark.parametrize(
    ("field_type", "operators"),
    [
        ("keyword", ("eq", "in")),
        ("integer", ("eq", "gte", "in", "lte")),
        ("float", ("eq", "gte", "in", "lte")),
        ("boolean", ("eq", "in")),
        ("datetime", ("eq", "gte", "in", "lte")),
    ],
)
def test_filter_field_accepts_only_the_type_operator_matrix(
    field_type: str,
    operators: tuple[str, ...],
) -> None:
    field = FilterSchemaField.model_validate(
        _field(field_type=field_type, operators=tuple(reversed(operators)))
    )
    assert field.operators == tuple(sorted(operators))


@pytest.mark.parametrize(
    "raw_field",
    [
        _field(name=""),
        _field(name="1department"),
        _field(name="department-name"),
        _field(name="department\n"),
        _field(name="département"),
        _field(name="a" * 65),
        _field(source_path=""),
        _field(source_path=".department"),
        _field(source_path="department."),
        _field(source_path="one.two.three.four.five"),
        _field(source_path="one.two-bad"),
        _field(source_path="one.two\n"),
        _field(source_path="one.täg"),
        _field(source_path=f"root.{'a' * 65}"),
        _field(field_type="text"),
        _field(field_type="keyword", operators=("gte",)),
        _field(field_type="boolean", operators=("lte",)),
        _field(operators=("eq", "eq")),
        _field(operators=()),
    ],
)
def test_filter_field_rejects_invalid_names_paths_types_and_operators(
    raw_field: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        FilterSchemaField.model_validate(raw_field)


def test_filter_schema_rejects_duplicate_names_paths_too_many_fields_and_extra_input() -> None:
    duplicate_name = [_field(), _field(source_path="other.path")]
    duplicate_path = [_field(), _field(name="other")]
    too_many = [_field(name=f"field{index}", source_path=f"path{index}") for index in range(65)]
    for fields in (duplicate_name, duplicate_path, too_many):
        with pytest.raises(ValidationError):
            FilterSchemaReplacement.model_validate({"fields": fields})

    sentinel = "must-not-appear-in-validation-errors"
    for forbidden in (
        "field_id",
        "payload_path",
        "qdrant_path",
        "script",
        "regex",
        "expression",
    ):
        with pytest.raises(ValidationError) as invalid:
            FilterSchemaReplacement.model_validate({"fields": [{**_field(), forbidden: sentinel}]})
        assert sentinel not in str(invalid.value)

    with pytest.raises(ValidationError):
        FilterSchemaReplacement.model_validate(
            {"fields": [_field()], "payload_path": "metadata.forbidden"}
        )


def test_filter_schema_bounds_work_before_iterating_oversized_lists() -> None:
    oversized_operators = _field()
    operator_values = _ExplodingList(["eq", "in", "gte", "lte", "eq"])
    oversized_operators["operators"] = operator_values
    with pytest.raises(ValidationError) as operator_error:
        FilterSchemaField.model_validate(oversized_operators)
    assert operator_values.iterated is False
    assert "oversized-input-must-not-be-iterated" not in str(operator_error.value)

    oversized_fields = _ExplodingList(
        [_field(name=f"field{index}", source_path=f"path{index}") for index in range(65)]
    )
    with pytest.raises(ValidationError) as fields_error:
        FilterSchemaReplacement.model_validate({"fields": oversized_fields})
    assert oversized_fields.iterated is False
    assert "oversized-input-must-not-be-iterated" not in str(fields_error.value)


def test_filter_schema_accepts_exactly_64_fields_and_one_to_four_path_segments() -> None:
    fields = [_field(name=f"field{index}", source_path=f"root.path{index}") for index in range(64)]
    replacement = FilterSchemaReplacement.model_validate({"fields": fields})
    assert len(replacement.fields) == 64

    one_segment = FilterSchemaField.model_validate(_field(source_path="department"))
    four_segments = FilterSchemaField.model_validate(
        _field(name="fourSegments", source_path="one.two.three.four")
    )
    assert one_segment.source_path == "department"
    assert four_segments.source_path == "one.two.three.four"


@pytest.mark.parametrize(
    "stored_field",
    [
        {
            **_field(),
            "operators": ["eq", "in"],
            "field_id": "fld_ERERERERQRGBEREREREREQ",
            "payload_path": "metadata.f_22222222222242228222222222222222",
        },
        {
            **_field(),
            "operators": ["eq", "in"],
            "field_id": "fld_ERERERERQRGBERERERERER",
            "payload_path": "metadata.f_11111111111141118111111111111111",
        },
        {
            **_field(),
            "operators": ["eq", "in"],
            "field_id": "fld_AAAAAAAAAAAAAAAAAAAAAA",
            "payload_path": "metadata.f_00000000000000000000000000000000",
        },
    ],
)
def test_stored_filter_schema_rejects_mismatched_noncanonical_or_non_v4_identifier_pairs(
    stored_field: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        canonicalize_filter_schema(
            FilterSchemaReplacement(fields=()),
            {"fields": [stored_field]},
        )


def test_filter_schema_rejects_deterministic_internal_identifier_collisions() -> None:
    repeated_identifier = "11111111-1111-4111-8111-111111111111"
    with pytest.raises(ValidationError):
        canonicalize_filter_schema(
            FilterSchemaReplacement.model_validate(
                {
                    "fields": [
                        _field(),
                        _field(name="priority", source_path="attributes.priority"),
                    ]
                }
            ),
            {"fields": []},
            id_factory=_uuid_factory(repeated_identifier, repeated_identifier),
        )


@pytest.mark.parametrize(
    "changed",
    [
        _field(field_type="integer", operators=("eq",)),
        _field(source_path="attributes.renamed"),
    ],
)
def test_retained_logical_name_cannot_change_type_or_source_path(
    changed: dict[str, object],
) -> None:
    original = canonicalize_filter_schema(
        FilterSchemaReplacement.model_validate({"fields": [_field()]}),
        {"fields": []},
        id_factory=_uuid_factory("11111111-1111-4111-8111-111111111111"),
    )
    with pytest.raises(BusinessError) as invalid:
        canonicalize_filter_schema(
            FilterSchemaReplacement.model_validate({"fields": [changed]}),
            original,
        )
    assert (invalid.value.status_code, invalid.value.code) == (422, "VALIDATION_ERROR")
