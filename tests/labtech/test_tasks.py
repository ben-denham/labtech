from enum import Enum

import pytest
from frozendict import frozendict
from labtech.tasks import immutable_param_value


class _ExampleEnum(Enum):
    A = 1
    B = 2


class TestImmutableParamValue:
    def test_empty_list(self) -> None:
        assert immutable_param_value("hello", []) == ()

    def test_empty_dict(self) -> None:
        assert immutable_param_value("hello", {}) == frozendict()

    def test_list(self) -> None:
        assert immutable_param_value("hello", [1, 2, 3]) == (1, 2, 3)

    def test_dict(self) -> None:
        assert immutable_param_value("hello", {"a": 1, "b": 2}) == frozendict(
            {"hello.a": 1, "hello.b": 2}
        )

    def test_frozendict(self) -> None:
        assert immutable_param_value(
            "hello", frozendict({"a": 1, "b": 2})
        ) == frozendict({"hello.a": 1, "hello.b": 2})

    def test_tuple(self) -> None:
        assert immutable_param_value("hello", (1, 2, 3)) == (1, 2, 3)

    def test_nested_list(self) -> None:
        assert immutable_param_value("hello", [1, [2, 3], 4]) == (1, (2, 3), 4)

    def test_nested_dict(self) -> None:
        assert immutable_param_value("hello", {"a": 1, "b": {"c": 2}}) == frozendict(
            {"hello.a": 1, "hello.b": frozendict({"hello.b.c": 2})}
        )

    def test_multiple_nesting(self) -> None:
        assert immutable_param_value("hello", {"a": [1, {"b": 2}]}) == frozendict(
            {"hello.a": (1, frozendict({"hello.a.1.b": 2}))}
        )

    def test_nested_list_dict(self) -> None:
        assert immutable_param_value("hello", [1, {"a": 2}]) == (1, frozendict({"hello.1.a": 2}))

    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            False,
            "hello",
            "world",
            3.14,
            42,
            _ExampleEnum.A,
        ],
    )
    def test_scalar(self, value) -> None:
        assert immutable_param_value("hello", value) is value
