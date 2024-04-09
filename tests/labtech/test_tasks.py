import re
from enum import Enum

import pandas as pd
import pytest
from frozendict import frozendict
from labtech.exceptions import TaskError
from labtech.tasks import immutable_param_value


class _BadObject:
    """An object which is not supported by `immutable_param_value`."""

    pass


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
            {"a": 1, "b": 2}
        )

    def test_frozendict(self) -> None:
        assert immutable_param_value(
            "hello", frozendict({"a": 1, "b": 2})
        ) == frozendict({"a": 1, "b": 2})

    def test_tuple(self) -> None:
        assert immutable_param_value("hello", (1, 2, 3)) == (1, 2, 3)

    def test_nested_list(self) -> None:
        assert immutable_param_value("hello", [1, [2, 3], 4]) == (1, (2, 3), 4)

    def test_nested_dict(self) -> None:
        assert immutable_param_value("hello", {"a": 1, "b": {"c": 2}}) == frozendict(
            {"a": 1, "b": frozendict({"c": 2})}
        )

    def test_multiple_nesting(self) -> None:
        assert immutable_param_value("hello", {"a": [1, {"b": 2}]}) == frozendict(
            {"a": (1, frozendict({"b": 2}))}
        )

    def test_nested_list_dict(self) -> None:
        assert immutable_param_value("hello", [1, {"a": 2}]) == (
            1,
            frozendict({"a": 2}),
        )

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

    def test_multiple_nested_error(self) -> None:
        match = re.escape(
            """Unsupported type '_BadObject' in parameter value 'hello["b"][2]["c"]'."""
        )
        with pytest.raises(TaskError, match=match):
            immutable_param_value("hello", {"a": 1, "b": (1, 2, {"c": _BadObject()})})
