from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from labtech.utils import LoggerFileProxy, OrderedSet

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestOrderedSet:
    def test_in(self) -> None:
        items = OrderedSet(['a', 'c', 'b', 'c', 'd'])
        assert 'a' in items

    def test_add(self) -> None:
        items = OrderedSet(['a', 'c', 'b', 'c', 'd'])
        items.add('e')
        assert 'e' in items
        assert list(items) == ['a', 'c', 'b', 'd', 'e']

    def test_remove(self) -> None:
        items = OrderedSet(['a', 'c', 'b', 'd', 'e'])
        items.remove('d')
        assert 'd' not in items
        assert len(items) == 4
        assert list(items) == ['a', 'c', 'b', 'e']

    def test_str(self) -> None:
        items = OrderedSet(['a', 'c', 'b', 'e'])
        assert str(items) == '{a, c, b, e}'

    def test_repr(self) -> None:
        items = OrderedSet(['a', 'c', 'b', 'e'])
        assert repr(items) == "{'a', 'c', 'b', 'e'}"

    def test_concat(self) -> None:
        items = OrderedSet(['a', 'c', 'b', 'e'])
        extra_items = OrderedSet(['a', 'f'])
        combined_items = items + extra_items
        assert combined_items is not items
        assert combined_items is not extra_items
        assert list(combined_items) == ['a', 'c', 'b', 'e', 'f']


class TestLoggerFileProxy:
    def test_prefix_added(self, mocker: MockerFixture) -> None:
        logger_func = mocker.Mock()
        mocker.patch('sys.stdout', LoggerFileProxy(logger_func, 'some_prefix:'))
        print('some_message')
        sys.stdout.flush()
        logger_func.assert_called_with('some_prefix:some_message')
