import sys

from labtech.utils import OrderedSet, LoggerFileProxy


def test_OrderedSet():
    items = OrderedSet(['a', 'c', 'b', 'c', 'd'])

    assert 'a' in items
    items.add('e')
    assert 'e' in items
    items.remove('d')
    assert 'd' not in items

    assert list(items) == ['a', 'c', 'b', 'e']
    assert str(items) == '{a, c, b, e}'
    assert repr(items) == "{'a', 'c', 'b', 'e'}"
    assert len(items) == 4

    extra_items = OrderedSet(['a', 'f'])
    combined_items = items + extra_items
    assert combined_items is not items
    assert combined_items is not extra_items
    assert list(combined_items) == ['a', 'c', 'b', 'e', 'f']


def test_LoggerFileProxy(mocker):
    logger_func = mocker.Mock()
    mocker.patch('sys.stdout', LoggerFileProxy(logger_func, 'some_prefix:'))
    print('some_message')
    sys.stdout.flush()
    logger_func.assert_called_with('some_prefix:some_message')
