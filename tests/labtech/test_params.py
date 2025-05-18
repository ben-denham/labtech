import pytest

import labtech
from labtech.exceptions import ParamHandlerError
from labtech.params import get_param_handler_manager


class TestParamHandler:

    def teardown_method(self, method):
        get_param_handler_manager().clear()

    def test_register(self):

        @labtech.param_handler
        class FrozensetParamHandler:

            def handles(self, value):
                return isinstance(value, frozenset)

            def find_tasks(self, value, *, find_tasks_in_param):
                return [
                    task
                    for item in sorted(value, key=hash)
                    for task in find_tasks_in_param(item)
                ]

            def serialize(self, value, *, serializer):
                return list(sorted(value, key=hash))

            def deserialize(self, value, *, serializer):
                return frozenset(value)

        assert [type(handler) for handler in get_param_handler_manager().prioritised_handlers] == [
            FrozensetParamHandler,
        ]

    def test_register_priority(self):

        class FrozensetParamHandler:

            def handles(self, value):
                return isinstance(value, frozenset)

            def find_tasks(self, value, *, find_tasks_in_param):
                return [
                    task
                    for item in sorted(value, key=hash)
                    for task in find_tasks_in_param(item)
                ]

            def serialize(self, value, *, serializer):
                return list(sorted(value, key=hash))

            def deserialize(self, value, *, serializer):
                return frozenset(value)

        @labtech.param_handler(priority=2000)
        class FrozensetParamHandlerOne(FrozensetParamHandler):
            pass

        @labtech.param_handler
        class FrozensetParamHandlerTwo(FrozensetParamHandler):
            pass

        @labtech.param_handler
        class FrozensetParamHandlerThree(FrozensetParamHandler):
            pass

        @labtech.param_handler(priority=100)
        class FrozensetParamHandlerFour(FrozensetParamHandler):
            pass

        assert [type(handler) for handler in get_param_handler_manager().prioritised_handlers] == [
            FrozensetParamHandlerFour,
            FrozensetParamHandlerTwo,
            FrozensetParamHandlerThree,
            FrozensetParamHandlerOne,
        ]

    def test_register_noncompliant(self):
        with pytest.raises(
                ParamHandlerError, match=(
                    "Cannot register 'TestParamHandler.test_register_noncompliant.<locals>.CustomParamHandler' "
                    "as a custom parameter handler, as it does not implement all methods of the 'ParamHandler' protocol."
                ),
        ):
            @labtech.param_handler
            class CustomParamHandler:
                pass
