import pytest

from labtech.exceptions import SerializationError
from labtech.serialization import Serializer


class TestSerializer:

    class TestSerializeClass:

        def test_local_object(self):

            class Example:
                pass

            serializer = Serializer()
            with pytest.raises(SerializationError, match=f'Unable to serialize class "{Example.__qualname__}" because it was defined in a function.'):
                serializer.serialize_class(Example)
