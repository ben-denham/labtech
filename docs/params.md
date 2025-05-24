To extend the types of data that can be used as Labtech parameters,
you can define a class that implements the
[`ParamHandler`][labtech.types.ParamHandler] protocol and decorate it
with the [`@param_handler`][labtech.param_handler] decorator. A full
example is given [in the cookbook](/cookbook#defining-a-custom-parameter-handler).

::: labtech.param_handler

::: labtech.types.ParamHandler

::: labtech.types.Serializer
