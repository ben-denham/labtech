## Task Runner Backends

You can control how tasks are executed in parallel by specifying an
instance of one of the following Runner Backend classes for the
`runner_backend` argument of your [`Lab`][labtech.Lab]:

::: labtech.runners.ForkRunnerBackend
    options:
        heading_level: 3
        show_source: False

::: labtech.runners.SpawnRunnerBackend
    options:
        heading_level: 3
        show_source: False

::: labtech.runners.ThreadRunnerBackend
    options:
        heading_level: 3
        show_source: False

::: labtech.runners.SerialRunnerBackend
    options:
        heading_level: 3
        show_source: False


### Distributed Task Runner Backends

Labtech offers support for running tasks across multiple machines.
See: [Multi-Machine Clusters](./distributed.md)


### Custom Task Runner Backends

You can define your own Runner Backend to execute tasks with a
different form of parallelism or distributed computing platform by
defining an implementation of the
[`RunnerBackend`][labtech.types.RunnerBackend] abstract base class:

::: labtech.types.RunnerBackend
    options:
        heading_level: 4

::: labtech.types.Runner
    options:
        heading_level: 4
