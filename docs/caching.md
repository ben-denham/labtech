## Caches

You can control how the results of a particular task type should be
formatted for caching by specifying an instance of one of the
following Cache classes for the `cache` argument of the
[`labtech.task`][labtech.task] decorator:

::: labtech.cache.PickleCache
    options:
        heading_level: 3

::: labtech.cache.NullCache
    options:
        heading_level: 3

### Custom Caches

You can define your own type of Cache with its own format or behaviour
by inheriting from [`BaseCache`][labtech.cache.BaseCache]:

::: labtech.cache.BaseCache
    options:
        heading_level: 4
        members: ['KEY_PREFIX', 'save_result', 'load_result']
        inherited_members: ['load_result']

## Storage

You can set the storage location for caching task results by
specifying an instance of one of the following Storage classes for the
`storage` argument of your [`Lab`][labtech.Lab]:

::: labtech.storage.LocalStorage
    options:
        heading_level: 3

::: labtech.storage.NullStorage
    options:
        heading_level: 3

### Custom Storage

To store cached results with an alternative storage provider (such as
a storage bucket in the cloud), you can define your own type of
Storage.

Many cloud storage providers can be implemented by inheriting from
[`FsspecStorage`][labtech.storage.FsspecStorage] and defining an
`fs_constructor()` method to return an
[`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/index.html)-compatible
filesystem:

::: labtech.storage.FsspecStorage
    options:
        heading_level: 4
        members: ['fs_constructor']

For other storage providers, inherit from
[`Storage`][labtech.storage.Storage]:

::: labtech.storage.Storage
    options:
        heading_level: 4
