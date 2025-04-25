## Specifying Ray remote function options

<!--
    This section is linked to from an error message in
    RayRunnerBackend.build_runner, so change the name with care.
-->

https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html

# TODO

* Only use distributed if you need it
* Handling Python dependencies across cluster: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
* Mention ray's special handling of numpy arrays
* For CPU/Memory, point users to the Metrics view on the dashboard, which requires Prometheus and Grafana: https://docs.ray.io/en/latest/ray-observability/getting-started.html#dash-metrics-view
* Log de-duplication
* Fault tolerance: It will not retry for app exceptions (unless you set retry_exceptions), but it will retry for:
  * Worker dying (you can stop by setting max_retries=0): https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html
  * Object loss (you can stop by setting max_retries=0): https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html
  * Out of memory (retries infinitely not respecting max_retries, but you can disable the memory monitor): https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html
* Use worker_process_setup_hook for mlflow or other setup
