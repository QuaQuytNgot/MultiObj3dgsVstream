# Request Handler

`src/client/request_handler.py` executes resolved HTTP GET requests using a reusable
`curl_cffi.requests.AsyncSession`. The caller supplies the URL, optional byte
range, and request context. The handler reports response bytes, protocol, timing,
headers, and a structured outcome.

It does not parse MPD/XML or Gaussian/Draco payloads, select LoD, calculate
visibility or quality utility, render frames, predict bandwidth, schedule
priorities, or choose retries.

## Public API and module integration

Application modules should import `src.client.request_api`. It defines the
canonical `RequestSpec`, `TransferResult`, `TransferError`, `ErrorKind`, and
`ProtocolUnavailableError` using Python's standard library. The original imports
from `request_handler.py` remain compatible aliases to these same types.
Importing the public contracts does not import `curl_cffi`; constructing
`NetworkClient` loads and owns the low-level `RequestHandler` internally.

`TransferClient` is a runtime-checkable structural protocol with the minimum
interface `async fetch(spec: RequestSpec) -> TransferResult`. A scheduler can
depend on it without accessing sessions, curl constants, or backend response
objects. Alternative backends and test doubles can implement the same method.
Consumer error handling should use `error.kind` and `error.message`; the optional
`curl_code` is a legacy backend diagnostic, not a portable decision rule.

```python
from src.client.request_api import (
    NetworkClient,
    RequestSpec,
    TransferClient,
    TransferResult,
    to_bandwidth_sample,
)


async def execute_resolved_request(
    client: TransferClient, spec: RequestSpec
) -> TransferResult:
    return await client.fetch(spec)


async def example():
    samples = []  # Raw observations only; no estimator is implemented here.
    async with NetworkClient(http_version="h3", concurrency=1) as client:
        unsubscribe = client.subscribe(
            lambda result: samples.append(to_bandwidth_sample(result))
        )
        spec = RequestSpec(
            request_id="ld_q1_seg_0001",
            url="https://your-server.example/longdress/q1/seg_0001.bin",
            object_id="longdress",
            representation_id="q1",
            segment_id=1,
            metadata={"quality_profile_uri": "profiles/longdress-q1.json"},
        )
        result = await execute_resolved_request(client, spec)
        if result.success:
            payload_bytes = result.body
        unsubscribe()  # Idempotent; also safe after the client closes.
```

The facade provides these methods, all returning `TransferResult`:

| Method | Input and behavior |
| --- | --- |
| `fetch(spec, *, output_path=None)` | A fully resolved `RequestSpec`; optional atomic file output |
| `get(url, *, request_id=None, metadata=None)` | A resolved resource URL; memory output |
| `get_segment(url, *, object_id, representation_id, segment_id, ...)` | Segment context, optional `layer_id`, `byte_range`, `priority`, and metadata; memory output |
| `get_range(url, start, end, *, request_id=None, metadata=None)` | One inclusive byte range; memory output |
| `download_to_file(request, output_path, ...)` | A `RequestSpec` or resolved URL; streamed atomic file output |

Helpers generate a unique request ID when one is omitted. Explicit request IDs
are validated unchanged. With a supplied `RequestSpec`, set identity and metadata
on that spec rather than overriding them in `download_to_file`. All URL resolution,
representation choice, dependencies, and priority decisions remain with the
caller. The facade forwards protocol, concurrency, timeout, and TLS settings;
`async with` or `aclose()` closes the owned handler. Existing transport failure,
Range validation, timing, and cancellation behavior is preserved.

`to_bandwidth_sample(result)` copies exactly these nine fields into a frozen
`BandwidthSample`: `timestamp`, `received_bytes`, `ttfb`, `transfer_duration`,
`total_duration`, `body_goodput_bps`, `effective_goodput_bps`, `http_version`, and
`success`. It does not recompute metrics, estimate capacity, aggregate concurrent
requests, or filter failures. Missing measurements stay `None`; timestamp and
units retain the definitions below. The future estimator chooses which samples
to use.

`subscribe(callback)` supports sync and async callbacks and returns an
idempotent unsubscribe function. `on_transfer_complete=callback` in the constructor
registers an initial subscriber. Subscribers receive the complete result, so a
consumer can convert it to a bandwidth sample or inspect request context.
Each registration is independent. Delivery is sequential in registration order,
using a snapshot; subscription changes affect the next event. Ordinary subscriber
exceptions are logged and do not prevent other subscribers from receiving the
event or change the result. Task cancellation propagates. Subscriber work remains
outside network metrics, and the client does not import `bandwidth.py`.

A future PayloadParser can select its input using standard Python values:

```python
# The application supplies payload_parser; codec/parsing logic is not implemented.
if result.success:
    if result.body is not None:
        payload_parser.parse_bytes(result.body)
    elif result.output_path is not None:
        payload_parser.parse_file(result.output_path)
```

For file output, create the parent directory first and use
`await client.download_to_file(spec, "downloads/segment.bin")`. On success,
`body` is `None` and `output_path` identifies the published file; the parser can
open that file without loading a duplicate full payload. Always check `success`
before parsing. Failed results can contain partial or error response bytes.

The project currently assumes an existing **static HTTP server supporting H2/H3
and Range requests**. It serves offline-prepared binary assets, returns `206`
with a valid `Content-Range` for supported ranges, and provides consistent length
information. This phase adds no custom web server. The new public API tests use
an offline session double through the real handler; the existing HTTP/1.1 transport
tests remain separate from real H2/H3 negotiation checks.

## Low-level API

```python
import asyncio

from src.client.request_handler import RequestHandler, RequestSpec


async def main():
    async with RequestHandler(
        http_version="h3",
        concurrency=1,
        connect_timeout_s=5,
        total_timeout_s=30,
        verify_tls=True,
    ) as handler:
        print(handler.capabilities)
        result = await handler.fetch(
            RequestSpec(
                request_id="ld_q1_seg_0001",
                url="https://your-server.example/longdress/q1/seg_0001.bin",
                object_id="longdress",
                representation_id="q1",
                segment_id=1,
                metadata={"quality_profile_uri": "profiles/longdress-q1.json"},
            )
        )
        if result.success:
            print(result.received_bytes, result.body_goodput_bps)
            print(result.ttfb, result.http_version)
            payload = result.body
        else:
            print(result.error)


asyncio.run(main())
```

Replace the example URL with a server you control. The baseline setting is `h2`;
the example illustrates explicit `h3` configuration.

`RequestSpec` is a frozen dataclass. `request_id` and an absolute `http://` or
`https://` URL are required. Optional `object_id`, `representation_id`,
`segment_id`, `layer_id`, `priority`, and `metadata` retain caller context.
`priority` is informational; it does not reorder requests. `metadata` can carry
segment duration, expected size, dependency IDs, access-point context, or a
`quality_profile_uri`. These values do not change transport behavior.

The API provides:

- `await handler.fetch(spec, output_path=None)` for one transfer.
- `await handler.fetch_many(specs, concurrency=None)` for an input-ordered list of
  individual results.
- `handler.capabilities` for HTTP/2, HTTP/3, HTTP/3-only build availability and the
  underlying libcurl version.
- `async with RequestHandler(...)` and idempotent `await handler.aclose()` for
  session lifetime management.

Use a handler within one asyncio event loop.

Each `TransferResult` retains its `spec`, request identity and context, status,
lowercase response headers, requested range, received byte count, payload or
published file path, timing, negotiated HTTP version, and `success`/`error`.
Convenience fields expose `content_range`, `content_length`, and `accept_ranges`
when available. Treat `success` as the gate for passing a segment onward.

## HTTP/2 and HTTP/3

| `http_version` | libcurl setting | Behavior |
| --- | --- | --- |
| `h1` | `CurlHttpVersion.V1_1` | HTTP/1.1; useful for local test servers |
| `h2` | `CurlHttpVersion.V2_0` | Attempts HTTP/2; may negotiate HTTP/1.1 |
| `h3` | `CurlHttpVersion.V3` | Attempts HTTP/3; may negotiate an earlier version |
| `h3-only` | `CurlHttpVersion.V3ONLY` | Requires HTTP/3; no protocol downgrade |

These fallback rules follow
[libcurl's HTTP version options](https://curl.se/libcurl/c/CURLOPT_HTTP_VERSION.html).
`result.http_version` reports the protocol actually negotiated (`h1.0`, `h1`,
`h2`, or `h3`), so callers can
detect fallback. A request for `h3-only` that cannot obtain HTTP/3 fails.

Construction probes the installed libcurl's support for the requested HTTP
version without sending network traffic. Unsupported HTTP/2 or HTTP/3 builds
raise `ProtocolUnavailableError` with a diagnostic, including the libcurl version.
The `HttpVersion` string enum can be used instead of the shown string values.
Invalid protocol names raise `ValueError`. A positive capability probe establishes
build support, not server support, reachable UDP/QUIC, or successful negotiation.

For a manual protocol check, run the API example with a known HTTP/2 or HTTP/3
endpoint, then inspect `result.success`, `result.http_version`, and `result.error`.
Use `h3-only` when validating that an endpoint truly transfers over HTTP/3.
The local integration tests exercise HTTP/1.1 and do not claim to verify HTTP/2
or HTTP/3 network transfers.

## Range requests

```python
spec = RequestSpec(
    request_id="chunk-1",
    url="https://your-server.example/segment.bin",
    byte_range=(1000, 1999),
)
result = await handler.fetch(spec)
```

`byte_range` is a closed, inclusive pair of nonnegative integers with
`start <= end`. The example sends `Range: bytes=1000-1999` and requests 1000
bytes. Open-ended, suffix, and multiple ranges are outside this API.

A successful range transfer requires `206 Partial Content`, a consistent
`Content-Range`, and consistent payload length. The handler also validates
`Content-Length` when supplied and preserves `Content-Range`, `Content-Length`,
and `Accept-Ranges` in response headers. If `Content-Range` gives a known resource
length, a requested end beyond EOF may be shortened to the resource's last byte.
A server returning `200` to a range request records `range_honored=False` and
produces `success=False`, normally with `error.kind="range_ignored"`. A transport,
encoding, or length failure can remain the primary error. The response is not
reported as a successful partial fetch.

Requests ask for identity encoding, keeping byte ranges and byte accounting tied
to the representation payload. Automatic HTTP content decoding is disabled;
nonidentity response encoding is rejected with a protocol error.
`received_bytes` counts body bytes delivered to
the handler; it excludes response headers, TLS records, and protocol overhead.

## Memory and file output

Without `output_path`, `result.body` contains bytes. On failure this can be an
incomplete or error response; check `success` before using it.

```python
result = await handler.fetch(spec, output_path="downloads/segment.bin")
if result.success:
    print(result.output_path)  # published final file
    assert result.body is None
```

File output writes received chunks to a temporary `.part` file beside the final
path. After transfer and response validation succeed, an atomic rename publishes
the destination. A failure or cancellation removes the temporary file and does
not publish an incomplete destination. An existing destination is replaced only
on success. Create the destination directory before calling `fetch`.

File mode does not retain a duplicate full payload in memory. Small libcurl
chunks are written from a callback; local disk write time can therefore influence
the measured network transfer interval. Atomic publication does not promise
filesystem crash durability.

## Timing and throughput

Durations are seconds; goodput fields are bits per second. `start_time`,
`first_byte_time`, and `end_time` are monotonic estimates, anchoring native
durations to observed completion. They are unsuitable as wall-clock dates.
`result.timestamp` separately records Unix wall-clock seconds at result creation.
`timing_source` identifies `libcurl` measurements versus
`monotonic` fallback measurements when no native response timing is available.

For completed transfers, the handler uses libcurl's
[`STARTTRANSFER_TIME`](https://curl.se/libcurl/c/CURLINFO_STARTTRANSFER_TIME.html)
and [`TOTAL_TIME`](https://curl.se/libcurl/c/CURLINFO_TOTAL_TIME.html):

```text
ttfb                     = STARTTRANSFER_TIME
total_duration           = TOTAL_TIME
transfer_duration        = TOTAL_TIME - STARTTRANSFER_TIME
body_goodput_bps          = 8 * received_bytes / transfer_duration
effective_goodput_bps     = 8 * received_bytes / total_duration
first_byte_time           = start_time + ttfb
end_time                  = start_time + total_duration
```

TTFB is time to the first response byte observed by libcurl, normally a header
byte. It is not an exact timestamp for the first payload byte. Accordingly,
`transfer_duration` is the standard interval after that first response byte: it
can include remaining headers and a delay before the body, along with body
transfer. The `body_goodput_bps` numerator is body bytes, while its denominator is
this documented post-first-byte interval. It is an application goodput estimate,
not wire throughput or a pure first-body-to-last-body rate.

`effective_goodput_bps` uses the full native request interval, including connection
setup and TTFB. Scheduler/semaphore waiting, final file publication, completion
callbacks, decoding, and rendering are excluded. Disk writes performed inside the
libcurl callback remain part of the transfer. Zero or unavailable timing does not
produce infinite goodput; an undefined goodput value is `None`.

Failures retain bytes and timing observed before failure. A failure before any
response can have no first-byte measurement. Cancelled transfers may lack native
TTFB and response headers. A bandwidth estimator should choose
which outcomes and metric to admit into its own model.

Without a declared length or range, a correctly terminated HTTP body cannot
prove that an application segment contains every expected byte. The content
index or caller can perform additional expected-size or integrity checks.

## Timeouts, cancellation, and errors

`connect_timeout_s` limits connection establishment; `total_timeout_s` limits the
active request. Both must be positive. Queue waiting is outside the network
timeout. TLS verification is enabled by default.

HTTP errors, connection errors, timeouts, invalid ranges, incomplete responses,
and file write failures produce `TransferResult(success=False, error=...)`.
`TransferError.kind` provides a machine-readable category; its message provides
the diagnostic. Configuration and `RequestSpec` validation errors raise before
any network request. There are no automatic retries; a scheduler may decide
whether and when to issue a new request.

Redirect following is disabled. A `3xx` response is an HTTP failure; supply a
resolved segment URL, including any required signed URL parameters, at the
integration boundary.

Task cancellation propagates `asyncio.CancelledError` after releasing transport
resources and cleaning up temporary output. The completion hook receives a
cancelled result for an active transfer; the caller awaiting `fetch` receives the
cancellation exception. Closing the handler releases its reusable session.

## Concurrency and completion events

`concurrency=1` is the baseline. A handler-wide semaphore caps active transfers,
including direct `fetch` calls and multiple batches. The optional
`fetch_many(..., concurrency=N)` applies a per-batch limit and never raises the
handler-wide cap. Result ordering follows the input sequence even when requests
finish in a different order. This provides transport concurrency without a
priority or ABR scheduler.

Pass a sync or async `on_transfer_complete(result)` callback to observe each
completed transfer. It receives the same structured result and can forward it
to an external bandwidth estimator. The handler does not import `bandwidth.py`.
Callback work is outside transfer metrics. Callback exceptions are logged without
changing the transfer result. Logging uses Python `logging` and
reports one completion record with request context, URL/range, actual HTTP
version, status, bytes, TTFB, transfer duration, and goodput; chunk-level logging
is not enabled by default.

Concurrent request goodputs are individual observations. The handler does not
sum them into path capacity. The future estimator must account for overlapping
time intervals when constructing aggregate measurements.

## Integration boundary

The intended flow remains:

```text
MPD Parser -> ContentIndex -> Request Scheduler -> RequestSpec
                                                       |
                                           TransferClient / NetworkClient
                                                       |
                                                 RequestHandler
                                                       |
                                                 TransferResult
                                                   /       \
                                    BandwidthSample       body / output_path
                                           |                    |
                                  BandwidthEstimator       PayloadParser
```

The content index or scheduler resolves URLs, ranges, dependencies, and object
context. Offline quality-profile generation can attach a `quality_profile_uri`
through `metadata`; the handler does not download or interpret that profile
unless the caller explicitly supplies its URL as an ordinary request.

`configs/client.yaml` demonstrates constructor settings. Configuration loading is
the application's responsibility, keeping this block independent of YAML and
future MPD schemas. The session implementation uses the
[curl_cffi async API](https://curl-cffi.readthedocs.io/en/latest/api.html).
