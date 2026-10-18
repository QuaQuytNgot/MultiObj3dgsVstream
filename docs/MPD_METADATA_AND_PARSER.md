# MPD metadata export and client parser

The exporter and client share the deterministic project MPD model in
`src/content_mpd.py`. `tools/content_preparation/mpd.py` keeps its existing
public export API and full payload-validation default. The client imports only
the pure model, metadata validation, and existing transport/planner contracts;
it does not import YAML, NumPy, CUDA, a renderer, or a decoder.

This is the experimental Gaussian/CPSEG profile, not ordinary DASH video
playback. The supported client format is writer 1.2.0, content-index v1 with
explicit revision and profile capabilities, one static Period, explicit
SegmentLists/SegmentTimelines, and one to four progressive quality targets.
Older exports must be regenerated. The exporter can still describe more layers,
but the current planner contracts accept only Q0–Q3.

## Authoritative sources

| Information | Authority | Consistency checks |
|---|---|---|
| Object discovery and order | Prepared-root `manifest.json` | Object-manifest bytes/SHA-256, unique object identities |
| Object identity, media frames/timestamps, duration, canonical/frame transforms, quality order, provenance | Object `manifest.json` | Embedded package equals `package_index.json`; frame/time and representation consistency |
| Payload dependencies, epochs, parent/decoded-state hashes, codec/version | `package_index.json` payload descriptors | Existing structural/package validator; same-frame preceding-quality parent and state hash |
| Requestable segment paths, full container bytes, SHA-256, member offsets and lengths | `package_index.json` segment/member descriptors | Complete header+payload accounting, coverage and boundaries; sizes are not inferred from bitrate |
| MSE samples, view/scale/time coordinates, decoded reference quality, renderer settings | Manifest-referenced `profiles/profile.json` | Manifest asset bytes/SHA-256, finite nonnegative MSE, consistent complete per-context quality rows |
| Normalized density, Q0 solo areas and enrichment provenance | Explicitly selected density JSON | Source-profile hash and original rows, Q0 state hashes, formula, common denominator, renderer/transform/revision, actual coverage |
| Published XML and convenient rate/timeline/profile summaries | Derived MPD and `content_index.json` | Rebuilt using the shared model from verified source JSON, compared to published XML/index |

`package.json` duplicates the package index in the inspected run, and the object
manifest embeds that index. The loader requires the embedded copy and
`package_index.json` to agree. It does not depend on the pipeline's `package.json`,
`encoding.json`, `decoding.json`, training logs, or precomputed decoded files.
They remain offline audit evidence. Hash verification here means verification
of the downloaded metadata bytes, not authentication of a server or inspection
of the media payloads.

`content_revision` is SHA-256 of the canonical ordered list of object identity,
object-manifest hash, package-index hash, MSE-profile hash and optional
density-profile hash. It is independent of deployment URLs. Adding/changing a
density publication changes the revision. A cache/retained-state adapter must
use that revision and verify actual media/state identities; parsing never
manufactures an `AvailabilitySnapshot` or retained decoder state.

## Export

From the repository root, with the existing preparation Python environment:

```bash
python tools/export_content_mpd.py \
  --prepared-root dataset \
  --metadata-only \
  --density-profile longdress=longdress/profiles/profile.density-subset.json \
  --resume
```

`--density-profile OBJECT=PATH` is repeatable and explicitly opt-in. PATH is
relative to the prepared root. Omitting the option publishes MSE without a
density capability. No filename-based density discovery is performed.

The metadata-only path reads JSON and the pinned local XSD files. It never opens
CPSEG/CPGS, images, alpha maps, NPZ states or checkpoints, even when media are
present or being transferred. It checks each JSON source and writes only:

- `manifest.mpd`
- `content_index.json`
- `mpd_validation.json`
- The existing export lock/ownership/journal under `.preparation/`.

Original object/server manifests and profiles are not modified. Generated files
use the existing atomic writes and journal. Repeating a matching `--resume`
executes zero tasks. After intentional source/export changes, add `--overwrite`
to regenerate only this export's owned outputs. Unowned files and differently
owned output paths are refused, even with `--overwrite`.

The report distinguishes pinned offline XSD/metadata validation from payload
integrity: metadata-only reports `metadata_only: true`, `exact_files: false`,
and `payload_integrity: "NOT_RUN"`. XML/XSD success is not streaming readiness.
The default without `--metadata-only` still verifies actual prepared payloads
and assets. It should not be used on a metadata-only copy. Neither mode decodes
or renders, but the full mode reads/hashes payload bytes.

For Soldier, point at its **server root**, containing the server `manifest.json`
and the `soldier/` directory, not directly at the object directory:

```bash
python tools/export_content_mpd.py \
  --prepared-root output/content_preparation/soldier_full \
  --metadata-only --resume
```

If an actual compatible Soldier density sidecar exists, append, for example,
`--density-profile soldier=soldier/profiles/profile.density-subset.json`.
The example filename is not an assumption that this artifact exists.

The exporter retains the preparation environment's existing `fcntl` lock and
`xmllint` requirements. The parser and pure model do not depend on those tools;
local client metadata loading also supports Windows file URIs.

## XML and capability contract

Descriptors use the existing prefix
`urn:multiobj3dgsvstream:content-preparation:`. The existing `quality-profile`
descriptor is reused; two versioned SupplementalProperty descriptors add
`content-revision:1` at MPD level and optional `distortion-density:1` per object:

```xml
<SupplementalProperty
  schemeIdUri="urn:multiobj3dgsvstream:content-preparation:quality-profile"
  value="longdress/profiles/profile.json" />
<SupplementalProperty
  schemeIdUri="urn:multiobj3dgsvstream:content-preparation:distortion-density:1"
  value="longdress/profiles/profile.density-subset.json" />
```

The density descriptor locates a sidecar; **its presence does not imply complete
coverage**. `content_index.json` stores the bound profile bytes/hash and distinct
capabilities:

- MSE: `metric = full-image-rgb-mse`, decoded reference quality, full-image
  support, renderer settings, sampled frame/quality/context counts.
- Density: `metric = q0-normalized-distortion-density`, decoded Q3 RGB reference,
  decoded Q0 solo-area normalization, full-image support, schema/version/source
  hashes, resolution, explicit fraction and absolute area guards, exact context
  samples, `partial`, and `producer_parity = not_verified_by_metadata_export`.

The publication preserves the supplied density values and threshold. An area
fraction guard is converted to alpha-pixel units by multiplying by W×H; it is
not silently replaced with the planner's provisional default. Validating hashes
and scalar arithmetic does not establish saved-alpha/analyzer parity.

Progressive `dependencyId` signaling and codec/container EssentialProperty
descriptors are preserved. Qk targets the dependency closure Base through Ek;
the XML's Representation points to its own layer's containers. Exact remaining
bytes still belong to `RequirementsResolver`, including shared containers and
verified cache/retained-state credit. Segment boundaries may differ by layer.
For rates above uint32, the required extended-bandwidth EssentialProperty is
preserved/read; the saturated standard attribute is not an incremental cost.

## Parser API and integration

```python
from src.client.mpd_parser import MPDParser
from src.client.adaptation_inputs import (
    AvailabilitySnapshot, QualityTarget, RequirementsResolver,
)

parser = MPDParser.from_local(
    "dataset/manifest.mpd",
    media_base_url="https://your-content-server.example/dataset/",
)
index = parser.get_content_index()       # ContentIndexSnapshot
objects = parser.get_objects()          # ordered tuple[str, ...]
levels = parser.get_quality_levels("longdress")  # tuple[int, ...]
obj = parser.get_object_metadata("longdress")   # frozen source-derived mapping
rep = parser.get_representation("longdress", 3) # RepresentationMetadata
segment = parser.get_segment_info("longdress", 3, 0.0)  # SegmentInfo
request = segment.request_unit.request  # existing RequestSpec, whole container
profile_ref = parser.get_quality_profile_reference("longdress") # ProfileReference
density_ref = parser.get_density_profile_reference("longdress") # optional

# Cold snapshot for cost inspection only, not a claim that future Q0 is decoded.
available = AvailabilitySnapshot(index.content_revision, "cold")
required = RequirementsResolver(index).required_units(
    QualityTarget("longdress", (1051,), 3), available,
)
```

The local loader reads bounded MPD/JSON files and refuses paths/symlinks outside
the MPD directory. The media base URL is explicit because `RequestSpec` requires
HTTP(S); local metadata is never assigned a fictitious server automatically.
`get_segment_info` uses object presentation time with half-open intervals;
original media timestamps and frame IDs remain in the snapshot/object metadata.
It returns only the target layer's containing container, not its dependency cost.
`get_payload_location(payload_id)` returns the member's complete-container
offset, length and checksum for a future decoder adapter. No HTTP range request
is invented from a member offset.

For HTTP metadata, reuse the existing client and observer:

```python
from src.client.request_api import NetworkClient

async with NetworkClient() as client:
    parser = await MPDParser.from_http(
        "https://your-content-server.example/dataset/manifest.mpd", client,
    )
```

`from_http` calls the existing `TransferClient.fetch(RequestSpec)` only for
metadata. MPD/index/manifest requests have `traffic_kind=metadata`, profile
requests `traffic_kind=profile`, and both set `bandwidth_eligible=False`.
Actual bytes remain observable through the existing transfer observer. Media
request units carry `traffic_kind=base` or `enhancement` without changing the
bandwidth estimator. No curl/HTTP implementation is duplicated.

`parse_mpd(xml_bytes, source_uri=..., documents={absolute_uri: json_bytes, ...})`
is the pure supplied-document interface. A synchronous `metadata_loader` may
also be injected. Missing references fail explicitly. Each document is limited
to 32 MiB and a parsed bundle to 128 MiB. The existing transport owns transfer
buffering; fetched sizes are checked before parsing.

Quality metadata is available through `get_quality_profile(object_id,
density=False|True) -> QualityProfileMetadata | None`. Its exact sampled lookup
methods are:

```python
mse_profile = parser.get_quality_profile("longdress")
mse = mse_profile.lookup_mse(
    0, azimuth=-180, elevation=-20, scale=0.5, timestamp=0.0,
)
density_profile = parser.get_quality_profile("longdress", density=True)
if density_profile is not None:
    density = density_profile.lookup_density(
        0, azimuth=-180, elevation=-20, scale=0.5, timestamp=0.0,
    )  # existing DensityValue; missing context has value=None and a reason
```

Raw MSE cannot be queried as density. Missing density is never filled with MSE,
zero, a nearest context, or an interpolated value. This class is deliberately not
an automatically calibrated `DistortionDensityProvider`. A producer-validated
adapter must establish calibration, renderer/RGB conventions, matching fixed-Q0
visibility and content revision before passing a `DensityCapability` to the
planner. The synthetic integration test supplies those assumptions explicitly.
Utility, optimal allocation, visibility analysis and decoder/cache availability
remain in their existing components.

## Validation and remaining gates

The parser verifies strict JSON (including duplicate/nonfinite rejection), UTF-8
XML without DTD/entities, namespace/profile/extension versions, source hashes,
progressive graphs/parents, exact byte accounting, frame coverage and each
layer's timeline. It reconstructs expected index/XML semantics with the same
pure model used by the exporter, so modified XML descriptors, URLs, bandwidth,
timing or sidecar summaries cannot silently disagree with source JSON.
Namespace prefixes and a root BaseURL are supported; nested BaseURLs, multiple
Periods, templates and other DASH profiles are rejected explicitly. References
must use safe canonical relative paths or valid HTTP(S) URIs; metadata paths in
the source index must remain relative. Query/fragment/credential-bearing URIs
and encoded path traversal are not supported.

The inspected Longdress metadata has 6,048 MSE rows / 1,512 contexts and a
48-row / 12-context density subset. Those are integration-test expectations for
this run, not parser constants. Full producer density coverage/parity, real
media integrity/HTTP delivery, calibrated profile-to-camera lookup, a decoder
state adapter and Base-first playback orchestration remain independent gates.
No end-to-end streaming readiness follows from these metadata checks.

```bash
python -m pytest tests/test_mpd_parser.py tests/content_preparation/test_mpd.py \
  tests/test_adaptation_inputs.py tests/test_adaptation_integration.py \
  tests/test_adaptation_planner.py tests/test_bandwidth.py
```

Tests use tiny temporary synthetic packaging/JSON fixtures. They never load
large images, original media, NPZ states, training checkpoints, a decoder or GPU.
The optional real Longdress integration test reads JSON only. Exporter XSD tests
need existing `xmllint`; the parser tests do not.
