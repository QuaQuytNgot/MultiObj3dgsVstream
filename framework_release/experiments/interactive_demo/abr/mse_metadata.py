"""Bind the prepared, validated MSE bank to verified original codec members."""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from .quality_metadata import MetadataError
from .validated.baseline_metadata import load_metadata
from .validated.common import digest, file_digest, read_json, require
from .validated.profiles import profiles_from_dict

DEFAULT_MSE_ROOT = Path(__file__).resolve().parents[3] / 'experiments/offline_profiles/runs/full_source_v1/prepared'
CRFS = (40, 36, 28, 20)
# Copied from offline_profiles.core.PROTOCOL; no native/offline profiler import.
PROTOCOL = dict(resolution=[640, 480], camera_ids=[0, 1, 6, 7, 12, 13, 18, 19],
    local_timestamps_s=[0., .5, 1., 1.5], alpha_threshold=.01, reference_crf=20,
    contribution_crf=40, rgb_normalization='clip', background=[0, 0, 0],
    sse_aggregation='equal_camera_mean_of_four_timestamp_summed_solo_SSE',
    mse_aggregation='pooled_SSE_over_foreground_RGB_scalars',
    alpha_aggregation='equal_camera_time_mean_continuous_alpha_weighted_pixels',
    object_yaw_degrees=dict(longdress=-15., soldier=0., loot=15., redandblack=0.),
    source_conditioning='recorded_center', gop_duration_s=2.)


class MSEBank:
    """Reuse copied strict experiment schema/reference validation, then bind media.

    Original package resource costs provide provenance. Runtime CPSEG closure
    sizes and publication revision are deliberately not compared to that format.
    """
    def __init__(self, location, quality):
        self.root = Path(location or DEFAULT_MSE_ROOT).expanduser().resolve()
        if self.root.is_file():
            self.root = self.root.parent
        try:
            self._load(quality)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise MetadataError(f'Incompatible MSE bank at {self.root}: {error}. '
                                'Use --mse-profiles with the complete validated prepared bank and validation.json.') from error

    def _load(self, quality):
        profile_path = self.root / 'profiles.json'
        baseline_path = self.root / 'baseline_metadata.json'
        data = read_json(profile_path)
        baseline = read_json(baseline_path)
        profiles = profiles_from_dict(data)
        profile_sha = file_digest(profile_path)
        metadata = load_metadata(baseline, profiles, profile_sha,
                                 'validated_cached_artifacts', ('VISIBILITY_MSE',))
        require(baseline['profile_revision'] == profiles.profile_revision,
                'baseline/profile revision mismatch')
        require(data['provenance'].get('protocol') == PROTOCOL, 'unsupported MSE preparation protocol')
        require(profiles.compatibility['distribution_id'] == digest(PROTOCOL),
                'MSE protocol/distribution checksum mismatch')
        validation_path = next((path for path in (self.root / 'validation.json', self.root.parent / 'validation.json')
                                if path.is_file()), None)
        require(validation_path is not None, 'saved preparation validation.json is required')
        validation = read_json(validation_path)
        require(validation.get('status') == 'PASS' and
                validation.get('content_revision') == profiles.content_revision and
                validation.get('distribution_id') == profiles.compatibility['distribution_id'],
                'saved validation/content/protocol binding mismatch')
        saved_hashes = {Path(name).name: sha for name, sha in validation.get('outputs_sha256', {}).items()}
        for path in (profile_path, baseline_path, self.root / 'bounds.json'):
            require(saved_hashes.get(path.name) == file_digest(path),
                    f'prepared checksum differs from saved validation: {path.name}')
        bounds = read_json(self.root / 'bounds.json')
        values = {}
        for row in profiles.objects.values():
            gop = row.source_gop_index
            require(gop is not None and row.instance_id == row.object_id,
                    'prepared bank must bind content identity and source GOP')
            require(tuple(q.quality_id for q in row.qualities) == tuple(f'CRF{crf}' for crf in CRFS),
                    'MSE candidate order must be CRF40, CRF36, CRF28, CRF20')
            for crf, candidate in zip(CRFS, row.qualities):
                key = row.object_id, gop, crf
                require(candidate.representation_id == f'{row.object_id}/{gop}/{crf}',
                        f'MSE representation identity mismatch: {key}')
                original = quality._manifests.get(key)
                require(original is not None, f'MSE content/GOP/CRF has no measured media manifest: {key}')
                measured = quality.profile(*key)
                components = {**original['files'], 'manifest.json': {
                    'bytes': (quality.metadata_root / row.object_id / 'content' / f'gop_{gop:03d}_crf_{crf}' / 'manifest.json').stat().st_size,
                    'sha256': measured.representation_manifest_sha256}}
                supplied = {}
                for resource_id in candidate.resource_ids:
                    resource = profiles.resources[resource_id]
                    name = Path(resource.url).name
                    require(name not in supplied, f'duplicate MSE media member: {key}/{name}')
                    supplied[name] = {'bytes': resource.bytes, 'sha256': resource.sha256}
                require(supplied == components, f'MSE/measured original media closure mismatch: {key}')
                values[key] = metadata['objects'][row.instance_id, row.gop_id]['qualities'][candidate.quality_id]['mse']
            bound = bounds.get(f'{row.object_id}/{gop}', {})
            reference_key = row.object_id, gop, 20
            reconstruction_path = (quality.metadata_root / row.object_id / 'content' /
                                   f'gop_{gop:03d}_crf_20' / 'reconstruction.json')
            require(bound.get('reference_crf') == 20 and
                    bound.get('manifest_sha256') == quality.profile(*reference_key).representation_manifest_sha256 and
                    bound.get('reconstruction_sha256') == file_digest(reconstruction_path) and
                    bound.get('bounds') == baseline_bounds(baseline, row.object_id, row.gop_id),
                    f'placement bounds/media binding mismatch: {reference_key}')
        for content in quality.catalog.contents.values():
            for rep in content.representations.values():
                for segment in rep.segments:
                    require((content.id, segment.gop_index, rep.crf) in values,
                            f'missing measured MSE: {content.id}/{segment.gop_index}/{rep.crf}')
        self.values = MappingProxyType(values)
        self.bounds = MappingProxyType(bounds)
        self.audit = MappingProxyType(dict(profile_count=len(values), profile_sha256=profile_sha,
            baseline_sha256=file_digest(baseline_path), content_revision=profiles.content_revision,
            distribution_id=profiles.compatibility['distribution_id'],
            quality_reference=metadata['quality_references']['mse'], validation_path=str(validation_path),
            policy='fixed own-CRF20 foreground RGB pooled MSE; exact original manifest/member media binding'))

    def mse(self, content_id, gop_index, crf):
        try:
            return self.values[content_id, gop_index, crf]
        except KeyError as error:
            raise MetadataError(f'No measured MSE for {content_id}/gop={gop_index}/crf={crf}') from error


def baseline_bounds(data, content, gop_id):
    for row in data['objects']:
        if row['instance_id'] == content:
            for gop in row['gops']:
                if gop['gop_id'] == gop_id:
                    return gop['bounds']
    raise ValueError(f'missing MSE bounds: {content}/{gop_id}')
