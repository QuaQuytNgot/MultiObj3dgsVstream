"""Bind recovered baseline quality references to the common physical profiles."""
from __future__ import annotations

from .common import finite, require


def load_metadata(data, profiles, profile_file_sha256, input_kind, methods):
    require(data.get("schema") == "adaptation-baselines.metadata.v1", "Unsupported baseline metadata schema")
    require(data.get("input_kind") == input_kind, "Baseline metadata input_kind mismatch")
    require(data.get("content_revision") == profiles.content_revision, "Baseline/content revision mismatch")
    require(data.get("profile_binding_sha256") == profile_file_sha256, "Baseline metadata profile binding mismatch")
    require(isinstance(data.get("profile_revision"), str) and bool(data["profile_revision"]),
            "Baseline profile revision required")
    provenance = data.get("provenance", {})
    require(provenance.get("information_role") == "OFFLINE_BASELINE_METADATA" and
            provenance.get("joint_quality_used") is False, "Baseline metadata cannot use joint evaluation results")
    need_psnr = bool(set(methods) & {"NAVA", "VISIBILITY_PSNR"})
    need_mse = bool(set(methods) & {"VISIBILITY_MSE", "AABB_MSE_HISTORICAL"})
    need_bounds = bool(set(methods) & {"NAVA", "AABB_MSE_HISTORICAL"})
    references = data.get("quality_references", {})
    validate_quality_references(references, profiles, methods)
    objects = {}
    require(isinstance(data.get("objects"), list) and data["objects"], "Baseline object profiles required")
    for obj in data["objects"]:
        iid, oid = obj.get("instance_id"), obj.get("object_id")
        require(isinstance(iid, str) and isinstance(oid, str), "Baseline object/instance IDs required")
        for gop in obj.get("gops", []):
            gid = gop.get("gop_id")
            key = iid, gid
            require(key in profiles.objects and key not in objects, "Unknown or duplicate baseline instance/GOP")
            profile = profiles.objects[key]
            require(oid == profile.object_id, "Baseline object identity mismatch")
            bounds = gop.get("bounds")
            if need_bounds:
                require(isinstance(bounds, list) and len(bounds) == 2 and
                        all(isinstance(v, list) and len(v) == 3 for v in bounds),
                        "Source-local GOP AABB bounds required")
                bounds = [[finite(v, "GOP bound") for v in side] for side in bounds]
                require(all(a < b for a, b in zip(*bounds)), "GOP bounds must be strictly ordered")
            qualities = {}
            records = gop.get("qualities")
            require(isinstance(records, list), "Baseline quality profiles required")
            for quality in records:
                qid = quality.get("quality_id")
                require(isinstance(qid, str) and qid not in qualities, "Duplicate or invalid baseline quality ID")
                row = dict(quality)
                if need_psnr:
                    row["psnr_db"] = finite(row.get("psnr_db"), "PSNR utility")
                if need_mse:
                    row["mse"] = finite(row.get("mse"), "MSE quality")
                    require(row["mse"] >= 0, "MSE quality must be nonnegative")
                qualities[qid] = row
            require(set(qualities) == {q.quality_id for q in profile.qualities}, "Baseline quality coverage mismatch")
            require(all(qualities[q.quality_id].get("representation_id") == q.representation_id
                        for q in profile.qualities), "Baseline representation identity mismatch")
            objects[key] = {"bounds": bounds, "qualities": qualities}
    require(set(objects) == set(profiles.objects), "Baseline object/GOP coverage mismatch")
    return {"objects": objects, "quality_references": references, "profile_revision": data["profile_revision"],
            "provenance": provenance}


def validate_quality_references(references, profiles, methods):
    """Also used at the direct Python planner boundary, before constructing utility."""
    need_psnr = bool(set(methods) & {"NAVA", "VISIBILITY_PSNR"})
    need_mse = bool(set(methods) & {"VISIBILITY_MSE", "AABB_MSE_HISTORICAL"})
    if need_psnr:
        psnr = references.get("psnr", {})
        require(psnr.get("reference") == "original_ply_synthetic_rgb" and
                psnr.get("aggregation") == "arithmetic_mean_db",
                "PSNR requires finite original-PLY arithmetic mean dB; CRF20 self-reference is incompatible")
    if need_mse:
        mse = references.get("mse", {})
        require(mse.get("reference") == "own_CRF20_fixed_alpha_gt_0.01_clipped_RGB" and
                mse.get("aggregation") == "pooled_SSE_over_foreground_RGB_scalars" and
                mse.get("channels") == 3 and mse.get("data_range") == 1.0 and
                mse.get("alpha_threshold") == 0.01,
                "MSE requires fixed own-CRF20 foreground RGB pooling")
        require(mse.get("distribution_id") == profiles.compatibility["distribution_id"],
                "MSE and SSE offline camera/time distribution mismatch")
