"""Small wrappers around the existing preprocessing and train.py entry point."""
from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import numpy as np
from .config import ROOT
from .paths import UPSTREAM_ROOT, require_upstream, configure_upstream_imports
from .checkpoint import sha256, write_json


def source_snapshot():
    require_upstream()
    files = [UPSTREAM_ROOT / "train.py", UPSTREAM_ROOT / "train_full_pipeline.py", UPSTREAM_ROOT / "dataset_prepare.py"]
    for folder in ("gaussian_renderer", "scene", "arguments", "utils"):
        files += sorted((UPSTREAM_ROOT/folder).rglob("*.py"))
    for folder in ("submodules/diff-gaussian-rasterization","submodules/simple-knn"):
        files += sorted(p for p in (UPSTREAM_ROOT/folder).rglob("*") if p.is_file() and p.suffix in {".py",".cpp",".cu",".cuh",".h",".hpp"} and "third_party" not in p.relative_to(UPSTREAM_ROOT/folder).parts)
    return {str(p.relative_to(UPSTREAM_ROOT)): sha256(p) for p in files}


def runtime_provenance(config):
    import zlib
    import importlib.metadata
    value={"python":sys.version.split()[0],"numpy":np.__version__,"zlib":zlib.ZLIB_VERSION,"upstream_training_rng_seed":0,
           "cpu_threads":config["runtime"].get("cpu_threads",2),
           "initialization_rng_seed":config["runtime"]["seed"]}
    value["compatibility"] = {
        "adapter_sha256": sha256(Path(__file__).with_name("compatibility.py")),
        "launcher_sha256": sha256(Path(__file__).with_name("backend.py")),
        "nerf_rgb_dtype": "uint8",
        "open3d_constructor": "original body without obsolete headless keyword",
    }
    extension_value=config["runtime"].get("extension_path")
    extension=Path(extension_value) if extension_value else None
    if extension is not None and not extension.is_absolute(): extension=ROOT/extension
    value["extension_artifacts"]={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:sha256(p) for p in extension.rglob("*.so")} if extension is not None and extension.is_dir() else {}
    from .codec_adapter import get_codec
    codec_configs=[("default",config.get("encoding",{}))]
    codec_configs += [(f"{obj['id']}/{q['id']}",dict(config.get("encoding",{}),**q.get("encoding",{})))
                      for obj in config.get("objects",[]) for q in obj["qualities"]]
    value["codec_executables"]={}
    for label,settings in codec_configs:
        codec=get_codec(settings)
        executable=(codec.runtime_config["executable"] if codec.name=="gaussian_attribute_draco_byteplanes"
                    else settings.get("executable"))
        if executable:
            binary=Path(executable).expanduser()
            if not binary.is_absolute(): binary=ROOT/binary
            value["codec_executables"][label]={"path":str(binary.resolve()),"sha256":sha256(binary) if binary.is_file() else None}
    native=Path(__file__).parent/"native"
    value["codec_native_sources"]={p.name:sha256(p) for p in sorted(native.glob("*")) if p.is_file() and p.suffix in {".cpp",".h",".txt"}}
    repositories={"project":ROOT,"dynamic_lapisgs":UPSTREAM_ROOT,"draco":ROOT/"third_party/draco",
                  "rasterizer":UPSTREAM_ROOT/"submodules/diff-gaussian-rasterization",
                  "simple_knn":UPSTREAM_ROOT/"submodules/simple-knn"}
    value["repository_commits"]={}
    for name,path in repositories.items():
        result=subprocess.run(["git","-C",str(path),"rev-parse","HEAD"],capture_output=True,text=True)
        value["repository_commits"][name]=result.stdout.strip() if result.returncode==0 else "unavailable"
    for package in ("torch","torchvision","PyYAML","plyfile"):
        try: value[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: value[package]="unavailable"
    if "torch" in sys.modules:
        torch=sys.modules["torch"]
        value.update(torch=torch.__version__,cuda_runtime=torch.version.cuda)
        if torch.cuda.is_available(): value["gpu"]=torch.cuda.get_device_name(0)
    return value


def execution_environment(config):
    threads=str(config["runtime"].get("cpu_threads",2))
    env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, WANDB_MODE="offline", EGL_PLATFORM="surfaceless")
    extension = ROOT / config["runtime"]["extension_path"] if config["runtime"].get("extension_path") else None
    if extension is not None and not extension.is_dir():
        raise FileNotFoundError(f"Configured extension_path does not exist: {extension}")
    require_upstream()
    paths = ([str(extension)] if extension else []) + [str(ROOT), str(UPSTREAM_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    return env


def run_command(command, log, config):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        f.write(json.dumps(command) + "\n")
        f.flush()
        result = subprocess.run(command, cwd=ROOT, env=execution_environment(config), stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log.read_text()[-4000:]
        hint = " CUDA OOM: reduce configured render resolution or training initialization/densification budget explicitly; batch size remains 1." if "out of memory" in tail.lower() else ""
        raise RuntimeError(f"Subprocess failed (exit {result.returncode}): {log}.{hint}\n{tail}")


def build_training_command(source, model, quality, training, steps, previous=None, foundation=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    cmd = [sys.executable, str(ROOT/"tools/content_preparation/backend.py"), "--entry", str(require_upstream()/"train.py"), "-s", str(source), "-m", str(model), "--iterations", str(steps),
           "--lambda_dssim", str(training["lambda_dssim"]), "--sh_degree", str(training["sh_degree"]),
           "--eval", "--data_device", "cpu", "--save_iterations", str(steps), "--test_iterations", str(steps), "--port", port]
    # Keep training images on CPU; upstream moves one sampled image to CUDA per step.
    extra = list(training.get("extra_args", [])) + list(quality.get("training_args", []))
    reserved = {"-s", "--source_path", "-m", "--model_path", "--initial_gs_path", "--foundation_gs_path", "--dynamic_lapis", "--dynamic_opacity", "--iterations", "--save_iterations", "--test_iterations", "--sh_degree", "--start_checkpoint", "--data_device", "--port"}
    if previous:
        reserved |= {"--densify_from_iter", "--densify_until_iter", "--opacity_reset_interval"}
    def protected(arg):
        name=arg.split("=",1)[0]
        return name in reserved or (name.startswith("--") and any(flag.startswith(name) for flag in reserved if flag.startswith("--"))) or (name.startswith("-s") and not name.startswith("--")) or (name.startswith("-m") and not name.startswith("--"))
    if any(not isinstance(arg,str) or protected(arg) for arg in extra):
        raise ValueError("training_args cannot override orchestration, lineage, or sequential-memory parameters")
    cmd += extra
    if previous:
        cmd += ["--dynamic_lapis", "--initial_gs_path", str(previous), "--densify_from_iter", str(steps+1),
                "--densify_until_iter", "0", "--opacity_reset_interval", str(steps+1)]
    elif foundation:
        cmd += ["--dynamic_opacity", "--foundation_gs_path", str(foundation)]
    return cmd


def verify_checkpoint_lineage(manifest, qualities):
    """Prove row IDs from the unchanged upstream commands; never guess matching."""
    from .assets import read_ply
    rows = manifest["frames"]
    commands = {(c["frame"], c["level"]): c for c in manifest.get("commands", [])}
    if not rows or not commands:
        raise ValueError("Progressive checkpoint import requires training argv and checkpoint hashes")
    def option(cmd, name):
        return cmd[cmd.index(name)+1] if name in cmd else None
    for row in rows:
        for quality in qualities:
            record = commands.get((row["frame"], quality))
            if not record or sha256(row[quality]) != record["sha256"]:
                raise ValueError(f"Missing or changed training provenance for {quality}/{row['frame']}")
    first = rows[0]
    for lower, higher in zip(qualities, qualities[1:]):
        cmd = commands[(first["frame"],higher)]["argv"]
        if option(cmd,"--foundation_gs_path") is None or Path(option(cmd,"--foundation_gs_path")).resolve() != Path(first[lower]).resolve() or "--dynamic_opacity" not in cmd:
            raise ValueError("Progressive import lacks inherited foundation prefix provenance")
        a,b = read_ply(first[lower]), read_ply(first[higher])
        if len(b.ids) < len(a.ids):
            raise ValueError("Inherited prefix shrank")
        for name in ("xyz", "rotation", "scale", "sh"):
            if not np.array_equal(a.arrays[name], b.arrays[name][:len(a.ids)]):
                raise ValueError(f"Unverified inherited prefix {lower}->{higher}: {name}")
    for before, row in zip(rows, rows[1:]):
        for quality in qualities:
            cmd = commands[(row["frame"],quality)]["argv"]
            initial = option(cmd,"--initial_gs_path")
            steps = int(option(cmd,"--iterations") or 0)
            if initial is None or Path(initial).resolve() != Path(before[quality]).resolve() or "--dynamic_lapis" not in cmd or option(cmd,"--densify_until_iter") != "0" or int(option(cmd,"--opacity_reset_interval") or 0) <= steps:
                raise ValueError("Temporal import lacks stable upstream row lineage")
            a,b=read_ply(before[quality]),read_ply(row[quality])
            if len(a.ids) != len(b.ids):
                raise ValueError("Temporal row counts changed")
            for name in ("scale", "opacity", "sh"):
                if not np.array_equal(a.arrays[name],b.arrays[name]):
                    raise ValueError(f"Temporal static state changed: {quality}.{name}")
    return True


def initialize_prepared_points(path,seed):
    """Materialize upstream's default random PLY before hashing training inputs."""
    from plyfile import PlyData,PlyElement
    path=Path(path)
    if path.exists(): return
    rng=np.random.RandomState(seed)
    xyz=rng.random_sample((100000,3))*2.6-1.3
    shs=rng.random_sample((100000,3))/255.
    rgb=(shs*.28209479177387814+.5)*255
    data=np.empty(len(xyz),dtype=[(n,"f4") for n in ("x","y","z","nx","ny","nz")]+[(n,"u1") for n in ("red","green","blue")])
    for i,name in enumerate(("x","y","z")): data[name]=xyz[:,i]
    for name in ("nx","ny","nz"): data[name]=0
    for i,name in enumerate(("red","green","blue")): data[name]=rgb[:,i]
    path.parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(data,"vertex")]).write(path)


def preprocess_worker(raw_path, dest, poses, scales, width, seed):
    """Reuse upstream normalization/rendering and resize, one frame/split at a time."""
    from .compatibility import render_2d_image, rescale_image, store_point_cloud
    configure_upstream_imports()
    from utils.sh_utils import SH2RGB
    from PIL import Image
    import open3d as o3d
    raw_path, dest = Path(raw_path), Path(dest)
    cloud = o3d.io.read_point_cloud(str(raw_path)).voxel_down_sample(voxel_size=1.7)
    center = np.asarray(cloud.get_axis_aligned_bounding_box().get_center())
    divisor = 960. if "thaidancer" in str(raw_path) else 480.
    rotation = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
    translate = rotation @ (-center/divisor - np.array([0,.1,0]))
    canonical = {"normalization": "upstream per-frame AABB centering", "raw_center": center.tolist(), "scale": 1/divisor,
                 "matrix": np.block([[rotation/divisor,translate[:,None]], [np.zeros((1,3)),np.ones((1,1))]]).tolist(),
                 "raw_sha256": sha256(raw_path), "voxel_size": 1.7, "point_size": 2}
    for split, pose in poses.items():
        render_2d_image(str(raw_path), str(dest/"res1"/split), str(pose), pt_size=2, width=width, height=width)
        meta = json.loads(Path(pose).read_text())
        meta["frames"] = [dict(f, file_path=f"./{split}/r_{i}") for i,f in enumerate(meta["frames"])]
        for scale in sorted(set(scales) | {1}):
            target=dest/f"res{scale}"
            if scale != 1:
                (target/split).mkdir(parents=True,exist_ok=True)
                for image in sorted((dest/"res1"/split).glob("*.png")):
                    rescale_image(str(image),scale).save(target/split/image.name)
            write_json(target/f"transforms_{split}.json",meta)
    # Same initialization distribution as upstream, explicitly seeded before training.
    rng=np.random.RandomState(seed)
    xyz=rng.random_sample((100000,3))*2.6-1.3
    shs=rng.random_sample((100000,3))/255.
    for scale in sorted(set(scales)|{1}):
        store_point_cloud(str(dest/f"res{scale}"/"points3d.ply"),xyz,SH2RGB(shs)*255)
    write_json(dest/"canonical.json",canonical)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--preprocess-spec",required=True)
    spec=json.loads(Path(parser.parse_args().preprocess_spec).read_text())
    preprocess_worker(**spec)
