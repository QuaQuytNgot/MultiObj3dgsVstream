import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from multiobj3dgs.content.preparation.runner import StageRunner,StageSpec
from multiobj3dgs.content.preparation.calibration import blender_to_opencv,project_points,validate_reprojection,streamstgs_poses_bounds,validate_camera


def test_external_stage_pin_resume_and_mutation_detection(tmp_path):
    script=tmp_path/'trainer.py';source=tmp_path/'input.txt';output=tmp_path/'checkpoint.bin'
    script.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes()+sys.argv[3].encode())\n')
    source.write_bytes(b'original')
    pin=hashlib.sha256(script.read_bytes()).hexdigest()
    literal='literal; $(touch should-not-exist)'
    stage=StageSpec('train',(sys.executable,str(script),str(source),str(output),literal),tmp_path,(source,),(output,),pinned_inputs={str(script):pin},training=True)
    runner=StageRunner(tmp_path/'journal.json',resume=True)
    assert runner.run(stage,dry_run=True)['status']=='dry_run'
    assert not (tmp_path/'journal.json').exists() and not output.exists()
    assert runner.run(stage)['status']=='complete'
    assert output.read_bytes()==b'original'+literal.encode()
    assert not (tmp_path/'should-not-exist').exists()
    assert runner.run(stage)['status']=='reused'
    output.write_bytes(b'corrupted')
    with pytest.raises(FileExistsError):runner.run(stage)
    script.write_text('raise SystemExit(9)')
    with pytest.raises(ValueError,match='Pinned'):runner.run(stage)


def test_stage_failure_is_journaled_and_never_complete(tmp_path):
    stage=StageSpec('fail',(sys.executable,'-c','raise SystemExit(3)'),tmp_path,(),(tmp_path/'output',))
    runner=StageRunner(tmp_path/'journal.json')
    with pytest.raises(RuntimeError,match='exit 3'):runner.run(stage)
    journal=json.loads((tmp_path/'journal.json').read_text())
    assert journal['stages']['fail']['status']=='failed'
    assert journal['stages']['fail']['returncode']==3
    # Retrying a failed stage without partial outputs retains separate logs.
    with pytest.raises(RuntimeError):runner.run(stage)
    assert len(list(tmp_path.glob('fail-*.log')))==2


def test_training_stage_requires_pin_of_actual_entrypoint(tmp_path):
    with pytest.raises(ValueError,match='pinned'):StageSpec('train',(sys.executable,'train.py'),tmp_path,(),(tmp_path/'out',),training=True)
    with pytest.raises(ValueError,match='entrypoint'):StageSpec('train',(sys.executable,'train.py'),tmp_path,(),(tmp_path/'out',),pinned_inputs={str(tmp_path/'unrelated.py'):'0'*64},training=True)
    with pytest.raises(ValueError,match='list'):StageSpec.from_dict({'name':'bad','argv':'python train.py','outputs':['out']},tmp_path)


def test_dry_run_can_describe_downstream_stage_without_upstream_outputs(tmp_path):
    stage=StageSpec('future',(sys.executable,'-c','pass'),tmp_path,(tmp_path/'future-input',),(tmp_path/'future-out',))
    report=StageRunner(tmp_path/'journal').run(stage,dry_run=True)
    assert report['command']['input_hashes'][str(tmp_path/'future-input')]=='pending-upstream-stage'
    assert list(tmp_path.iterdir())==[]


def _camera():
    angle=.3;c,s=np.cos(angle),np.sin(angle)
    return {'convention':'opencv','width':640,'height':480,'K':[[500,0,320],[0,500,240],[0,0,1]],
        'w2c':[[c,0,s,.1],[0,1,0,-.2],[-s,0,c,.3],[0,0,0,1]]}


def test_calibration_reprojection_and_blender_axes():
    camera=_camera();k,w2c=validate_camera(camera)
    local=np.array([[0,0,2],[.2,.1,2],[-.1,.2,3]])
    world=(local-w2c[:3,3])@w2c[:3,:3]
    image=np.array([[320,240],[370,265],[320-50/3,240+100/3]])
    report=validate_reprojection(camera,world,image,tolerance_pixels=1e-9)
    assert report.max_error_pixels<1e-9
    with pytest.raises(ValueError,match='Reprojection'):validate_reprojection(camera,world,image+1,tolerance_pixels=.1)
    with pytest.raises(ValueError,match='front'):project_points(camera,(np.array([[0,0,-1]])-w2c[:3,3])@w2c[:3,:3])
    blender=np.linalg.inv(w2c)@np.diag([1,-1,-1,1])
    recovered=blender_to_opencv(blender,k,640,480)
    assert np.allclose(recovered['w2c'],w2c)


def test_poses_bounds_refuses_lossy_intrinsics_and_bad_bounds():
    camera=_camera();camera['K'][0][2]=319
    with pytest.raises(ValueError,match='centered'):streamstgs_poses_bounds([camera],[[.1,10]])
    with pytest.raises(ValueError,match='near/far'):streamstgs_poses_bounds([_camera()],[[10,.1]])
    camera=_camera();camera['K'][1][1]=501
    with pytest.raises(ValueError,match='square-pixel'):streamstgs_poses_bounds([camera],[[.1,10]])


def test_calibration_roundtrips_actual_pinned_loader_pose_code(tmp_path):
    """Execute only upstream's pose code via AST, avoiding CUDA/training imports."""
    repository=Path(__file__).resolve().parents[2]
    reference=repository/'third_party/streamstgs/scene/neural_3D_dataset_NDC.py'
    if not reference.exists():pytest.skip('Pinned inference source not available')
    manifest=json.loads((repository/'third_party/streamstgs/manifest.json').read_text())
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==manifest['files']['scene/neural_3D_dataset_NDC.py']
    source=ast.parse(reference.read_text())
    loader=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='Neural3D_NDC_Dataset')
    metadata=next(node for node in loader.body if isinstance(node,ast.FunctionDef) and node.name=='load_meta')
    pose_assignments=[node for node in metadata.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('poses_arr','poses')]
    images=next(node for node in loader.body if isinstance(node,ast.FunctionDef) and node.name=='load_images_path')
    camera=_camera();np.save(tmp_path/'poses_bounds.npy',streamstgs_poses_bounds([camera],[[.1,10]]))
    image_dir=tmp_path/'cam000/images';image_dir.mkdir(parents=True)
    for name in ('0000.png','0001.png'):(image_dir/name).write_bytes(b'not-read-in-pose-path')
    self=SimpleNamespace(root_dir=str(tmp_path),eval_index=0)
    namespace={'np':np,'os':os,'self':self}
    exec(compile(ast.Module(body=pose_assignments,type_ignores=[]),str(reference),'exec'),namespace)
    self.poses_all=namespace['poses']
    exec(compile(ast.Module(body=[images],type_ignores=[]),str(reference),'exec'),namespace)
    _,poses,times,_,_,_=namespace['load_images_path'](self,[str(tmp_path/'cam000.mp4')],'test',[0,2])
    stored_rotation,translation=poses[0]
    desired=np.array(camera['w2c'])
    assert np.allclose(stored_rotation.T,desired[:3,:3])
    assert np.allclose(translation,desired[:3,3])
    assert times==[0.,.5]
