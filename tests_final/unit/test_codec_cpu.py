from concurrent.futures import CancelledError
import threading
import numpy as np
import pytest
from multiobj3dgs.codecs.streamstgs.adapter import unpack_features,check_cancel,safe_file
from multiobj3dgs.codecs.streamstgs.vendor import verify_vendor


@pytest.mark.parametrize('columns', [3, 4])
def test_video_roundtrip_preserves_channel_and_gaussian_identity(tmp_path, columns):
    """Known features -> row-major atlas -> actual lossless video -> decoder.

    Spatially varying values detect a tile transpose even if dimensions fit.
    This does not use the historical decoder as an oracle.
    """
    import shutil
    import subprocess
    from multiobj3dgs.codecs.streamstgs.adapter import decode_video
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg/ffprobe required for video roundtrip')
    grid=8
    features=np.arange(grid*grid*12,dtype=np.uint16).reshape(grid,grid,12)*73
    atlas=np.zeros((12//columns*grid,columns*grid),dtype=np.uint16)
    for channel in range(12):
        y,x=divmod(channel,columns)
        atlas[y*grid:(y+1)*grid,x*grid:(x+1)*grid]=features[:,:,channel]
    video=tmp_path/'atlas.mkv'
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','gray16le',
                    '-s',f'{atlas.shape[1]}x{atlas.shape[0]}','-i','pipe:0',
                    '-frames:v','1','-c:v','ffv1',str(video)],
                   input=atlas.astype('<u2').tobytes(),check=True)
    decoded=list(decode_video(video,1,grid,atlas_shape=(12//columns,columns)))[0]
    np.testing.assert_array_equal(decoded,atlas)
    actual=unpack_features(decoded,grid,12,-2.,3.)
    np.testing.assert_array_equal(actual,(features.reshape(-1,12)/65535*5-2).astype(np.float32))
    with pytest.raises(ValueError,match='disagree'):
        list(decode_video(video,1,grid,atlas_shape=(columns,12//columns)))


def test_atlas_rejects_extra_or_partial_tiles():
    for shape in ((8,8),(7,6)):
        with pytest.raises(ValueError,match='exactly'):
            unpack_features(np.zeros(shape),2,12,0,1)


def test_upstream_atlas_channel_order_and_normalization():
    # Row-major 4x3 tile layout; each channel has distinguishable spatial data.
    grid=2;atlas=np.arange(8*6,dtype=np.uint16).reshape(8,6)*1000
    actual=unpack_features(atlas,grid,12,-1,2)
    expected=np.stack([atlas[(i//3)*grid:(i//3+1)*grid,(i%3)*grid:(i%3+1)*grid]/65535*3-1 for i in range(12)],axis=-1).reshape(4,12).astype(np.float32)
    np.testing.assert_array_equal(actual,expected)


def test_cancel_and_member_path(tmp_path):
    stop=threading.Event();stop.set()
    with pytest.raises(CancelledError):check_cancel(stop)
    with pytest.raises(ValueError):safe_file(tmp_path,'../outside')


def test_vendored_source_fingerprints():
    assert verify_vendor()['commit']=='ca8602516352efd83e9e69e665b2e26ed03cb5cc'
