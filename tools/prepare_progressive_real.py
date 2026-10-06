"""Fetch a bounded 8i subset and call the repository's original preprocessing.

No training/rendering method changes. Data and derivatives stay in ignored output/.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.content_preparation.paths import UPSTREAM_ROOT, configure_upstream_imports
from tools.content_preparation.compatibility import store_point_cloud
from tools.check_progressive_composability import dump, digest


class RangeReader(io.RawIOBase):
    """Seekable ZIP input: retrieve only central directory and selected members."""
    def __init__(self, url):
        self.url, self.pos, self.transferred = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.length = int(r.headers['Content-Length'])
            self.etag = r.headers.get('ETag')

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.length)
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, size=-1):
        size = min(self.length-self.pos, size if size >= 0 else self.length-self.pos)
        if size <= 0:
            return b''
        end = self.pos+size-1
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(request, timeout=120) as r:
            assert r.status == 206, 'Server must support ranges; refusing full archive download'
            assert r.headers['Content-Range'] == f'bytes {self.pos}-{end}/{self.length}'
            assert r.headers.get('ETag') == self.etag
            data = r.read()
        assert len(data) == size
        self.pos += size
        self.transferred += size
        return data


def download(out, start, count):
    raw = out/'raw'; raw.mkdir(parents=True, exist_ok=True)
    url = 'http://plenodb.jpeg.org/pc/8ilabs/longdress.zip'
    license_path = raw/'license.pdf'
    if not license_path.exists():
        urllib.request.urlretrieve('http://plenodb.jpeg.org/pc/8ilabs/license.pdf', license_path)
    source = RangeReader(url)
    records = []
    with zipfile.ZipFile(source) as archive:
        for frame in range(start, start+count):
            name = f'longdress_vox10_{frame:04d}.ply'
            matches = [n for n in archive.namelist() if Path(n).name == name and not n.startswith('__MACOSX/')]
            assert len(matches) == 1, (name, matches)
            member = archive.getinfo(matches[0])
            target = raw/'longdress/Ply'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            # archive.read verifies ZIP CRC before a file is committed to disk.
            data = archive.read(member)
            if target.exists():
                assert target.read_bytes() == data, f'Refusing to replace {target}'
            else:
                target.write_bytes(data)
            records.append(dict(frame=frame, path=str(target), member=member.filename,
                                bytes=len(data), compressed_bytes=member.compress_size,
                                zip_crc32=f'{member.CRC:08x}', sha256=digest(target)))
            print(f'Downloaded {name}: {len(data):,} bytes, CRC verified', flush=True)
    dump(raw/'provenance.json', dict(source_url=url, archive_bytes=source.length,
         archive_etag=source.etag, transferred_bytes=source.transferred, frames=records,
         license=str(license_path), license_sha256=digest(license_path),
         transport='HTTP from official JPEG host; HTTPS certificate chain failed locally; ZIP CRC is not a cryptographic publisher signature'))


def prepare(out, test_views):
    from PIL import Image
    raw = json.loads((out/'raw/provenance.json').read_text())
    assert 1 <= test_views <= 200
    spec = dict(sequence='8i_Longdress', synthetic=False, frames=[],
                raw_provenance=str(out/'raw/provenance.json'),
                preprocessing='original dataset_prepare.render_2d_image: voxel 1.7, per-frame AABB center, /480, y offset -0.1, X rotation +90, unlit point size 2, 1024x1024; original rescale_image Lanczos',
                preprocessing_sha256=digest(UPSTREAM_ROOT/'dataset_prepare.py'),
                train_views=100, test_views=test_views, levels={'Q0':'res8 (128x128)','Q1':'res4 (256x256)'},
                initialization='original NeRF reader default 100000 random points; no supplied surface PLY')
    selected = {}
    for split in ['train','test']:
        original = json.loads((UPSTREAM_ROOT/f'transforms_{split}.json').read_text())
        indices = list(range(len(original['frames']))) if split == 'train' else [i*200//test_views for i in range(test_views)]
        meta = dict(original, frames=[dict(original['frames'][i], file_path=f'./{split}/r_{j}') for j,i in enumerate(indices)])
        pose = out/'poses'/f'transforms_{split}.json'
        dump(pose, meta)
        selected[split] = indices
    spec['original_camera_indices'] = selected
    for record in raw['frames']:
        frame = record['frame']; pc = Path(record['path'])
        assert digest(pc) == record['sha256']
        for split in ['train','test']:
            folder = out/'source/8i/longdress/longdress_res1'/f'{frame:04d}'
            pose = out/'poses'/f'transforms_{split}.json'
            marker = folder/f'{split}.complete.json'
            if not marker.exists():
                log = out/'logs'/f'prepare_{frame}_{split}.log'; log.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(Path(__file__).resolve()), '--render-one', str(pc), str(folder/split), str(pose)]
                print(f'Original Open3D preprocessing {frame} {split}', flush=True)
                with log.open('w') as f:
                    subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, EGL_PLATFORM='surfaceless'), stdout=f, stderr=subprocess.STDOUT, check=True)
                dump(marker, dict(raw_sha256=record['sha256'], pose_sha256=digest(pose)))
            else:
                saved = json.loads(marker.read_text())
                assert saved == dict(raw_sha256=record['sha256'], pose_sha256=digest(pose))
            meta = json.loads(pose.read_text())
            # Call rescale_image directly: same resize and RGBA behavior as upstream.
            from tools.content_preparation.compatibility import rescale_image
            for scale in [8,4]:
                dest = out/f'source/8i/longdress/longdress_res{scale}'/f'{frame:04d}'
                dump(dest/f'transforms_{split}.json', meta)
                for item in meta['frames']:
                    relative = item['file_path']+'.png'
                    image = folder/relative
                    with Image.open(image) as im:
                        assert im.size == (1024,1024) and im.mode == 'RGBA'
                    target = dest/relative; target.parent.mkdir(parents=True,exist_ok=True)
                    rescale_image(image, scale).save(target)
        spec['frames'].append(dict(frame=frame, Q0_source=str(out/'source/8i/longdress/longdress_res8'/f'{frame:04d}'),
                                  Q1_source=str(out/'source/8i/longdress/longdress_res4'/f'{frame:04d}')))
        dump(out/'data_spec.json', spec)
        print(f'Prepared frame {frame}: 100 train / {test_views} test, res8 + res4',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'output/progressive_gap_real')
    p.add_argument('--download',action='store_true')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--start',type=int,default=1051)
    p.add_argument('--frames',type=int,default=5)
    p.add_argument('--test-views',type=int,default=16)
    p.add_argument('--render-one',nargs=3,metavar=('PLY','OUTPUT','POSES'))
    args = p.parse_args()
    if args.render_one:
        os.environ.setdefault('EGL_PLATFORM','surfaceless')
        from tools.content_preparation.compatibility import render_2d_image
        render_2d_image(*args.render_one,pt_size=2,width=1024,height=1024)
        return
    out=args.output.resolve()
    if args.download:
        assert 5 <= args.frames <= 10
        download(out,args.start,args.frames)
    if args.prepare:
        prepare(out,args.test_views)


if __name__ == '__main__':
    main()
