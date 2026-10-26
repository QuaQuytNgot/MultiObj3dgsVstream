"""CLI/configuration for the single quality-aware StreamSTGS entry point."""
import argparse
import json
import math
import os
from pathlib import Path

def project_root():
    """Locate checkout defaults, or resolve installed-package paths from cwd."""
    configured=os.environ.get('MULTIOBJ3DGS_PROJECT_ROOT')
    if configured:return Path(configured).expanduser().resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate/'pyproject.toml').is_file() and (candidate/'src_final/multiobj3dgs').is_dir():
            return candidate
    return Path.cwd()


ROOT=project_root()
DEFAULTS={
    'mode':'native','quality_metadata':'data_stream/streamstgs_8i_quality_metadata',
    'contents':['longdress'],'abr_mode':'proposed-dp','fixed_crf':40,
    'safety_factor':.85,'delta_mbps':.25,'startup_mbps':20.,'server_mbps':None,
    'fetch_concurrency':2,'workers':2,'prefetch':1,'frames':None,'network_gops':5,
    'width':1024,'height':768,'visibility_interval_s':2.,'analysis_width':256,
    'cache_mib':4096,'host_mib':2048,'raw_mib':512,'gpu_mib':4096,
    'gpu_transient_mib':256,'upload_budget_ms':2.,'ffmpeg_threads':2,
    'startup_timeout':300.,'loop':False,'unpaced':False,'diagnostics':True,
    'bandwidth_window':5,'bandwidth_max_age_s':30.,'cache':'.cache/abr_pipeline',
}


def load_config(argv=None):
    parser=argparse.ArgumentParser(description='Quality-aware ABR over the existing StreamSTGS DASH runtime')
    parser.add_argument('--config',type=Path,help='JSON config; relative paths are repository-relative')
    parser.add_argument('--mpd',help='HTTP(S) URL, local publication MPD, or publication directory')
    parser.add_argument('--publication',help='New publication output when no --mpd is supplied')
    parser.add_argument('--archive',action='append',dest='archives',help='Existing verified compressed archive; repeat per content')
    parser.add_argument('--quality-metadata')
    parser.add_argument('--base-url',help='HTTP publication base URL for a local MPD')
    parser.add_argument('--contents',help='Comma-separated content identities; repeats create independent instances')
    parser.add_argument('--objects',type=int,help='Repeat requested contents to this positive instance count')
    parser.add_argument('--scene',type=Path,help='Scene JSON with instances/transforms/offsets and duration')
    parser.add_argument('--mode',choices=('native','headless','network-eval'))
    parser.add_argument('--unthrottled',action='store_true',help='Use ordinary localhost HTTP without delivery pacing')
    parser.add_argument('--abr-mode',choices=('proposed-dp','fixed'))
    parser.add_argument('--fixed-crf',type=int)
    for key in ('safety_factor','delta_mbps','startup_mbps','server_mbps','visibility_interval_s',
                'upload_budget_ms','startup_timeout','bandwidth_max_age_s'):
        parser.add_argument('--'+key.replace('_','-'),type=float)
    for key in ('fetch_concurrency','workers','prefetch','frames','network_gops','width','height',
                'analysis_width','cache_mib','host_mib','raw_mib','gpu_mib','gpu_transient_mib',
                'ffmpeg_threads','bandwidth_window'):
        parser.add_argument('--'+key.replace('_','-'),type=int)
    parser.add_argument('--delivery-trace',type=Path,help='JSON [{at_s,mbps}] controlling actual local HTTP delivery')
    parser.add_argument('--control-trace',type=Path,help='Actions after_frame: seek,pause,play,loop,quality,camera (orbit/pan/zoom/target)')
    parser.add_argument('--vendor-root',type=Path)
    parser.add_argument('--cache')
    parser.add_argument('--report',help='New report path; adjacent .jsonl contains actual transfer/decision events')
    parser.add_argument('--capture-dir',help='Optional framebuffer PNG evidence from existing render calls')
    parser.add_argument('--capture-frames',help='Comma-separated zero-based fresh frame IDs to capture')
    parser.add_argument('--loop',action='store_true',default=None)
    parser.add_argument('--unpaced',action='store_true',default=None)
    parser.add_argument('--no-diagnostics',action='store_false',dest='diagnostics',default=None)
    args=vars(parser.parse_args(argv))
    config=dict(DEFAULTS)
    if args['config']:
        document=json.loads(args['config'].read_text())
        if not isinstance(document,dict):parser.error('config must be a JSON object')
        unknown=set(document)-set(DEFAULTS)-{'mpd','publication','archives','base_url','objects','scene','delivery_trace','control_trace','vendor_root','report','capture_dir','capture_frames'}
        if unknown:parser.error('Unknown config keys: '+', '.join(sorted(unknown)))
        config.update(document)
    unthrottled=args.pop('unthrottled')
    config.update({k:v for k,v in args.items() if k!='config' and v is not None})
    if unthrottled:
        config['server_mbps']=None
        config.pop('delivery_trace',None)
    if isinstance(config['contents'],str):config['contents']=[x.strip().lower() for x in config['contents'].split(',') if x.strip()]
    config['contents']=[x.lower() for x in config['contents']]
    if not config['contents']:parser.error('At least one content required')
    for key in ('fetch_concurrency','workers','network_gops','width','height','analysis_width','cache_mib',
                'host_mib','raw_mib','gpu_mib','gpu_transient_mib','ffmpeg_threads','bandwidth_window','objects','frames'):
        value=config.get(key)
        if value is not None and (type(value) is not int or value<=0):parser.error(key+' must be a positive integer')
    if type(config['prefetch']) is not int or config['prefetch']<0:parser.error('prefetch must be nonnegative')
    for key in ('delta_mbps','startup_mbps','visibility_interval_s','startup_timeout','bandwidth_max_age_s'):
        if not math.isfinite(config[key]) or config[key]<=0:parser.error(key+' must be finite and positive')
    if config['server_mbps'] is not None and (not math.isfinite(config['server_mbps']) or config['server_mbps']<=0):
        parser.error('server_mbps must be finite and positive, or null for unthrottled HTTP')
    if not math.isfinite(config['safety_factor']) or not 0<config['safety_factor']<=1:parser.error('safety_factor must be in (0,1]')
    if not math.isfinite(config['upload_budget_ms']) or config['upload_budget_ms']<0:parser.error('upload_budget_ms must be nonnegative')
    if config['gpu_transient_mib']>=config['gpu_mib']:parser.error('gpu_transient_mib must be less than gpu_mib')
    if config['mode']=='native' and config['unpaced']:parser.error('unpaced is a headless experiment option')
    if config['abr_mode'] not in ('proposed-dp','fixed') or config['mode'] not in ('native','headless','network-eval'):parser.error('Invalid mode')
    return config


def path(value):
    result=Path(value)
    return result if result.is_absolute() else ROOT/result
