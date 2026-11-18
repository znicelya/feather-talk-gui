"""Developer-only resource packaging. Inputs are explicit; never downloads or imports source code."""
import argparse, hashlib, json, shutil
from pathlib import Path

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def main():
    a=argparse.ArgumentParser()
    for key in ['assets','media','probe','probe-build','cuda','source','supplied','output','catalog']: a.add_argument('--'+key,type=Path,required=True)
    args=a.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    components=[]
    def package(id,kind,source,paths,executables,notices,license,backend=None,original=False):
        root=args.output/id; root.mkdir(exist_ok=True)
        for rel,p in paths.items():
            dest=root/rel; dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        if original: shutil.copyfile(source/'manifest.json',root/'manifest.json');m=json.loads((root/'manifest.json').read_text())
        else:
            files=sorted(paths); h=digest(root/files[0])
            m=dict(identity=dict(kind=id,architecture_version='1',source_sha256=h,weights_sha256=h),relative_files=files,sizes={f:(root/f).stat().st_size for f in files},hashes={f:digest(root/f) for f in files},urls={},min_engine_version='0.1.0',provenance=f'Local development package: {kind}; exact selected input files, no redistribution approval asserted')
            write(root/'manifest.json',m)
        supplemental={f:dict(size=(root/f).stat().st_size,sha256=digest(root/f),urls=[]) for f in paths if f not in m['relative_files']}
        mh=digest(root/'manifest.json')
        c=dict(pin=dict(id=id,version=mh[:16]),target=None if backend is None else dict(os='windows',architecture='x86_64',backend=backend),kind=kind,manifest=m,manifest_sha256=mh,manifest_size=(root/'manifest.json').stat().st_size,manifest_urls=[],supplemental=supplemental,executables=executables,notices=notices,redistribution_approved=False,license=license)
        components.append(c)
    m=json.loads((args.assets/'manifest.json').read_text())
    files={f:args.assets/f for f in m['relative_files']}
    evidence={'evidence/source-LICENSE':args.source/'LICENSE','evidence/historical-hubert-LICENSES.json':args.supplied/'feather_hubert/LICENSES.json','evidence/vgg-LICENSES.json':args.supplied/'vgg19/LICENSES.json','evidence/pfld-historical-manifest.json':args.supplied/'pfld_ghost_one/manifest.json','evidence/scrfd-historical-manifest.json':args.supplied/'scrfd_2_5g/manifest.json'}
    package('public-assets','model_bundle',args.assets,files|evidence,[],list(evidence),'NOASSERTION',original=True)
    media={str(p.relative_to(args.media)).replace('\\','/'):p for p in args.media.rglob('*') if p.is_file() and (p.suffix!='.exe' or p.name in ['ffmpeg.exe','ffprobe.exe'])}
    package('media','media',args.media,media,['bin/ffmpeg.exe','bin/ffprobe.exe'],['LICENSE','README.txt'],'GPL-3.0-only','any')
    package('probe','probe',args.probe.parent,{'feathertalk-probe.exe':args.probe,'BUILD.json':args.probe_build},['feathertalk-probe.exe'],['BUILD.json'],'NOASSERTION','cpu')
    package('cuda-math','cuda_runtime',args.cuda,{'nvvm/libdevice/libdevice.10.bc':args.cuda/'nvvm/libdevice/libdevice.10.bc','LICENSE':args.cuda/'LICENSE'},[],['LICENSE'],'NVIDIA CUDA SDK license','cuda')
    defaults=[]
    pins={c['pin']['id']:c['pin'] for c in components}
    for os,arch,backends in [('windows','x86_64',['cpu','cuda','vulkan']),('linux','x86_64',['cpu','cuda','vulkan']),('macos','x86_64',['cpu']),('macos','aarch64',['cpu','metal'])]:
        for backend in backends:
            selected=[pins['public-assets'],pins['media'] if os=='windows' else dict(id='media',version='unconfigured'),pins['probe'] if os=='windows' and backend=='cpu' else dict(id='probe',version='unconfigured'),dict(id='worker',version='unconfigured')]
            if backend=='cuda': selected.append(pins['cuda-math'] if os=='windows' else dict(id='cuda-math',version='unconfigured'))
            defaults.append(dict(target=dict(os=os,architecture=arch,backend=backend),resources=selected))
    write(args.catalog,dict(schema_version=1,components=components,defaults=defaults))
    write(args.output/'imports.json',[str((args.output/c['pin']['id']/'manifest.json').resolve()) for c in components])
if __name__=='__main__': main()
