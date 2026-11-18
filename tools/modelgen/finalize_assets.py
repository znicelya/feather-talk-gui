"""Update component/bundle identities after verified generated SCRFD export."""
import sys,json,hashlib,shutil
from pathlib import Path
from prepare_scrfd import SOURCE,digest

def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def finalize(directory):
    directory=Path(directory)
    config=directory/'config.json'
    if not config.exists():
        write(config,{'architecture_version':'scrfd_2_5g-source-v1','source_sha256':SOURCE,'configuration':{'graph':'architecture.onnx','input':[1,3,640,640],'opsets':{'':12},'initializer_names':'original ONNX'}})
    manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'identity':{'kind':'scrfd_2_5g','architecture_version':'scrfd_2_5g-source-v1','source_sha256':SOURCE},'min_engine_version':'0.1.0','urls':{}}

    assert manifest['identity']['source_sha256']==SOURCE
    shutil.copyfile(directory/'modelgen-staging/scrfd.provenance.json',directory/'generation.json')
    shutil.copyfile(directory/'modelgen-staging/scrfd.onnx',directory/'generated.onnx')
    names=['architecture.onnx','config.json','model.bpk','generation.json','generated.onnx']
    if (directory/'model.safetensors').exists():names.append('model.safetensors')
    names.sort()
    manifest['relative_files']=names
    manifest['hashes']={p:digest(directory/p) for p in names}
    manifest['sizes']={p:(directory/p).stat().st_size for p in names}
    manifest['identity']['weights_sha256']=digest(directory/'model.bpk')
    manifest['provenance']='burn-onnx=0.22.0; scrfd-fixed640-redundant-ceil-v1; original ONNX retained; see generation.json'
    write(directory/'manifest.json',manifest)
    bundle_path=directory.parent/'manifest.json'
    if bundle_path.exists():
        bundle=json.loads(bundle_path.read_text())
        component=directory.name+'/'
        names=[p for p in bundle['relative_files'] if not p.startswith(component)]+[component+p for p in [*names,'manifest.json']]
        names=sorted(names)
        bundle['relative_files']=names
        bundle['hashes']={p:digest(directory.parent/p) for p in names}
        bundle['sizes']={p:(directory.parent/p).stat().st_size for p in names}
        identities={p.parent.name:json.loads(p.read_text())['identity']['weights_sha256'] for p in directory.parent.glob('*/manifest.json')}
        bundle['identity']['weights_sha256']=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
        write(bundle_path,bundle)
        reference=directory.parent.parent/'reference/manifest.json'
        if reference.exists():
            index=json.loads(reference.read_text());index['assets_manifest_sha256']=digest(bundle_path);write(reference,index)
if __name__=='__main__':finalize(sys.argv[1])
