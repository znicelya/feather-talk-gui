"""Read-only provenance audit. Refuse the historical 188 audio conversion."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys
import os

CURRENT_HUBERT = '3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46'
MEAN_FACE = '40bed2653ae861e9a24b4766c3d085986b7c1245a27373913a9a5b1a6ce069ef'

def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def require_current_hubert(digest: str):
    if digest != CURRENT_HUBERT:
        raise ValueError(f'Not the approved current FeatherHuBERT source: {digest}')

def source_modules(source: Path):
    sys.path[:0] = [str(source.resolve()), str((source / 'data_utils').resolve())]
    return importlib.import_module('data_utils.feather_hubert.feather_hubert')

def tensor_inventory(state):
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in sorted(state.items())}

def canonical_vgg_state(state):
    indices = {'conv1_1':0, 'conv1_2':2, 'conv2_1':5, 'conv2_2':7, 'conv3_1':10, 'conv3_2':12, 'conv3_3':14}
    result = {}
    for name, value in state.items():
        head, suffix = name.rsplit('.', 1)
        index = indices.get(head)
        result[(str(index) + '.' + suffix) if index is not None else name.removeprefix('features.')] = value
    return result

def audit_sources(source_dir: Path, supplied_models: Path) -> dict:
    import numpy as np
    import torch
    from safetensors.torch import load_file
    source_dir, supplied_models = Path(source_dir), Path(supplied_models)
    hubert = source_dir / 'feather_hubert.pth'
    require_current_hubert(sha256(hubert))
    checkpoint = torch.load(hubert, map_location='cpu', weights_only=True)
    module = source_modules(source_dir)
    from dataclasses import asdict
    config = asdict(module.feather_hubert_config_from_mapping(checkpoint.get('config', checkpoint.get('args'))))
    model = module.FeatherHuBERTEncoder(module.FeatherHuBERTConfig(**config))
    model.load_state_dict(checkpoint['model'], strict=True)
    old = json.loads((supplied_models / 'feather_hubert/manifest.json').read_text())
    result = {'feather_hubert': {'source_sha256': sha256(hubert), 'configuration': config,
        'supplied_source_sha256': old['source']['sha256'], 'supplied_is_current': old['source']['sha256'] == CURRENT_HUBERT,
        'tensors': tensor_inventory(checkpoint['model'])}}
    face = source_dir / 'data_utils/mean_face.txt'
    values = np.fromstring(face.read_text(), sep=' ', dtype=np.float32)
    if sha256(face) != MEAN_FACE or values.size != 220:
        raise ValueError('Mean face provenance mismatch')
    result['mean_face'] = {'source_sha256': sha256(face), 'count': int(values.size)}
    from pfld_mobileone import PFLD_GhostOne
    pfld_source = source_dir / 'data_utils/checkpoint_epoch_335.pth.tar'
    pfld_checkpoint = torch.load(pfld_source, map_location='cpu', weights_only=True)
    state = pfld_checkpoint['pfld_backbone']
    PFLD_GhostOne().load_state_dict(state, strict=True)
    supplied = load_file(supplied_models / 'pfld_ghost_one/model.safetensors')
    equal = set(state) == set(supplied) and all(torch.equal(v, supplied[k]) for k, v in state.items())
    result['pfld_ghost_one'] = {'source_sha256': sha256(pfld_source), 'supplied_matches_source': equal,
        'tensors': tensor_inventory(state)}
    import onnx
    graph_path = source_dir / 'data_utils/scrfd_2.5g_kps.onnx'
    graph = onnx.load(str(graph_path))
    onnx.checker.check_model(graph)
    scrfd_manifest = json.loads((supplied_models / 'scrfd_2_5g/manifest.json').read_text())
    supplied = load_file(supplied_models / 'scrfd_2_5g/model.safetensors')
    result['scrfd_2_5g'] = {'source_sha256': sha256(graph_path),
        'supplied_source_matches': scrfd_manifest['source']['sha256'] == sha256(graph_path),
        'supplied_tensors': tensor_inventory(supplied),
        'source_initializers': {t.name: list(t.dims) for t in graph.graph.initializer},
        'conversion': 'Rebuild with original ONNX initializer names; avoid opaque Burn 0.21 names'}
    from torchvision.models.vgg import make_layers, cfgs
    features = make_layers(cfgs['E'])[:15]
    vgg_path = supplied_models / 'vgg19/model.safetensors'
    vgg = load_file(vgg_path)
    # Supplied keys are torchvision feature indices, not classifier weights.
    features.load_state_dict(canonical_vgg_state(vgg), strict=True)
    manifest = json.loads((supplied_models / 'vgg19/manifest.json').read_text())
    if sha256(vgg_path) != manifest['model']['sha256']:
        raise ValueError('VGG supplied digest mismatch')
    result['vgg19'] = {'source_sha256': manifest['source']['sha256'], 'supplied_sha256': sha256(vgg_path),
        'tensors': tensor_inventory(vgg), 'provenance': 'Supplied converted subset; upstream source digest is manifest assertion, original full checkpoint unavailable'}
    upstream = Path(os.environ.get('FT_VGG_SOURCE', '.artifacts/source-downloads/vgg19-dcbb9e9d.pth'))
    if not upstream.is_file() or sha256(upstream) != manifest['source']['sha256']:
        raise ValueError('Verified original VGG checkpoint required; set FT_VGG_SOURCE (see README)')
    original = torch.load(upstream, map_location='cpu', weights_only=True)
    subset = {k: v for k,v in original.items() if k.startswith('features.') and int(k.split('.')[1]) <= 14}
    converted = {'features.' + k: v for k,v in canonical_vgg_state(vgg).items()}
    if set(subset) != set(converted) or not all(torch.equal(v, converted[k]) for k,v in subset.items()):
        raise ValueError('Supplied VGG subset differs from verified upstream checkpoint')
    result['vgg19']['supplied_matches_source'] = True
    result['vgg19']['provenance'] = 'Official torchvision checkpoint SHA-256 verified; features[0..14] tensors exactly equal supplied subset'
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    payload = json.dumps(audit_sources(args.source, args.models), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + '\n', encoding='utf-8')
    print(payload)

if __name__ == '__main__':
    main()
