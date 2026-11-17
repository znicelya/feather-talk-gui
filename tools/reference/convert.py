"""Deterministic CPU conversion with architecture and Rust AssetManifest sidecars."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from audit import audit_sources, sha256, canonical_vgg_state, CURRENT_HUBERT

def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')

def convert_current_assets(source_dir: Path, supplied_models: Path, output_dir: Path) -> Path:
    import numpy as np
    import onnx
    import torch
    from safetensors.torch import save_file, load_file
    source_dir, supplied_models, output_dir = map(Path, (source_dir, supplied_models, output_dir))
    if output_dir.resolve() in [source_dir.resolve(), supplied_models.resolve()] or any(
        root.resolve() in output_dir.resolve().parents for root in [source_dir, supplied_models]):
        raise ValueError('Conversion output must not modify original source or models')
    audit = audit_sources(source_dir, supplied_models)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'audit.json', audit)
    states = {}
    states['feather_hubert'] = torch.load(source_dir / 'feather_hubert.pth', map_location='cpu', weights_only=True)['model']
    states['pfld_ghost_one'] = torch.load(source_dir / 'data_utils/checkpoint_epoch_335.pth.tar', map_location='cpu', weights_only=True)['pfld_backbone']
    graph_path = source_dir / 'data_utils/scrfd_2.5g_kps.onnx'
    graph = onnx.load(str(graph_path))
    states['scrfd_2_5g'] = {t.name: torch.from_numpy(onnx.numpy_helper.to_array(t).copy()) for t in graph.graph.initializer}
    states['vgg19'] = canonical_vgg_state(load_file(supplied_models / 'vgg19/model.safetensors'))
    states['mean_face'] = {'mean_face': torch.from_numpy(np.fromstring((source_dir / 'data_utils/mean_face.txt').read_text(), sep=' ', dtype=np.float32))}
    configurations = {
        'feather_hubert': audit['feather_hubert']['configuration'],
        'pfld_ghost_one': {'width_factor': 0.5, 'input_size': 192, 'landmark_number': 110, 'inference_mode': False, 'num_conv_branches': 6},
        'scrfd_2_5g': {'graph': 'architecture.onnx', 'input': [1,3,640,640], 'opsets': {x.domain: x.version for x in graph.opset_import}, 'initializer_names': 'original ONNX'},
        'vgg19': {'architecture': 'torchvision.vgg19.features', 'inclusive_last_layer': 14, 'color': 'BGR', 'range': [0,1], 'normalization': 'none'},
        'mean_face': {'count': 220, 'shape': [110,2]},
    }
    for name, state in states.items():
        directory = output_dir / name
        directory.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k,v in sorted(state.items())}, directory / 'model.safetensors')
        write_json(directory / 'config.json', {'architecture_version': name + '-source-v1', 'configuration': configurations[name], 'source_sha256': audit[name]['source_sha256']})
        if name == 'scrfd_2_5g':
            # Retain executable source graph as architecture evidence and ORT oracle.
            shutil.copyfile(graph_path, directory / 'architecture.onnx')
        paths = sorted(p.relative_to(directory).as_posix() for p in directory.iterdir() if p.is_file() and p.name != 'manifest.json')
        write_json(directory / 'manifest.json', {'identity': {'kind': name, 'architecture_version': name + '-source-v1',
            'source_sha256': audit[name]['source_sha256'], 'weights_sha256': sha256(directory / 'model.safetensors')},
            'relative_files': paths, 'sizes': {p: (directory/p).stat().st_size for p in paths},
            'hashes': {p: sha256(directory/p) for p in paths}, 'urls': {}, 'min_engine_version': '0.1.0',
            'provenance': audit[name].get('provenance', 'Rebuilt from original source; exact strict state loading')})
    paths = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob('*') if p.is_file() and p != output_dir / 'manifest.json')
    manifest = {'identity': {'kind': 'feathertalk-public-assets', 'architecture_version': 'source-v1', 'source_sha256': CURRENT_HUBERT,
        'weights_sha256': hashlib.sha256(json.dumps({name: sha256(output_dir/name/'model.safetensors') for name in sorted(states)},sort_keys=True).encode()).hexdigest()}, 'relative_files': paths,
        'sizes': {p: (output_dir/p).stat().st_size for p in paths}, 'hashes': {p: sha256(output_dir/p) for p in paths},
        'urls': {}, 'min_engine_version': '0.1.0', 'provenance': 'See component manifests and audit.json; original checkpoint provenance verified including official VGG subset comparison'}
    write_json(output_dir / 'manifest.json', manifest)
    return output_dir / 'manifest.json'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(convert_current_assets(a.source, a.models, a.output))
