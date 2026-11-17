"""Verify every tensor file before exposing a reference to a port parity test."""
from pathlib import Path
import argparse
import json
from audit import sha256

def verify_reference(reference: Path):
    import torch
    from safetensors.torch import load_file
    reference = Path(reference)
    index = json.loads((reference / 'manifest.json').read_text(encoding='utf-8'))
    for name, case in index['cases'].items():
        path = reference / case['file']
        if path.resolve().parent != reference.resolve():
            raise ValueError(f'Invalid tensor path: {name}')
        if sha256(path) != case['sha256']:
            raise ValueError(f'Tensor digest mismatch: {name}')
        tensors = load_file(path)
        if set(tensors) != set(case['tensors']):
            raise ValueError(f'Tensor keys differ: {name}')
        for key, tensor in tensors.items():
            spec = case['tensors'][key]
            if list(tensor.shape) != spec['shape'] or str(tensor.dtype) != spec['dtype']:
                raise ValueError(f'Tensor metadata differs: {name}/{key}')
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite tensor: {name}/{key}')
    return len(index['cases'])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    a = p.parse_args()
    print(f'Verified {verify_reference(a.reference)} reference cases')
