"""Source-only allowance for two numerically sensitive intermediate taps.

The complete original model is evaluated in FP64, with identical frozen FP32
inputs and weights cast to FP64. No Rust/Burn output enters this calculation.
"""
import copy
import hashlib
import inspect
from pathlib import Path

TAPS = ('frontend.layers.0', 'encoder')

def calibrate_hubert(module, waveform, fp32_taps, original):
    import torch
    model = copy.deepcopy(module).double().eval()
    taps = {}
    handles = []
    for key in TAPS:
        handles.append(model.get_submodule(key).register_forward_hook(
            lambda mod, inputs, output, key=key: taps.update({key: output.detach().clone()})))
    with torch.no_grad():
        model(waveform.double())
    for handle in handles:
        handle.remove()
    result = {}
    for key in TAPS:
        difference = (fp32_taps['layer.' + key].double() - taps[key]).abs()
        maximum = difference.max().item()
        violations = (difference > original['atol'] + original['rtol'] * fp32_taps['layer.' + key].abs()).sum().item()
        result['layer.' + key] = {
            'atol': max(original['atol'], 4 * maximum), 'rtol': original['rtol'],
            'calibration': {
                'source_file_sha256': hashlib.sha256(Path(inspect.getfile(type(module))).read_bytes()).hexdigest(),
                'source_weights_sha256': '3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46',
                'method': 'complete original PyTorch FeatherHuBERT FP32 versus FP64; identical frozen FP32 input/weights cast to FP64',
                'max_abs_source_error': maximum, 'original_gate_violations': violations,
                'factor': 4, 'original_atol': original['atol'],
                'rationale': 'Source-only empirical reduction allowance for these intermediate taps; final outputs and extraction gates remain fixed. No Burn outputs used.',
            },
        }
    return result
