"""Source-only normalization gradient calibration; never consumes Rust results."""
import copy

def calibrate_normalization(module, inputs, upstream, fp32_gradient, original):
    import torch
    model = copy.deepcopy(module).to(torch.float64)
    model.zero_grad(set_to_none=True)
    values = tuple(x.detach().to(torch.float64).requires_grad_() for x in inputs)
    (model(*values) * upstream.to(torch.float64)).sum().backward()
    error = (values[0].grad - fp32_gradient.to(torch.float64)).abs().max().item()
    return {
        'input_gradient.0': {
            'atol': max(original['atol'], 4 * error), 'rtol': original['rtol'],
            'calibration': {
                'method': 'original PyTorch module FP32 versus FP64 autograd on identical frozen FP32 inputs/upstream',
                'max_abs_source_error': error, 'factor': 4,
                'original_atol': original['atol'],
                'rationale': 'Empirical allowance for two FP32 reduction paths; not a mathematical error bound. No Burn outputs used.',
            },
        }
    }
