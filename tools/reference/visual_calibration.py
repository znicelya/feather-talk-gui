"""Source-only visual VJP calibration and a strict, independent local block oracle."""
import copy
import hashlib
import inspect
from pathlib import Path

UNSTABLE = ('image_gradient', 'audio_gradient', 'gradient.inc.inconv.conv.0.weight',
            'gradient.audio_model.norm.weight')

def calibrate_character(module, initial_state, reference, upstream, original):
    model = copy.deepcopy(module).double().train()
    model.load_state_dict(initial_state, strict=True)
    model.zero_grad(set_to_none=True)
    image = reference['image'].double().requires_grad_()
    audio = reference['audio'].double().requires_grad_()
    (model(image, audio) * upstream.double()).sum().backward()
    actual = {'image_gradient': image.grad, 'audio_gradient': audio.grad}
    actual.update({'gradient.' + key: p.grad for key, p in model.named_parameters()})
    result = {}
    for key in UNSTABLE:
        expected = reference[key].double()
        difference = (expected - actual[key]).abs()
        maximum = difference.max().item()
        atol = max(original['atol'], 4 * maximum)
        # A detached/zero implementation must still fail every named tensor.
        assert (expected.abs() > atol + original['rtol'] * expected.abs()).any(), key
        result[key] = {'atol': atol, 'rtol': original['rtol'], 'calibration': {
            'source_file_sha256': hashlib.sha256(Path(inspect.getfile(type(module))).read_bytes()).hexdigest(),
            'method': 'complete ORIGINAL FeatherTalk native FP32 VJP versus complete ORIGINAL FP64 VJP; identical frozen FP32 state, inputs and upstream widened',
            'max_abs_source_error': maximum, 'factor': 4, 'original_atol': original['atol'],
            'original_gate_violations': int((difference > original['atol'] + original['rtol'] * expected.abs()).sum()),
            'zero_gradient_rejected': True,
            'rationale': 'Deep ReLU branch sensitivity to forward rounding; native forward values with wide derivatives pass original gates. Four named full-model gradients only. Empirical bound does not ensure 1% global mutation sensitivity; strict local block and attention gates remain unchanged.'}}
    return result

def local_block(visual, save, tolerance):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261123)
        model = visual.InvertedResidual(6, 24, stride=1, use_res_connect=False, expand_ratio=2).train()
        # Nonzero offsets avoid scale-invariant consecutive depthwise BN degeneracy.
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    channel=torch.arange(layer.num_features,dtype=torch.float32)
                    layer.weight.copy_(0.9+0.2*torch.sin(channel*0.37))
                    layer.bias.copy_(0.15*torch.cos(channel*0.29))
        initial = {k:v.clone() for k,v in model.state_dict().items()}
        x = (torch.randn(2,6,7,9)*0.7).requires_grad_()
        y = model(x)
        upstream = (torch.sin(torch.arange(y.numel())*0.071)*0.01).reshape_as(y)
        (y*upstream).sum().backward()
        data = {'input':x, 'output':y, 'upstream_gradient':upstream, 'input_gradient':x.grad}
        data.update({'state.'+k:v for k,v in initial.items()})
        data.update({'updated_state.'+k:v for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k})
        data.update({'gradient.'+k:p.grad for k,p in model.named_parameters()})
        wide = copy.deepcopy(model).double();wide.load_state_dict(initial);wide.zero_grad(set_to_none=True)
        x64=x.detach().double().requires_grad_();(wide(x64)*upstream.double()).sum().backward()
        wide_grads={'input_gradient':x64.grad, **{'gradient.'+k:p.grad for k,p in wide.named_parameters()}}
        evidence={}
        for key,value in wide_grads.items():
            expected=data[key].double();bound=tolerance['atol']+tolerance['rtol']*expected.abs()
            error=(value-expected).abs()
            assert (error<=bound).all(), ('unstable local derivative',key,error.max())
            assert (expected.abs()>bound).any(), ('zero mutation not rejected',key)
            assert ((expected*0.01).abs()>bound).any(), ('one percent mutation not rejected',key)
            evidence[key]={'fp64_max_abs_error':error.max().item(),'zero_mutation_rejected':True,'one_percent_mutation_rejected':True}
        save('visual_inverted_residual',data,{'input_channels':6,'output_channels':24,'expansion':2,'stride':1,'residual':False,'seed':20261123,'objective':'sum(output * nonlinear upstream_gradient)','source_stability_and_mutations':evidence},'gradient')
