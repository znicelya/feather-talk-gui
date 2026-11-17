import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import audit_sources, require_current_hubert
from convert import convert_current_assets

SOURCE = Path(os.environ['FT_SOURCE_DIR'])
MODELS = Path(os.environ['FT_SUPPLIED_MODELS_DIR'])

def test_rejects_188_source():
    old = __import__('json').loads((MODELS / 'feather_hubert/manifest.json').read_text())
    with pytest.raises(ValueError, match='current'):
        require_current_hubert(old['source']['sha256'])
    audit = audit_sources(SOURCE, MODELS)
    assert audit['feather_hubert']['source_sha256'] == '3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46'
    assert audit['feather_hubert']['supplied_is_current'] is False

def test_mean_face_provenance():
    face = audit_sources(SOURCE, MODELS)['mean_face']
    assert face['count'] == 220
    assert face['source_sha256'] == '40bed2653ae861e9a24b4766c3d085986b7c1245a27373913a9a5b1a6ce069ef'

def test_exports_checkpoint_configuration(tmp_path):
    import json
    from safetensors.torch import load_file
    import torch
    path = convert_current_assets(SOURCE, MODELS, tmp_path)
    bundle = json.loads(path.read_text())
    config = json.loads((tmp_path / 'feather_hubert/config.json').read_text())
    actual = torch.load(SOURCE / 'feather_hubert.pth', map_location='cpu', weights_only=True)
    assert config['configuration'] == {k: actual['config'][k] for k in config['configuration']}
    assert bundle['identity']['source_sha256'] == '3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46'
    state = load_file(tmp_path / 'feather_hubert/model.safetensors')
    assert set(state) == set(actual['model'])
    assert all(torch.equal(state[k], v) for k, v in actual['model'].items())
    assert load_file(tmp_path / 'mean_face/model.safetensors')['mean_face'].numel() == 220
    assert len(load_file(tmp_path / 'vgg19/model.safetensors')) == 14
    for filename in bundle['relative_files']:
        from audit import sha256
        assert sha256(tmp_path / filename) == bundle['hashes'][filename]
        assert (tmp_path / filename).stat().st_size == bundle['sizes'][filename]

def test_references_are_complete_and_reproducible(tmp_path):
    from generate import generate_reference
    from verify import verify_reference
    import json
    assets = Path(os.environ['FT_ASSETS_DIR'])
    first = generate_reference(SOURCE, assets, tmp_path / 'first')
    second = generate_reference(SOURCE, assets, tmp_path / 'second')
    assert first.read_bytes() == second.read_bytes()
    verify_reference(first.parent)
    from safetensors.torch import load_file
    import torch
    first_bn = load_file(first.parent / 'batch_norm_train.safetensors')
    second_bn = load_file(first.parent / 'batch_norm_train_second.safetensors')
    value = first_bn['input.0']
    mean, variance = value.mean((0,2,3)), value.var((0,2,3), unbiased=True)
    torch.testing.assert_close(first_bn['state.running_mean'], 0.1*mean)
    torch.testing.assert_close(first_bn['state.running_var'], 0.9 + 0.1*variance)
    value2 = second_bn['input.0']
    torch.testing.assert_close(second_bn['state.running_var'], 0.9*first_bn['state.running_var'] + 0.1*value2.var((0,2,3), unbiased=True))
    assert second_bn['state.num_batches_tracked'].item() == 2
    resize = load_file(first.parent / 'resize_align_corners.safetensors')
    torch.testing.assert_close(resize['output'][...,0,0], resize['input.0'][...,0,0])
    torch.testing.assert_close(resize['output'][...,-1,-1], resize['input.0'][...,-1,-1])
    torch.testing.assert_close(resize['input_gradient.0'].sum(), resize['upstream_gradient'].sum())
    assert resize['upstream_gradient'].std() > 0
    manifest = json.loads(first.read_text())
    for key in ['resize_align_corners','batch_norm_train','batch_norm_train_second','character_training','feather_hubert','pfld_ghost_one','vgg19','scrfd_2_5g','geometry','audio_clip_first','audio_extract_320400']:
        assert key in manifest['cases']
    broken = first.parent / manifest['cases']['geometry']['file']
    broken.write_bytes(broken.read_bytes()[:-1] + b'x')
    with pytest.raises(ValueError, match='digest'):
        verify_reference(first.parent)
    from safetensors.torch import load_file
    import torch
    first_bn = load_file(first.parent / 'batch_norm_train.safetensors')
    second_bn = load_file(first.parent / 'batch_norm_train_second.safetensors')
    value = first_bn['input.0']
    mean, variance = value.mean((0,2,3)), value.var((0,2,3), unbiased=True)
    torch.testing.assert_close(first_bn['state.running_mean'], 0.1*mean)
    torch.testing.assert_close(first_bn['state.running_var'], 0.9 + 0.1*variance)
    value2 = second_bn['input.0']
    torch.testing.assert_close(second_bn['state.running_var'], 0.9*first_bn['state.running_var'] + 0.1*value2.var((0,2,3), unbiased=True))
    assert second_bn['state.num_batches_tracked'].item() == 2
    resize = load_file(first.parent / 'resize_align_corners.safetensors')
    torch.testing.assert_close(resize['output'][...,0,0], resize['input.0'][...,0,0])
    torch.testing.assert_close(resize['output'][...,-1,-1], resize['input.0'][...,-1,-1])
    torch.testing.assert_close(resize['input_gradient.0'].sum(), resize['upstream_gradient'].sum())
    assert resize['upstream_gradient'].std() > 0
