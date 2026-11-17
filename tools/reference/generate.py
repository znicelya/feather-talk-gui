"""Create portable, deterministic FP32 oracle tensors; no real footage implied."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import sys
from audit import sha256, source_modules
from convert import write_json

SEED = 20261117

def generate_reference(source_dir: Path, assets: Path, output_dir: Path) -> Path:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    source_dir, assets, output_dir = map(Path, (source_dir, assets, output_dir))
    for root in (source_dir, assets):
        if output_dir.resolve() == root.resolve() or root.resolve() in output_dir.resolve().parents:
            raise ValueError('References must be separate from sources and converted assets')
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    hubert_module = source_modules(source_dir)
    import model as visual
    import train
    import face_utils as face
    import dataset_mouth_roi as roi
    from pfld_mobileone import PFLD_GhostOne
    from torchvision.models.vgg import make_layers, cfgs
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    # Tolerances are prospective FP32 parity gates, never fitted to a failing port.
    # Unit operators accumulate few terms. Networks amplify reduction-order error.
    tolerances = {'exact': {'atol': 0.0, 'rtol': 0.0}, 'operator': {'atol': 2e-6, 'rtol': 2e-5},
        'network': {'atol': 2e-5, 'rtol': 2e-4}, 'gradient': {'atol': 2e-6, 'rtol': 5e-4}}
    def save(name, tensors, configuration, tolerance='operator'):
        path = output_dir / (name + '.safetensors')
        clean = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).detach().cpu().contiguous().clone() for k,v in sorted(tensors.items())}
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in clean.values()):
            raise ValueError(f'Nonfinite reference: {name}')
        save_file(clean, path)
        entries[name] = {'file': path.name, 'sha256': sha256(path), 'configuration': configuration,
            'tolerance': tolerances[tolerance], 'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in clean.items()}}
    def operator(name, module, inputs, config):
        module.train()
        module.zero_grad(set_to_none=True)
        inputs = tuple(x.detach().requires_grad_() for x in inputs)
        y = module(*inputs)
        upstream = torch.linspace(-0.7, 1.3, y.numel()).reshape_as(y)
        (y * upstream).sum().backward()
        data = {f'input.{i}': x for i,x in enumerate(inputs)}
        data.update({'output': y, 'upstream_gradient': upstream, 'objective': (y * upstream).sum()})
        data.update({f'input_gradient.{i}': x.grad for i,x in enumerate(inputs)})
        data.update({'state.'+k:v for k,v in module.state_dict().items()})
        data.update({'gradient.'+k:p.grad for k,p in module.named_parameters() if p.grad is not None})
        save(name, data, config)
    x = torch.linspace(-1, 1, 2*4*5*7).reshape(2,4,5,7)
    operator('resize_align_corners', torch.nn.Upsample(size=(9,11), mode='bilinear', align_corners=True), (x,), {'align_corners': True, 'mode': 'bilinear', 'size': [9,11], 'objective': 'sum(output * upstream_gradient)'})
    operator('group_dilated_conv2d', torch.nn.Conv2d(4,8,3,padding=2,dilation=2,groups=2), (x,), {'groups':2,'dilation':2,'padding':2,'kernel':3})
    operator('depthwise_dilated_conv1d', torch.nn.Conv1d(4,4,5,padding=4,dilation=2,groups=4), (x.reshape(2,4,35),), {'groups':4,'dilation':2,'padding':4,'kernel':5})
    gn = torch.nn.GroupNorm(2,4)
    operator('group_norm', gn, (x,), {'groups':2,'epsilon':1e-5,'variance':'biased'})
    gn.eval()
    save('group_norm_eval', {'input':x,'output':gn(x),**{'state.'+k:v for k,v in gn.state_dict().items()}}, {'groups':2,'epsilon':1e-5,'variance':'biased','running_statistics':False})
    bn = torch.nn.BatchNorm2d(4)
    operator('batch_norm_train', bn, (x,), {'epsilon':1e-5,'momentum':0.1,'normalization_variance':'biased','running_variance':'unbiased','initial_running_mean':[0]*4,'initial_running_variance':[1]*4,'initial_num_batches_tracked':0})
    operator('batch_norm_train_second', bn, (x.flip(-1) * 0.3 + 2,), {'epsilon':1e-5,'momentum':0.1,'initial_state':'batch_norm_train state','normalization_variance':'biased','running_variance':'unbiased'})
    bn.eval()
    save('batch_norm_eval', {'input':x,'output':bn(x), **{'state.'+k:v for k,v in bn.state_dict().items()}}, {'epsilon':1e-5,'uses_running_statistics':True})
    operator('avg_pool_include_padding', torch.nn.AvgPool2d(5,1,2,count_include_pad=True), (x,), {'kernel':5,'stride':1,'padding':2,'count_include_pad':True})
    operator('spatial_audio_attention', visual.SpatialAudioFusion(8,audio_dim=12,attention_dim=8,num_heads=2,max_audio_steps=5,center_index=2), (torch.randn(2,8,3,4),torch.randn(2,5,12)), {'visual_channels':8,'audio_dim':12,'attention_dim':8,'num_heads':2,'max_audio_steps':5,'center_index':2})
    def network(name, module, inputs, layers):
        captured = {}
        handles = []
        for layer in layers:
            handles.append(module.get_submodule(layer).register_forward_hook(lambda m,i,o,key=layer: captured.update({'layer.'+key:o.detach().clone()})))
        module.eval()
        with torch.no_grad():
            out = module(*inputs)
        for handle in handles:
            handle.remove()
        data = {f'input.{i}':v for i,v in enumerate(inputs)}
        data['output'] = out
        data.update(captured)
        save(name, data, {'layers':layers,'mode':'eval'}, 'network')
    config = json.loads((assets/'feather_hubert/config.json').read_text())['configuration']
    hubert = hubert_module.FeatherHuBERTEncoder(hubert_module.FeatherHuBERTConfig(**config))
    hubert.load_state_dict(load_file(assets/'feather_hubert/model.safetensors'), strict=True)
    waveform = torch.sin(torch.arange(16000,dtype=torch.float32)*0.017)[None]
    network('feather_hubert', hubert, (waveform,), ['frontend.layers.0','frontend','encoder.0','encoder','final_norm','proj'])
    pfld = PFLD_GhostOne()
    pfld.load_state_dict(load_file(assets/'pfld_ghost_one/model.safetensors'), strict=True)
    network('pfld_ghost_one', pfld, (torch.rand(1,3,192,192),), ['conv1','conv3_3','conv4_3','conv5_4','conv8','conv_out'])
    vgg = make_layers(cfgs['E'])[:15].eval()
    vgg.load_state_dict({k.removeprefix('features.'):v for k,v in load_file(assets/'vgg19/model.safetensors').items()},strict=True)
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)
    network('vgg19',vgg,(torch.rand(1,3,144,144),),['0','4','9','14'])
    import onnxruntime as ort
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(assets/'scrfd_2_5g/architecture.onnx'),sess_options=session_options,providers=['CPUExecutionProvider'])
    scrfd_input = torch.linspace(-1,1,3*640*640).reshape(1,3,640,640)
    outputs = session.run(None,{session.get_inputs()[0].name:scrfd_input.numpy()})
    save('scrfd_2_5g',{'input.0':scrfd_input,**{o.name:torch.from_numpy(v) for o,v in zip(session.get_outputs(), outputs)}}, {'provider':'CPUExecutionProvider','input_name':session.get_inputs()[0].name,'output_order':[o.name for o in session.get_outputs()]},'network')
    landmarks = np.zeros((110,2),dtype=np.int32)
    landmarks[1] = [20,40]; landmarks[31] = [180,40]; landmarks[52] = [90,30]
    landmarks[90:] = np.stack([np.arange(20)*3+65,np.arange(20)%4*4+110],axis=1)
    frame = (np.arange(220*240*3)%256).astype(np.uint8).reshape(220,240,3)
    bbox = face.compute_face_bbox(landmarks)
    crop = face.crop_face(frame,bbox)
    inner = face.extract_inner(crop)
    # Freeze all initial character parameters, rather than requiring identical RNGs in Rust.
    torch.manual_seed(SEED)
    character = visual.Model()
    save('character_initial_state', character.state_dict(), {'architecture':'FeatherTalk.Model','image_size':144,'n_channels':6,'audio_tokens':40,'audio_dim':1024,'seed':SEED}, 'exact')
    # Mirror TemporalMouthRoiDataset: one reference, two adjacent targets,
    # source mouth masking and landmark ROI masks from those same targets.
    reference_bgr = face.extract_inner(face.crop_face(np.roll(frame,9,axis=1),bbox))
    reference = face.hwc_to_chw_tensor(reference_bgr)
    pair_frames = np.stack([frame,np.roll(frame,1,axis=1)])
    pair_landmarks = np.stack([landmarks,landmarks.copy()])
    pair_landmarks[1,90:] += [2,1]
    targets, masked_targets, mouth_masks = [], [], []
    for target_frame, points in zip(pair_frames,pair_landmarks):
        target_inner = face.extract_inner(face.crop_face(target_frame,face.compute_face_bbox(points)))
        targets.append(face.hwc_to_chw_tensor(target_inner.copy()))
        masked_targets.append(face.hwc_to_chw_tensor(face.mask_mouth(target_inner.copy())))
        mouth_masks.append(roi.mouth_mask_from_landmarks(points,roi.MouthRoiConfig()))
    labels = torch.stack(targets)[None]
    masks = torch.stack(mouth_masks)[None]
    image = torch.stack([torch.cat([reference,masked],dim=0) for masked in masked_targets]).requires_grad_()
    assert torch.equal(image[0,:3],image[1,:3]), 'Adjacent frames must share one reference'
    audio_features = torch.randn(22,2,1024).numpy()
    audio = torch.stack([face.reshape_audio_feat(face.gather_audio_window(audio_features,index)) for index in (10,11)]).requires_grad_()
    network('character_eval',character,(image.detach(),audio.detach()),['inc','down2','down4','audio_model','spatial_x3','fuse_conv','up4','outc'])
    character.train()
    prediction = character(image,audio)
    perceptual = train.PerceptualLoss.__new__(train.PerceptualLoss)
    perceptual.content, perceptual.criterion = vgg, torch.nn.MSELoss()
    total, parts = train.compute_total_loss(prediction.reshape(1,2,3,144,144),labels,masks,torch.nn.L1Loss(),perceptual,4.0,0.5,4.0,0.01,0.5,5)
    total.backward()
    named = dict(character.named_parameters())
    # Select by module families while retaining actual source names in the index.
    chosen = [next(k for k,p in named.items() if k.startswith(prefix) and p.grad is not None) for prefix in ['inc.','audio_model.','spatial_x3.query.','spatial_x5.residual_gate','outc.']]
    save('character_training',{'image':image,'audio':audio,'labels':labels,'masks':masks,'prediction':prediction,'total':total,
        'image_gradient':image.grad,'audio_gradient':audio.grad,'pair_frames_bgr':pair_frames,'pair_landmarks':pair_landmarks,'reference_bgr':reference_bgr,**{'loss.'+k:torch.tensor(v) for k,v in parts.items()},
        **{'gradient.'+k:named[k].grad for k in chosen}, **{'updated_state.'+k:v for k,v in character.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}},
        {'mode':'train','pair_shape':[1,2],'frame_indices':[10,11],'shared_reference':True,'input_channel_order':['reference_Bgr','masked_target_Bgr'],'target_source':'deterministic geometry frame and one-pixel shift','loss_coefficients':{'full':1,'mouth':4,'temp':0.5,'temp_mouth':4,'percep':0.01,'mouth_grad':0.5},'temporal_lowpass_kernel':5,'gradients':chosen}, 'gradient')
    # Independent derivatives isolate each original loss's backward semantics.
    pred = prediction.detach().clone().reshape_as(labels).requires_grad_()
    flat, target, flat_masks = pred.flatten(0,1), labels.flatten(0,1), masks.flatten(0,1)
    temporal, temporal_mouth = train.temporal_delta_losses(pred, labels, masks, torch.nn.L1Loss(), lowpass_kernel=5)
    terms = {'full': F.l1_loss(flat,target), 'mouth': train.mouth_l1_loss(flat,target,flat_masks),
        'temp': temporal, 'temp_mouth': temporal_mouth, 'percep': perceptual(flat,target),
        'mouth_grad': train.mouth_gradient_loss(flat,target,flat_masks)}
    term_data = {'prediction':pred,'labels':labels,'masks':masks}
    for key, value in terms.items():
        term_data['loss.'+key] = value
        term_data['gradient.'+key] = torch.autograd.grad(value,pred,retain_graph=True)[0]
    save('loss_individual_terms',term_data,{'coefficients':'unweighted individual loss derivatives','lowpass_kernel':5},'gradient')
    # Loss edge conditions: disjoint masks, no ROI, and temporal supervision disabled.
    for case, mask, temporal in [('empty',torch.zeros_like(masks),True),('disabled',masks,False)]:
        pred = prediction.detach().clone().reshape_as(labels).requires_grad_()
        loss, detail = train.compute_total_loss(pred,labels,mask,torch.nn.L1Loss(),perceptual,4,0.5 if temporal else 0,4 if temporal else 0,0.01,0.5,5)
        loss.backward()
        save('loss_'+case,{'prediction':pred,'labels':labels,'masks':mask,'total':loss,'gradient':pred.grad,**{'loss.'+k:torch.tensor(v) for k,v in detail.items()}},{'temporal_enabled':temporal,'lowpass_kernel':5},'gradient')
    save('geometry',{'landmarks':landmarks,'frame_bgr':frame,'bbox':torch.tensor(bbox),'crop':crop,'inner':inner,'masked':face.mask_mouth(inner.copy()),'chw':face.hwc_to_chw_tensor(inner),'fill_mask':face.mouth_fill_mask(),'roi_mask':roi.mouth_mask_from_landmarks(landmarks,roi.MouthRoiConfig()),'roi_bbox':torch.tensor(roi.mouth_roi_from_landmarks(landmarks,roi.MouthRoiConfig()))},{'crop':152,'inner':144,'color':'BGR','resize':'INTER_AREA'},'exact')
    for case, offset in [('left_top',-180),('right_bottom',180),('collapsed',0)]:
        points = landmarks.copy()
        points[90:] = ([90,110] if case == 'collapsed' else points[90:] + offset)
        save('geometry_roi_'+case,{'landmarks':points,'roi_mask':roi.mouth_mask_from_landmarks(points,roi.MouthRoiConfig()),
            'roi_bbox':torch.tensor(roi.mouth_roi_from_landmarks(points,roi.MouthRoiConfig()))},
            {'roi':'clamp to image; retain at least one pixel; min ROI width/height apply'},'exact')
    from detect_face import SCRFD
    detector = SCRFD(str(assets/'scrfd_2_5g/architecture.onnx'),confThreshold=0.1,nmsThreshold=0.5)
    for case, height, width in [('portrait',101,57),('landscape',57,101)]:
        bgr = (np.arange(height*width*3)%256).astype(np.uint8).reshape(height,width,3)
        resized,newh,neww,padh,padw = detector.resize_image(bgr)
        blob = cv2.dnn.blobFromImage(resized,1.0/128,(640,640),(127.5,127.5,127.5),swapRB=True)
        save('detector_preprocess_'+case,{'bgr':bgr,'resized':resized,'blob':blob,'geometry':torch.tensor([newh,neww,padh,padw])},
            {'keep_ratio':True,'resize':'INTER_AREA','swap_rb':True,'mean':127.5,'scale':1/128},'exact')
    anchors = np.array([[0,0],[8,0],[8,8]],dtype=np.float32)
    distances = np.array([[1,2,3,4],[3,2,1,4],[0,0,2,2]],dtype=np.float32)
    keypoints = np.arange(30,dtype=np.float32).reshape(3,10)/2
    boxes = np.array([[0,0,10,10],[1,1,10,10],[30,30,10,10]],dtype=np.float32)
    scores = np.array([0.9,0.8,0.05],dtype=np.float32)
    save('detector_decode',{'anchors':anchors,'distances':distances,'keypoint_distances':keypoints,
        'decoded_boxes':detector.distance2bbox(anchors,distances),'decoded_keypoints':detector.distance2kps(anchors,keypoints),
        'nms_boxes_xywh':boxes,'scores':scores,'nms_indices':torch.as_tensor(cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),0.1,0.5))},
        {'bbox':'xyxy = anchor +/- distance; detection NMS boxes are xywh','score_threshold':0.1,'nms_threshold':0.5},'exact')
    feats = np.arange(24*2*1024,dtype=np.float32).reshape(24,2,1024)/1024
    for case,index,start,end in [('first',0,0,24),('last',23,0,24),('clip_first',8,8,14),('clip_last',13,8,14)]:
        window = face.gather_audio_window(feats,index,valid_start=start,valid_end=end)
        save('audio_'+case,{'features':feats,'window':window,'tokens':face.reshape_audio_feat(window)}, {'index':index,'valid_start':start,'valid_end':end,'half_window':10}, 'exact')
    boundaries = [0,1,399,400,719,720,1039,1040,16000,320000,320080,320400]
    save('audio_frame_counts',{'samples':torch.tensor(boundaries),'frames':torch.tensor([hubert_module.expected_hubert_frames(n) for n in boundaries])},{'kernel':400,'stride':320},'exact')
    for count in [399,400,719,720,1040,320000,320080,320400]:
        speech = np.sin(np.arange(count,dtype=np.float32)*0.017)
        hidden = hubert_module.get_feather_hubert_from_16k_speech(speech,hubert,device=torch.device('cpu'))
        save('audio_extract_'+str(count),{'speech':speech,'hidden':hidden,'even':hubert_module.make_even_first_dim(hidden)}, {'normalize':True,'chunk_samples':320000},'network')
    for case, speech in [('silence',np.zeros(720,dtype=np.float32)),('stereo',np.stack([np.sin(np.arange(720,dtype=np.float32)*0.017),np.ones(720,dtype=np.float32)],axis=1))]:
        hidden = hubert_module.get_feather_hubert_from_16k_speech(speech,hubert,device=torch.device('cpu'))
        save('audio_extract_'+case,{'speech':speech,'hidden':hidden},{'normalize':True,'epsilon':1e-7,'stereo_channel':'first'},'network')
    versions = {name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','safetensors','onnx','onnxruntime','opencv-python-headless','pytest']}
    source_paths = ['model.py','train.py','face_utils.py','dataset_mouth_roi.py','data_utils/base_module.py','data_utils/pfld_mobileone.py','data_utils/feather_hubert/feather_hubert.py']
    manifest = {'schema_version':1,'seed':SEED,'device':'cpu','dtype':'float32','threads':1,'python':platform.python_version(),'versions':versions,
        'assets_manifest_sha256':sha256(assets/'manifest.json'),'source_hashes':{p:sha256(source_dir/p) for p in source_paths},
        'tolerance_rationale':'Prospective FP32 gates: 2e-6 absolute operator/backward and 2e-5 network; relative gates allow reduction-order drift. Near-zero gradients retain absolute gate. Must report and investigate failures; no automatic widening. Same-environment regeneration must be byte-identical.',
        'effect_validation':'pending: real training and acceptance footage not supplied; synthetic tensors test numerical behavior only', 'cases':entries}
    write_json(output_dir/'manifest.json',manifest)
    return output_dir/'manifest.json'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    print(generate_reference(a.source,a.assets,a.output))
