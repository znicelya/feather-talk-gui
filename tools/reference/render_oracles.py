"""Independent source-only renderer cases; never rewrites existing reference cases."""
from pathlib import Path
import hashlib, json, sys, tempfile
import numpy as np
import torch
from safetensors.numpy import save_file

source, output = map(Path, sys.argv[1:])
if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
    raise ValueError('Renderer oracles must be outside the read-only source tree')
sys.dont_write_bytecode = True
sys.path.insert(0, str(source.resolve()))
import inference
output.mkdir(parents=True, exist_ok=True)
cases = {}
for size in (151, 152, 161, 203):
    width, height = size+24, size+32
    image = (np.arange(height*width*3, dtype=np.int64)*37 % 256).astype(np.uint8).reshape(height,width,3)
    landmarks = np.zeros((110,2),dtype=np.float32)
    landmarks[1,0], landmarks[31,0], landmarks[52,1] = 8, 8+size, 12
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        path=Path(tmp)/'points.lms'; np.savetxt(path,landmarks)
        model_input,crop,bbox,original_size=inference.prepare_model_input(image.copy(),str(path),torch.device('cpu'))
    prediction=np.linspace(-0.01,1.01,3*144*144,dtype=np.float32).reshape(3,144,144)
    cast=np.clip(prediction.transpose(1,2,0)*255.,0,255).astype(np.uint8)
    final=image.copy(); mutated=crop.copy()
    inference.paste_prediction(final,cast,mutated,bbox,original_size)
    name=f'render_{size}.safetensors'
    save_file(dict(image=image,landmarks=landmarks.astype(np.int32),input=model_input.numpy(),crop=crop,prediction=prediction,cast=cast,mutated=mutated,final=final),str(output/name))
    cases[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
pickers={}
import face_utils
for count in (1,2,24):
    features=np.arange(count*2048,dtype=np.float32).reshape(count,2,1024)
    windows=np.stack([face_utils.reshape_audio_feat(face_utils.gather_audio_window(features,k,half_window=10)).numpy() for k in range(count)])
    name=f'windows_{count}.safetensors'
    save_file(dict(features=features,windows=windows),str(output/name))
    cases[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
for n in (2,3,4,5):
    p=inference.FramePicker(n); pickers[n]=[p.next() for _ in range(4*(n-1)+1)]
(output/'manifest.json').write_text(json.dumps(dict(sources={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('inference.py','face_utils.py')},cases=cases,pickers=pickers),indent=2))
