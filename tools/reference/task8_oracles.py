"""Focused source-only geometry oracles. Does not import or advance any RNG."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
import cv2
import numpy as np
import face_utils as face
import dataset_mouth_roi as roi
from safetensors.numpy import save_file

args.output.mkdir(parents=True, exist_ok=True)
entries = {}
for size in [71, 100, 151, 153, 159, 161, 213]:
    frame = (np.arange(260*270*3) % 256).astype(np.uint8).reshape(260,270,3)
    landmarks = np.zeros((110,2), np.int32)
    landmarks[1] = [12, 40]
    landmarks[31] = [12+size, 40]
    landmarks[52] = [90, 20]
    landmarks[90:] = np.stack([np.arange(20)*3+45, np.arange(20)%4*4+80],axis=1)
    if size == 100:
        landmarks[90:] = [41,80]
        landmarks[100:,0] += 35
    inner = face.extract_inner(face.crop_face(frame,face.compute_face_bbox(landmarks)))
    tensors = dict(frame=frame,landmarks=landmarks,target=face.hwc_to_chw_tensor(inner.copy()).numpy(),
                   masked=face.hwc_to_chw_tensor(face.mask_mouth(inner.copy())).numpy(),
                   mask=roi.mouth_mask_from_landmarks(landmarks,roi.MouthRoiConfig()).numpy())
    path=args.output/f'geometry_{size}.safetensors'
    save_file(tensors,path)
    entries[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=dict(opencv=cv2.__version__,numpy=np.__version__,files=entries,
              sources={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in ['face_utils.py','dataset_mouth_roi.py']})
(args.output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
