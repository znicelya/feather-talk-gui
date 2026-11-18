"""Source-only final landmark/crop oracle; consumes no NumPy or Torch RNG."""
from pathlib import Path
import sys
from audit import sha256


def coordinate_cases(source_dir):
    import cv2
    import numpy as np
    import torch
    source_dir = Path(source_dir)
    sys.path.insert(0, str(source_dir / 'data_utils'))
    from get_landmark import Landmark, face_det
    mean = np.fromstring((source_dir / 'data_utils/mean_face.txt').read_text(), sep=' ', dtype=np.float32)
    # Force negative fractional local coordinates, making cast-before-origin observable.
    desired = np.linspace(-.35, 1.2, 220, dtype=np.float32)
    raw = (desired - mean).astype(np.float32)
    frame = (np.arange(37*53*3, dtype=np.uint32) % 251).astype(np.uint8).reshape(37,53,3)
    class Network:
        def __call__(self, value):
            self.input = value.clone()
            return torch.from_numpy(raw.copy())[None]
    class Detector:
        def __init__(self, boxes):
            self.boxes = np.asarray(boxes, dtype=np.float32)
        def detect(self, frame):
            return self.boxes, np.arange(len(self.boxes)), np.tile(np.array([[[1,1],[2,1],[0,0],[0,0],[0,0]]], dtype=np.float32), (len(self.boxes),1,1))
    for name, boxes in [('left_top', [[-3.7,-2.4,22.8,24.1],[20,10,30,25]]),
                        ('right_bottom', [[37.6,22.4,20.2,21.8]]),
                        ('negative_center', [[-13.2,-12.8,19.9,20.4]])]:
        model = Landmark.__new__(Landmark)
        model.device = torch.device('cpu')
        model.mean_face = mean.copy()
        model.det_net = Detector(boxes)
        model.pfld_backbone = Network()
        original_imread = cv2.imread
        try:
            cv2.imread = lambda _: frame.copy()
            local, x, y = model.detect('controlled-source-only-frame')
        finally:
            cv2.imread = original_imread
        crops, _, _, _ = face_det(frame.copy(), model.det_net)
        points = local + np.array([x,y], dtype=np.int32)
        assert points.shape == (110,2)
        assert np.any(local < 0)
        yield 'landmark_coordinates_' + name, {
            'frame': frame, 'boxes_xywh': np.asarray(boxes,dtype=np.float32),
            'raw': raw, 'mean_face': mean, 'crop': crops[0],
            'pfld_input': model.pfld_backbone.input,
            'origin': np.array([x,y],dtype=np.int32), 'local':local, 'points':points,
        }, {'method':'original Landmark.detect and face_det with controlled detector and raw prediction; no RNG',
            'source_sha256':sha256(source_dir/'data_utils/get_landmark.py'),
            'mean_face_sha256':sha256(source_dir/'data_utils/mean_face.txt'),
            'first_face_only':True, 'expansion':1.05, 'local_cast_before_origin':True}
