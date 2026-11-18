"""Versioned fixed-640 SCRFD workaround for onnx-ir 0.22.0 ceil-mode validation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
SOURCE='32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a'
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prepare(source,destination):
    assert digest(source)==SOURCE, 'Unexpected SCRFD source graph'
    model=onnx.load(source)
    inferred=onnx.shape_inference.infer_shapes(model)
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*inferred.graph.input,*inferred.graph.value_info]}
    expected={'AveragePool_25':([1,24,160,160],80),'AveragePool_52':([1,48,80,80],40),'AveragePool_69':([1,48,40,40],20)}
    proof=[]
    for node in model.graph.node:
        if node.op_type!='AveragePool': continue
        shape,count=expected.pop(node.name)
        assert shapes[node.input[0]]==shape
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs=={'ceil_mode':1,'kernel_shape':[2,2],'pads':[0,0,0,0],'strides':[2,2]}
        assert all((v-2)%2==0 and (v-2)//2+1==count for v in shape[2:])
        for attr in node.attribute:
            if attr.name=='ceil_mode': attr.i=0
        proof.append({'node':node.name,'input':shape,'output_spatial':[count,count]})
    assert not expected
    onnx.checker.check_model(model)
    onnx.save(model,destination)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    original=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider'])
    derived=ort.InferenceSession(str(destination),options,providers=['CPUExecutionProvider'])
    # First input exactly matches the frozen original fixture generator.
    inputs=[torch.linspace(-1,1,3*640*640).reshape(1,3,640,640).numpy(),
            ((np.arange(3*640*640,dtype=np.int32)%257)/128-1).astype(np.float32).reshape(1,3,640,640)]
    for value in inputs:
        a=original.run(None,{'images':value}); b=derived.run(None,{'images':value})
        assert len(a)==len(b)==9
        assert all(np.array_equal(x,y) for x,y in zip(a,b)), 'Derived graph changes outputs'
    provenance={'transform_version':'scrfd-fixed640-redundant-ceil-v1','source_sha256':SOURCE,'derived_sha256':digest(destination),'proof':proof,'ort_equivalence':'9 outputs bit-identical on 2 deterministic inputs','generator':'burn-onnx=0.22.0','param_id_strategy':'sha256-full-tensor-path-first64le-v1-collision-checked'}
    Path(destination).with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance))
if __name__=='__main__': prepare(Path(sys.argv[1]),Path(sys.argv[2]))
