from onnxsim import simplify
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
output_onnx = 'weights/mobilev3Fpn_20210427.onnx'
sim_onnx = 'weights/mobilev3Fpn_20210427_simplify.onnx'
onnx_model = onnx.load(output_onnx)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, sim_onnx)
print('finished exporting onnx')