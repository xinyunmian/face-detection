from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from ParseYamlModel import Yaml2Pytorch
from FaceConfig import facecfg
from FaceUtils import *

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    net = Yaml2Pytorch(cfg=facecfg.yaml_model, ch=3, nc=1)  # 需要修改
    anchors = net.yaml["anchors"]
    anchors = np.array(anchors, np.float32).reshape([3, 3, 2])

    netW = 640
    netH = 640
    weight_path = "weights/FaceShuffle_150.pth"
    # load weight
    net = load_model(net, weight_path, True)
    net.eval()
    print('Finished loading model!')
    device = torch.device("cpu")
    net = net.to(device)

    ##################export###############
    output_onnx = 'D:/codes/pytorch_projects/pytorch2ncnn/yolo5face.onnx'  #统一在该文件夹下转换
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["img"]
    output_names = ["out8", "out16" , "out32"]
    inputs = torch.randn(1, 3, netH, netW).to(device)
    torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, input_names=input_names, output_names=output_names)

    from onnxsim import simplify
    import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
    import onnx

    sim_onnx = 'D:/codes/pytorch_projects/pytorch2ncnn/yolo5face_simplify.onnx'
    onnx_model = onnx.load(output_onnx)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, sim_onnx)
    print('finished exporting onnx')
    ##################end###############




