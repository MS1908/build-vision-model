import argparse
import onnx
import os
import shutil
import torch
import timm
import numpy as np
import tensorflow as tf
from onnx_tf.backend import prepare
import warnings

from utils import albu_img_aug_pipeline
from dl_utils import CustomSampleImageDataset

warnings.filterwarnings("ignore")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
        
        
class RepresentativeDatasetGen:
    
    def __init__(self, path, input_h=224, input_w=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), num_calib=-1):
        transform = albu_img_aug_pipeline(
            phase='export',
            imgsz=(input_h, input_w),
            mean=mean,
            std=std,
            to_tensor=False
        )
        self.dataset = CustomSampleImageDataset(
            image_root=path,
            annotation_file=None,
            transforms_pipeline=transform,
            class_sampling_ratio=None,
            return_image=True
        )

        self.num_calibration_images = num_calib if num_calib != -1 else len(self.dataset)
        
    def __call__(self):
        for i in range(self.num_calibration_images):
            image = self.dataset[i][0]
            image = np.transpose(image, (1, 2, 0))
            image = np.expand_dims(image, axis=0)
            image = tf.convert_to_tensor(image)
            yield [image]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, help='Model architecture', default=None)
    parser.add_argument('--img-h', type=int, help='Height of model inputs', default=None)
    parser.add_argument('--img-w', type=int, help='Width of model inputs', default=None)
    parser.add_argument('--rep-dataset', type=str, help='Representative dataset path', default=None)
    parser.add_argument('--classes', type=int, help='Number of output classes', default=2)
    parser.add_argument('--int-quantize', action='store_true')
    parser.add_argument('--path', type=str, help='Weight path')
    parser.add_argument('--tflite-path', type=str, help='Export path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, w, arch = args.img_h, args.img_w, args.arch

    print('Building PyTorch model...')
    model = timm.create_model(arch, num_classes=args.classes)
    state_dict = torch.load(args.path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, h, w)
    input_tensor = input_tensor.to(device)

    result = model(input_tensor)
    input_names = ["input"]
    output_names = ["output"]

    print('Exporting to ONNX...')
    torch.onnx.export(model, input_tensor, './tmp_onnx.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)
    
    print('Loading ONNX...')
    onnx_model = onnx.load('./tmp_onnx.onnx')
    onnx.checker.check_model(onnx_model)
    
    print('Exporting to TF Saved Model...')
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('./tmp_tf')
    
    converter = tf.lite.TFLiteConverter.from_saved_model('./tmp_tf')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    if not args.int_quantize:
        print('Exporting to FP16 TFLite...')
        
        converter.target_spec.supported_types = [tf.float16]
        tflite_model  = converter.convert()
        with open(args.tflite_path, 'wb') as f:
            f.write(tflite_model)
    
    else:
        assert args.rep_dataset is not None, "Need representative dataset for quantization"
        
        print('Exporting to INT8 TFLite...')
        
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        rep_dataset = RepresentativeDatasetGen(path=args.rep_dataset, input_h=h, input_w=w)
        converter.representative_dataset = rep_dataset
        tflite_model  = converter.convert()
        with open(args.tflite_path, 'wb') as f:
            f.write(tflite_model)
        
    os.remove('./tmp_onnx.onnx')
    shutil.rmtree('./tmp_tf')
