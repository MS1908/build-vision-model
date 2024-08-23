import argparse
import cv2
import time
import torch
import yaml
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_performance_report_utils import compute_stats, plot_multiclass_pr_curve, plot_binary_pr_curve
from data_utils import ImageDataset, albu_img_prepro
from deep_learning_utils import (classify, binary_classify, classify_onnx, binary_classify_onnx, 
                                 build_timm_model)
from misc_utils import model_summary


def model_evaluation(
    image_config,
    model_config,
    infer_config,
    dataset_config,
    device_id=-1,
    comp_stat=True
):
    val_ds = ImageDataset(dataset_config.get('root', None), 
                          dataset_config.get('annot', None), 
                          return_image=False)
    literal_labels = val_ds.get_literal_labels() or dataset_config.get('lit_labels', None)

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() and device_id != -1 else "cpu"

    ckpt_path = model_config['ckpt_path']
    if ckpt_path.endswith('.onnx'):
        print('Model format: ONNX')
        
        onnx_infer = True
        model = ort.InferenceSession(
            ckpt_path, None,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    else:
        print('Model format: PyTorch')
        
        onnx_infer = False
        model = build_timm_model(config=model_config, device=device, phase='val')
        model_summary(model)
    
    infer_mode = infer_config.get('mode', 'multiclass')
    assert infer_mode in ['multiclass', 'binary'], "Inference mode must be in ['multiclass', 'binary']"
    
    threshold = infer_config.get('thres', None)
    imgsz = image_config.get('imgsz', (224, 224))
    mean = image_config.get('mean', (0.485, 0.456, 0.406))
    std = image_config.get('std', (0.229, 0.224, 0.225))
    to_rgb = image_config.get('to_rgb', True)
    
    y_true = []
    y_pred = []
    probs = []
    failed = []
    results = []
    total_time = 0.
    
    no_ground_truth = False
    for path, gt in tqdm(val_ds, total=len(val_ds)):
        image = cv2.imread(path)
        
        start = time.time()
        
        input_image = albu_img_prepro(image, imgsz=imgsz, mean=mean, std=std,
                                      to_rgb=to_rgb, squeeze=False, to_np=onnx_infer)
        if gt == -1:  # If gt = -1, that means this dataset doesn't have ground truth
            no_ground_truth = True
        
        if infer_mode == 'binary':
            if onnx_infer:
                label, logit = binary_classify_onnx(input_image, model, threshold)
            else:
                label, logit = binary_classify(input_image, model, device, threshold)
                
        else:
            if onnx_infer:
                label, logit = classify_onnx(input_image, model)
            else:
                label, logit = classify(input_image, model, device)
                
        elapsed_time = time.time() - start
        total_time += elapsed_time
                
        if no_ground_truth:
            results.append((path, label, logit))
        else:
            results.append((path, gt, label, logit))

        if not no_ground_truth:
            if gt != label:
                failed.append((path, gt, label))
            y_true.append(gt)
        
        y_pred.append(label)
        if infer_mode == 'binary':
            probs.append(logit[1])
        else:
            probs.append(logit)
            
    if comp_stat and (not no_ground_truth):
        try:
            compute_stats(y_true, y_pred, literal_labels, mode=infer_mode, cm_plot_name='confusion_matrix.jpg')
            plt.cla()
            plt.clf()
            if infer_mode == 'binary':
                plot_binary_pr_curve(y_true, probs, pr_plot_name='precision_recall_curve.jpg')
            else:
                plot_multiclass_pr_curve(y_true, probs, literal_labels, pr_plot_name='precision_recall_curve.jpg')
        except ValueError:
            print("Can\'t calculate stats")
    print(f'Avg infer time per image: {total_time / len(val_ds):.3f}s')
            
    if no_ground_truth:
        df = pd.DataFrame(results, columns=['path', 'label', 'logits'])
        print(df['label'].value_counts())
        df.to_csv('inference_log.csv', lineterminator='\n', index=False)
        
    else:
        df = pd.DataFrame(results, columns=['path', 'gt', 'label', 'logits'])
        print(df['label'].value_counts())
        df.to_csv('inference_log.csv', lineterminator='\n', index=False)
        
        df = pd.DataFrame(failed, columns=['path', 'gt', 'label'])
        print(df['label'].value_counts())
        df.to_csv('failed_cases_log.csv', lineterminator='\n', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    infer_config = config['infer_config']
    model_config = config['model']
    dataset_config = config['dataset']
    image_config = config['image_config']
    device_id = config['device_id']
    
    model_evaluation(
        image_config=image_config,
        model_config=model_config,
        infer_config=infer_config,
        dataset_config=dataset_config,
        device_id=device_id,
        comp_stat=True
    )
