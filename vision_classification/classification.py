import torch
import torch.nn.functional as F
import numpy as np
from scipy import special


def binary_classify(input_tensor, model, device=None, threshold=None):
    """Binary classification using PyTorch deep learning model

    Args:
        input_tensor (torch.Tensor): Input tensor
        model (nn.Module): Classification model
        device (torch.device, optional): Device for inference. Defaults to None.
        threshold (float, optional): Positive threshold. Defaults to None.
        
    Return:
        Tuple[int, torch.Tensor]: Label and logits
    """
    if device is not None:
        input_tensor = input_tensor.to(device)
        model.to(device)
        
    model.eval()
    with torch.no_grad():
        outputs = model.forward(input_tensor)
        logit = F.softmax(outputs, dim=-1).cpu().numpy()

    if threshold is None:
        label = np.argmax(logit)
    else:
        # binary classification, hence indices of labels are only 0 and 1. Assuming positive label is 1.
        label = 1 if logit[0][1] >= threshold else 0

    return label, logit.squeeze()


def classify(input_tensor, model, device=None):
    """Classification using PyTorch deep learning model

    Args:
        input_tensor (torch.Tensor): Input tensor
        model (nn.Module): Classification model
        device (torch.device, optional): Device for inference. Defaults to None.
        
    Return:
        Tuple[int, torch.Tensor]: Label and logits
    """
    if device is not None:
        input_tensor = input_tensor.to(device)
        model.to(device)
        
    model.eval()
    with torch.no_grad():
        outputs = model.forward(input_tensor)
        logit = F.softmax(outputs, dim=-1).cpu().numpy()
        
    label = np.argmax(logit)
    
    return label, logit.squeeze()


def binary_classify_onnx(input_tensor, ort_sess, threshold=None):
    """Binary classification using ONNX deep learning model

    Args:
        input_tensor (np.ndarray): Input tensor
        ort_sess (ort.InferenceSession): Classification ONNX session
        threshold (float, optional): Positive threshold. Defaults to None.
        
    Return:
        Tuple[int, np.ndarray]: Label and logits
    """
    inp_name = ort_sess.get_inputs()[0].name
    out_name = ort_sess.get_outputs()[0].name

    outputs = ort_sess.run([out_name], {inp_name: input_tensor})[0]
    logit = special.softmax(outputs)

    if threshold is None:
        label = np.argmax(logit)
    else:
        # binary classification, hence indices of labels are only 0 and 1. Assuming positive label is 1.
        label = 1 if logit[0][1] >= threshold else 0

    return label, logit.squeeze()


def classify_onnx(input_tensor, ort_sess):
    """Classification using ONNX deep learning model

    Args:
        input_tensor (np.ndarray): Input tensor
        ort_sess (ort.InferenceSession): Classification ONNX session
        
    Return:
        Tuple[int, np.ndarray]: Label and logits
    """
    inp_name = ort_sess.get_inputs()[0].name
    out_name = ort_sess.get_outputs()[0].name

    outputs = ort_sess.run([out_name], {inp_name: input_tensor})[0]
    logit = special.softmax(outputs)

    label = np.argmax(logit)

    return label, logit.squeeze()
