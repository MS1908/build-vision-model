import cv2
import torch
from glob import glob
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from vision_classification import albu_img_prepro

IMAGE_EXTS = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']


def mean_std_stats(dataset, bs=32, verbose=False):
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=2)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(dataset) / h / w

    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(dataset) * h * w - 1))
    
    mean = mean.view(-1).numpy() / 255.
    std = std.view(-1).numpy() / 255.
    
    if verbose:
        print(f"MEAN: {mean}")
        print(f"STD: {std}")
    
    return mean, std


def image_size_stats(image_root):
    image_paths = []
    for image_ext in IMAGE_EXTS:
        image_paths.extend(glob(image_root + f'/**/*.{image_ext}', recursive=True))
    height_stats = {}
    width_stats = {}
    ratio_stats = {}
    average_height = 0.
    average_width = 0.
    vertical_rect = 0
    horizontal_rect = 0
    square = 0
    average_vert_ratio = 0.
    averate_horiz_ratio = 0.
    for image_path in tqdm(image_paths, total=len(image_paths)):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        height_stats[h] = height_stats.get(h, 0) + 1
        width_stats[w] = width_stats.get(w, 0) + 1
        average_height += h
        average_width += w
        if h > w:
            vertical_rect += 1
            average_vert_ratio += h / w
        elif h < w:
            horizontal_rect += 1
            averate_horiz_ratio += h / w
        else:
            square += 1
        ratio_stats[h / w] = ratio_stats.get(h / w, 0) + 1
        
        
    average_height /= len(image_paths)
    average_width /= len(image_paths)

    print('Average height:', average_height)
    print('Average width:', average_width)
    print('Vertical rectangle', vertical_rect)
    print('Horizontal rectangle', horizontal_rect)
    print('Square', square)
    if vertical_rect != 0:
        average_vert_ratio /= vertical_rect
        print('Average h/w ratio of vertical rect', average_vert_ratio)
    if horizontal_rect != 0:
        averate_horiz_ratio /= horizontal_rect
        print('Average h/w ratio of horizontal rect', averate_horiz_ratio)
        
        
def grad_cam_visualize(input_image, model_arch, model_param,
                       device=None, imgsz=None, 
                       mean=None, std=None):
    image = albu_img_prepro(input_image.copy(), imgsz=imgsz,
                            mean=mean, std=std,
                            to_rgb=True, squeeze=False, to_np=False)
    
    rgb_image = albu_img_prepro(input_image.copy(), imgsz=imgsz,
                                mean=mean, std=std,
                                to_rgb=True, squeeze=True, 
                                channel_format='HWC', to_np=False)
    rgb_image = torch.clamp(rgb_image, min=0., max=1.)
    rgb_image = rgb_image.cpu().numpy()
    
    if device is not None:
        image = image.to(device)
    
    if 'efficientnet' in model_arch:
        target_layers = [model_param.conv_head, model_param.bn2]
    elif 'convnext' in model_arch:
        target_layers = [model_param.stages[-1].blocks[-1].conv_dw]
    else:
        raise NotImplementedError(f"Grad CAM visualization is not implemented for {model_arch}")
    
    cam = GradCAM(model=model_param, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return visualization
