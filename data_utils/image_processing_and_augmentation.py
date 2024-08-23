import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def build_resize_pipeline_albu(imgsz=(224, 224)):
    if isinstance(imgsz, tuple) or isinstance(imgsz, list):
        assert len(imgsz) <= 2, "List/tuple of sizes must be of format (h, w) or (imgsz,)"
        
        if len(imgsz) == 2:
            target_h, target_w = imgsz
            resize_pipeline = A.Resize(height=target_h, width=target_w)
        else:
            # Only longest size is provided
            resize_pipeline = [
                A.LongestMaxSize(max_size=imgsz[0]),
                A.PadIfNeeded(min_height=imgsz[0], min_width=imgsz[0],
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            ]
    else:
        # Only longest size is provided
        resize_pipeline = [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz,
                          border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ]
    return resize_pipeline


def preprocess_albu(
    image, 
    imgsz=(224, 224), 
    mean=(0.485, 0.456, 0.406), 
    std=(0.229, 0.224, 0.225), 
    to_rgb=False, 
    to_np=False, 
    squeeze=False, 
    channel_format='CHW'
):
    if to_rgb:
        ret_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    else:
        ret_image = image.copy()

    preprocess_pipeline = build_resize_pipeline_albu(imgsz=imgsz)
    
    # Add normalization and conversion to PyTorch tensor.
    preprocess_pipeline.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    preprocessor = A.Compose(preprocess_pipeline)
    ret_image = preprocessor(image=ret_image)['image']

    if not squeeze:  # Include batch size dimension for input tensor ([B, ...])
        ret_image = ret_image.unsqueeze(0)
    if to_np:
        ret_image = ret_image.cpu().numpy()
    
    if channel_format == 'HWC':  # Swap channel CHW -> HWC
        ret_image = np.transpose(ret_image, (1, 2, 0))
    elif channel_format != 'CHW':
        raise ValueError("Invalid channel format. Must be either CHW or HWC")

    return ret_image


def resize_np(img, imgsz=(224, 224)):
    ret_img = img.copy()  # Don't mutate original image

    if isinstance(imgsz, tuple) or isinstance(imgsz, list):
        assert len(imgsz) <= 2, "List/tuple of sizes must be of format (h, w) or (imgsz,)"
        
        if len(imgsz) == 2:
            target_h, target_w = imgsz
            ret_img = cv2.resize(ret_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return ret_img
        
        else:
            max_sz = imgsz[0]
    
    else:
        max_sz = imgsz

    # Resize the image while preserving the aspect ratio to fit within the max_size
    h, w = ret_img.shape[:2]
    scale = max_sz / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    ret_img = cv2.resize(ret_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    delta_w = max_sz - new_w
    delta_h = max_sz - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    ret_img = cv2.copyMakeBorder(ret_img, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return ret_img


def preprocess_np(
    image, 
    imgsz=(224, 224), 
    mean=(0.485, 0.456, 0.406), 
    std=(0.229, 0.224, 0.225), 
    to_rgb=False, 
    squeeze=False, 
    channel_format='CHW'
):
    if to_rgb:
        ret_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    else:
        ret_image = image.copy()

    # Apply resizing
    ret_image = resize_np(ret_image, imgsz=imgsz)

    # Normalize the image
    ret_image = ret_image.astype(np.float32) / 255.0
    ret_image -= np.array(mean)
    ret_image /= np.array(std)

    if channel_format == 'CHW':  # Swap channel HWC -> CHW
        ret_image = np.transpose(ret_image, (2, 0, 1))
    elif channel_format != 'HWC':
        raise ValueError("Invalid channel format. Must be either CHW or HWC")

    if not squeeze:  # Include batch size dimension for input tensor ([B, ...])
        ret_image = np.expand_dims(ret_image, axis=0)

    return ret_image


def parse_augmentation(config):
    aug_name = config['name']
    aug_params = config.get('params', {})

    if aug_name == 'OneOf':
        # Parse the augmentations inside OneOf
        augmentations = [parse_augmentation(aug) for aug in config['augmentations']]
        return A.OneOf(augmentations, **aug_params)
    
    elif aug_name == 'resize':
        augmentations = build_resize_pipeline_albu(config['imgsz'])
        return A.Compose(augmentations)

    else:
        aug_class = getattr(A, aug_name)
        return aug_class(**aug_params)
    

class AlbumentationsWrapper:
    def __init__(self, pipeline):
        self.transform = A.Compose(pipeline)

    def __call__(self, img, *args, **kwargs):
        input_img = np.asarray(img)
        return self.transform(image=input_img)['image']


def build_augmentation_pipeline(aug_config, to_tensor=True, wrap=False):
    pipeline = [parse_augmentation(config) for config in aug_config]
    if to_tensor:
        pipeline.append(ToTensorV2(transpose_mask=True))

    if not wrap:
        return pipeline
    else:
        return AlbumentationsWrapper(pipeline)
