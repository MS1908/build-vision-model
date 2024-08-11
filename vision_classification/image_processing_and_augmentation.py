import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def resize_operation(imgsz=(224, 224)):
    if isinstance(imgsz, tuple) or isinstance(imgsz, list):
        assert len(imgsz) <= 2, "List/tuple of sizes must be of format (h, w) or (imgsz,)"
        
        if len(imgsz) == 2:
            output_h, output_w = imgsz
            resize_op = A.Resize(height=output_h, width=output_w)
        else:
            # Only longest size is provided
            resize_op = A.Compose([
                A.LongestMaxSize(max_size=imgsz[0]),
                A.PadIfNeeded(min_height=imgsz[0], min_width=imgsz[0],
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            ])
    else:
        # Only longest size is provided
        resize_op = A.Compose([
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz,
                          border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ])
    return resize_op


def albu_img_prepro(
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
    
    resize_op = resize_operation(imgsz)
    transforms = A.Compose([
        resize_op,
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], p=1.0)
    ret_image = transforms(image=ret_image)['image']

    if not squeeze:  # Include batch size dimension for input tensor ([B, ...])
        ret_image = ret_image.unsqueeze(0)
    if to_np:
        ret_image = ret_image.cpu().numpy()
    
    if channel_format == 'HWC':  # Swap channel CHW -> HWC
        ret_image = np.transpose(ret_image, (1, 2, 0))
    elif channel_format != 'CHW':
        raise ValueError("Invalid channel format. Must be either CHW or HWC")

    return ret_image


class AlbumentationsWrapper:

    def __init__(self, pipeline):
        self.transform = A.Compose(pipeline)

    def __call__(self, img, *args, **kwargs):
        input_img = np.asarray(img)
        return self.transform(image=input_img)['image']


def albu_img_aug_pipeline(
    aug_level='lo', 
    phase='train', 
    imgsz=(224, 224), 
    mean=(0.485, 0.456, 0.406), 
    std=(0.229, 0.224, 0.225), 
    to_tensor=True,
    wrap=False
):        
    resize_op = resize_operation(imgsz)
    if phase != 'train' or aug_level not in ['lo', 'med', 'hi']:
        pipeline = [
            resize_op,
            A.Normalize(mean=mean, std=std)
        ]
        
    elif aug_level == 'lo':  # Low level data augmentation
        print('AUGMENTATION LEVEL: LOW.')
        
        pipeline = [
            resize_op,
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1)
            ], p=0.2),
            A.HueSaturationValue(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.025,
                rotate_limit=10,
                shift_limit_x=0.025,
                shift_limit_y=0.025,
                p=0.3),
            A.ImageCompression(quality_lower=90),
            A.Normalize(mean=mean, std=std)
        ]
        
    elif aug_level == 'med':  # Medium level data augmentation (just blur, adjust brightness, JPG compression , rotation with small angle)
        print('AUGMENTATION LEVEL: MEDIUM.')
        
        pipeline = [
            resize_op,
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1)
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast()
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=70),
            A.ShiftScaleRotate(
                shift_limit=0.025,
                rotate_limit=10,
                shift_limit_x=0.025,
                shift_limit_y=0.025,
                p=0.3),
            A.Normalize(mean=mean, std=std)
        ]

    else:  # Strong data augmentation (noise, blur, optical distortion, adjust brightness, JPG compression with low quality, rotation with big angle)
        print('AUGMENTATION LEVEL: HIGH.')
        
        pipeline = [
            resize_op,
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.OpticalDistortion(),
                A.ElasticTransform()
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3)
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast()
            ], p=0.4),
            A.HueSaturationValue(p=0.4),
            A.ImageCompression(quality_lower=50),
            A.ShiftScaleRotate(
                shift_limit=0.025,
                rotate_limit=90,
                shift_limit_x=0.025,
                shift_limit_y=0.025,
                p=0.3),
            A.Normalize(mean=mean, std=std)
        ]
        
    if to_tensor:
        pipeline.append(ToTensorV2(transpose_mask=True))

    if not wrap:
        return pipeline

    return AlbumentationsWrapper(pipeline)
