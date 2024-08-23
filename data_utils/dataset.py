import cv2
import os
import random
import torch
import albumentations as A
import numpy as np
import pandas as pd
from glob import glob
from torch.utils import data

from .image_processing_and_augmentation import build_augmentation_pipeline

IMAGE_EXTS = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']


def onehot(n_classes, target):
    vec = torch.zeros(n_classes, dtype=torch.float32)
    vec[target] = 1.
    return vec


def is_image_folder(image_root):
    """Check if the directory image_root is an image folder. 
    Image folder is a directory with the following format:
    
    image_root
        |----class_1
            |----img_1
            |----img_2
            ...
        |----class_2
            |----img_1
            |----img_2
            ...
        ...
        |----class_n
            |----img_1
            |----img_2
            ...
    
    Args:
        image_root (str): The directory to check
    """
    image_paths = []
    for image_ext in IMAGE_EXTS:
        image_paths.extend(glob(image_root + f'/**/*.{image_ext}', recursive=True))
    class_paths = []
    for path in image_paths:
        class_paths.append(os.path.dirname(path))
    root = []
    for path in class_paths:
        root.append(os.path.dirname(path))
    root = list(set(root))
    
    if len(root) == 1 and os.path.normpath(root[0]) == os.path.normpath(image_root):
        return True
    
    return False


class ImageDataset(data.Dataset):
    """Construct an images dataset for image classification. There are three types of dataset that is supported.
    
    1. Dataset with CSV annotation file: The annotation file has the following format:
    path                    label
    img_1 absolute path     0
    img_2 absolute path     2
    img_3 absolute path     3
    ...
    
    2. Image folder dataset: Check the definition of image folder dataset in the function is_image_folder()
    
    3. A folder of images.
    """

    def __init__(
        self, 
        image_root=None, 
        annotation_file=None, 
        transforms_pipeline=None, 
        categorical=False, 
        return_image=True,
        to_rgb=True
    ):
        """
        Args:
            image_root (str): The directory of images dataset
            
            annotation_file (str, optional): Path to CSV annotation file. Defaults to None.
            
            transforms_pipeline (List, optional): A list of Albumentations transform for augmentation. Defaults to None.
            
            categorical (bool, optional): Transform label to categorical label flag. If set to True then
            the label is transformed to categorical label. Defaults to False.
            
            return_image (bool, optional): Return image flag. If set to true, then the dataset with return
            image when get item, otherwise it return the path to image. Defaults to True.
            
            to_rgb (bool, optional): Convert image to RGB color space. If set to true, then the image will be convert to RGB color space.
            Otherwise the image will be in default BGR color space of OpenCV
        """
        super(ImageDataset, self).__init__()

        assert image_root is not None or annotation_file is not None, "Either csv annotation file or root of image dataset must be provided"
        
        if annotation_file is not None:
            df = pd.read_csv(annotation_file)
            self.literal_labels = sorted(df["label"].unique().tolist())
            self.n_classes = len(self.literal_labels)
            
            self.samples = []
            for i, row in df.iterrows():
                image_path = row["path"]
                label = self.literal_labels.index(row["label"])
                self.samples.append((image_path, label))
        
        elif is_image_folder(image_root):
            image_paths = []
            self.literal_labels = sorted(os.listdir(image_root))
            self.n_classes = len(self.literal_labels)
            for image_ext in IMAGE_EXTS:
                image_paths.extend(glob(image_root + f'/**/*.{image_ext}', recursive=True))
                
            self.samples = []
            for path in image_paths:
                label = os.path.dirname(path).split('/')[-1]
                self.samples.append((path, self.literal_labels.index(label)))

        else:
            self.literal_labels = None
            image_paths = []
            self.n_classes = -1
            for image_ext in IMAGE_EXTS:
                image_paths.extend(glob(image_root + f'/**/*.{image_ext}', recursive=True))
                
            self.samples = []
            for path in image_paths:
                self.samples.append((path, -1))

        self.transforms = None if transforms_pipeline is None else A.Compose(transforms_pipeline)
        self.categorical = categorical
        self.return_image = return_image
        self.to_rgb = to_rgb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.return_image:
            sample = cv2.imread(path)
            if self.to_rgb:
                sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            
            if self.transforms is not None:
                sample = self.transforms(image=sample)['image']
        else:
            sample = path

        if self.categorical and target != -1:
            target = onehot(self.n_classes, target)

        return sample, target

    def get_literal_labels(self):
        return self.literal_labels


class CustomSampleImageDataset(ImageDataset):

    def __init__(
        self, 
        image_root, 
        annotation_file, 
        transforms_pipeline=None, 
        class_sampling_ratio=None, 
        categorical=False,
        return_image=True,
        to_rgb=True
    ):
        """
        This dataset is used in the case a dataset is imbalanced, we need to oversample the class with low
        population. Inherits from ImageDataset.
        
        Args:
            image_root (str): The directory of images dataset
            
            annotation_file (str, optional): Path to CSV annotation file. Defaults to None.
            
            transforms_pipeline (List, optional): A list of Albumentations transform for augmentation. Defaults to None.
            
            class_sampling_ratio(List[int], optional): The oversampling rate. For example, [1, 3] means each sample with label
            0 is sampled once, each sample with label 1 is sampled thrice.
            
            categorical (bool, optional): Transform label to categorical label flag. If set to True then
            the label is transformed to categorical label. Defaults to False.
            
            return_image (bool, optional): Return image flag. If set to true, then the dataset with return
            image when get item, otherwise it return the path to image. Defaults to True.
            
            to_rgb (bool, optional): Convert image to RGB color space. If set to true, then the image will be convert to RGB color space.
            Otherwise the image will be in default BGR color space of OpenCV
        """
        super(CustomSampleImageDataset, self).__init__(image_root=image_root, 
                                                       annotation_file=annotation_file, 
                                                       transforms_pipeline=transforms_pipeline, 
                                                       categorical=categorical, 
                                                       return_image=return_image,
                                                       to_rgb=to_rgb)
        self.class_sampling_ratio = class_sampling_ratio
        
        if class_sampling_ratio is not None:
            assert self.literal_labels is not None, "Dataset need to be able to parse labels to custom sample classes"
            assert len(class_sampling_ratio) == len(self.literal_labels), "Class sampling ratio list must have equal size to number of classes."
            self.n_classes = len(self.literal_labels)
            self.n_sample_per_image = {}
            oversampling_samples = []
            for path, label in self.samples:
                oversampling_samples.extend([(path, label)] * int(class_sampling_ratio[label]))
            self.samples = oversampling_samples[:]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(dataset_config, aug_config, num_worker=8, mode='train',
                       return_classes=False, random_seed=None):
    image_root = dataset_config.get(f'{mode}_root', None)
    annotation_file = dataset_config.get(f'{mode}_label', None)

    to_rgb = aug_config.get('to_rgb', True)
    
    pipeline = build_augmentation_pipeline(aug_config['augmentations'], to_tensor=True, wrap=False)

    if mode == 'train':
        class_sampling_ratio = dataset_config.get('class_sample_ratio', None)
        dataset = CustomSampleImageDataset(image_root, annotation_file, transforms_pipeline=pipeline,
                                           class_sampling_ratio=class_sampling_ratio, 
                                           return_image=True, to_rgb=to_rgb)

    else:
        dataset = CustomSampleImageDataset(image_root, annotation_file, transforms_pipeline=pipeline,
                                           class_sampling_ratio=None, return_image=True, to_rgb=to_rgb)

    if random_seed:
        g = torch.Generator()
        g.manual_seed(random_seed)
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset_config.get('batch_size', 1),
            shuffle=mode == 'train',
            num_workers=num_worker,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset_config.get('batch_size', 1),
            shuffle=mode == 'train',
            num_workers=num_worker
        )
    if return_classes:
        return data_loader, dataset.n_classes, dataset.get_literal_labels()
    else:
        return data_loader, dataset.n_classes
