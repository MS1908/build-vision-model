from .classification import binary_classify, binary_classify_onnx, classify, classify_onnx
from .dataset import ImageDataset, CustomSampleImageDataset, create_data_loader
from .image_processing_and_augmentation import albu_img_prepro, albu_img_aug_pipeline
from .timm_models import build_timm_model

from prettytable import PrettyTable


def model_summary(model):
    print()
    print('model_summary')
    
    table = PrettyTable(["Modules", "Parameters", "Trainable parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if not parameter.requires_grad:
            trainable_params = 0
        else:
            trainable_params = params
        
        table.add_row([name, params, trainable_params])
        total_params += params
        total_trainable_params += trainable_params
        
    print(table)
    print(f"Total params: {total_params}")
    print(f"Total trainable params: {total_trainable_params}")
    print()
