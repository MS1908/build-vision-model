import timm
import torch
from torchvision.models.feature_extraction import create_feature_extractor


def freeze_params(arch, model, n_block_to_train=-1):
    if n_block_to_train == -1:
        return

    for param in model.parameters():
        param.requires_grad = False

    if ('convnext' in arch) or ('inception_next' in arch):
        for param in model.head.parameters():
            param.requires_grad = True
        if n_block_to_train > 0:
            for param in model.stages[-n_block_to_train:].parameters():
                param.requires_grad = True

    elif 'mobilevit' in arch:
        for param in model.head.parameters():
            param.requires_grad = True
        for param in model.final_conv.parameters():
            param.requires_grad = True
        if n_block_to_train > 0:
            for param in model.stages[-n_block_to_train:].parameters():
                param.requires_grad = True

    elif 'vit' in arch:
        for param in model.head.parameters():
            param.requires_grad = True
        for param in model.fc_norm.parameters():
            param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True
        if n_block_to_train > 0:
            for param in model.blocks[-n_block_to_train:].parameters():
                param.requires_grad = True

    elif ('resnet' in arch) or ('resnext' in arch):
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.global_pool.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif 'efficientnet' in arch:
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.global_pool.parameters():
            param.requires_grad = True
        for param in model.conv_head.parameters():
            param.requires_grad = True
        for param in model.bn2.parameters():
            param.requires_grad = True
        if n_block_to_train > 0:
            for param in model.blocks[-n_block_to_train:].parameters():
                param.requires_grad = True

    else:
        for param in model.parameters():
            param.requires_grad = True


def unfreeze_param(model):
    for param in model.parameters():
        param.requires_grad = True


def build_timm_model(config, n_classes=None, device=None, features_only=False, phase='train'):
    """Build a model for image classification by using pytorch image model (timm) library

    Args:
        config (dictionary): Configuration to build model
        config = {
            'arch': Model architecture,
            'n_classes' (optional): The number of output classes,
            'pretrained' (optional): Use pretrained weight provided in timm library for corresponding model or not.
            'ckpt_path' (optional): Pretrained weight path. Defaults to None.
        }
        
        n_classes (int, optional): The number of output classes. If both this argument and n_classes are specified in config,
        this argument will be given preference.
        
        device: Device to load model on. Defaults to None.
        
        phase (str): The mode of model. Either in 'train' mode (for training) or 'val' mode (for evaluation). 
        Defaults to 'train'.

    Raises:
        ValueError: When phase is not either 'train' or 'val'

    Returns:
        A PyTorch model built by timm library in accordance to the config and specified arguments.
    """
    if n_classes is None:
        assert 'n_classes' in config, "Number of classes must be either specify in model config or in argument 'n_classes'"
        n_classes = config['n_classes']
    
    checkpoint_path = config.get('ckpt_path', None)
    if checkpoint_path:  # If checkpoint path is provided, then use the provided weight.
        model = timm.create_model(config['arch'],
                                  checkpoint_path=checkpoint_path,
                                  num_classes=n_classes,
                                  drop_rate=config.get('dropout', 0.),
                                  exportable=True,
                                  features_only=features_only)
    
    else:
        try:
            if phase == 'train':
                model = timm.create_model(config['arch'],
                                          pretrained=config.get('pretrained', False),
                                          num_classes=n_classes,
                                          drop_rate=config.get('dropout', 0.),
                                          exportable=True,
                                          features_only=features_only)
            else:
                model = timm.create_model(config['arch'],
                                          pretrained=False,  # Don't load timm pretrain weight when use model in eval mode
                                          num_classes=n_classes,
                                          drop_rate=config.get('dropout', 0.),
                                          exportable=True,
                                          features_only=features_only)
        except RuntimeError:  # Timm doesn't provide pretrain weight for this model
            model = timm.create_model(config['arch'],
                                      pretrained=False,
                                      num_classes=n_classes,
                                      drop_rate=config.get('dropout', 0.),
                                      exportable=True,
                                      features_only=features_only)

    freeze_params(
        arch=config['arch'], model=model,
        n_block_to_train=config.get('n_block_to_train', -1))

    if device is not None:
        model.to(device)

    if phase == 'train':
        model.train()
    elif phase == 'val':
        model.eval()
    else:
        raise ValueError(f"Invalid argument for 'phase'. Expect ['train', 'val'], found {phase}")

    return model
