import argparse
import copy
import math
import os
import random
import json
import torch
import warnings
import yaml
import numpy as np
import pandas as pd
import pytorch_warmup as warmup
from timm.utils import ModelEmaV2
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchmetrics import F1Score

from vision_classification import build_timm_model, model_summary, create_data_loader

warnings.filterwarnings("ignore")


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
def check_model_metric_is_best(best_metric, cur_metric):
    metric_names = ['f1', 'train_f1', 'acc', 'train_acc']
    for metric in metric_names:
        best = best_metric[metric]
        cur = cur_metric[metric]
        if best < cur:
            print(f"{metric} improved: {best:.4f} ===> {cur:.4f}")
            return True
        if not math.isclose(cur, best, abs_tol=1e-6):
            return False
    return False


def train(
    train_config,
    model_config,
    dataset_config,
    image_config,
    device_config
):
    seed = train_config.get('random_seed', None)
    if seed:
        set_all_seeds(seed)

    device_id = device_config.get('device_id', -1)
    use_gpu = torch.cuda.is_available() and device_id != -1
    device = torch.device(f"cuda:{device_id}" if use_gpu else "cpu")
    num_workers = device_config.get('num_worker', 1)
    
    num_epochs = train_config.get('epochs', 1)
    patience = train_config.get('patience', None)

    train_loader, n_classes = create_data_loader(
        dataset_config=dataset_config,
        image_config=image_config,
        num_worker=num_workers,
        mode='train',
        return_classes=False,
        random_seed=seed
    )
    val_loader, _ = create_data_loader(
        dataset_config=dataset_config,
        image_config=image_config,
        num_worker=num_workers,
        mode='val',
        return_classes=False,
        random_seed=seed
    )

    model = build_timm_model(config=model_config, n_classes=n_classes, device=device, phase='train')
    model_summary(model)

    smoothing = train_config.get('loss_smoothing', 0.)
    weight = train_config.get('loss_weight', None)
    if weight:
        weight = torch.Tensor(weight).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.get('lr', 1e-3))
    
    total_iters = num_epochs * len(train_loader)
    warmup_iters = total_iters // 6
    train_iters = total_iters - warmup_iters
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_iters)
    
    use_ema = train_config.get('ema', False)

    date_str = datetime.now().strftime('%Y%m%d-%H%M')
    ckpt_path = train_config.get('ckpt_path', None)
    if ckpt_path is not None:
        ckpt_path = os.path.join(ckpt_path, date_str)
        os.makedirs(ckpt_path, exist_ok=True)
    log_path = train_config.get('log_path', None)
    if log_path is not None:
        log_path = os.path.join(log_path, date_str)

    model_config_to_save = {
        'arch': model_config['arch'],
        'n_classes': n_classes,
        'imgsz': image_config.get('imgsz', (224, 224)),
        'mean': image_config.get('mean', (0.485, 0.456, 0.406)),
        'std': image_config.get('std', (0.229, 0.224, 0.225))
    }
    if ckpt_path:
        json.dump(model_config_to_save, open(os.path.join(ckpt_path, 'model_config.json'), 'w'))
    else:
        json.dump(model_config_to_save, open('model_config.json', 'w'))

    if use_ema:
        df = pd.DataFrame(columns=['epoch', 'val_f1', 'val_ema_f1', 'val_acc', 'val_ema_acc', 
                                   'train_f1', 'train_ema_f1', 'train_acc', 'train_ema_acc'])
    else:
        df = pd.DataFrame(columns=['epoch', 'val_f1', 'val_acc', 'train_f1', 'train_acc'])
    
    best_metric = {
        'f1': float('-inf'),
        'train_f1': float('-inf'),
        'acc': float('-inf'),
        'train_acc': float('-inf')
    }
    
    if log_path:
        logger = SummaryWriter(log_path)
        logger.add_scalar('Accuracy/val', 0., 0)
    else:
        logger = None
    
    if use_ema:
        model_ema = ModelEmaV2(model=model, device=device, decay=0.998)
    else:
        model_ema = None
    
    scaler = torch.cuda.amp.GradScaler()
    
    cnt_not_improve = 0

    if n_classes > 2:
        f1_score = F1Score(task='multiclass', num_classes=n_classes)
    else:
        f1_score = F1Score(task='binary')
    f1_score = f1_score.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch + 1}/{num_epochs}")
        total = 0
        correct = 0
        ema_correct = 0
        total_f1 = 0
        total_ema_f1 = 0
        for i, (images, targets) in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
                if model_ema:
                    outputs_ema = model_ema.module(images)
                    ema_loss = criterion(outputs_ema, targets)
            
            # loss.backward()
            # optimizer.step()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs.data, 1)
            
            if model_ema:
                model_ema.update(model)
                _, predicted_ema = torch.max(outputs_ema.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            cur_f1 = f1_score(predicted, targets)
            total_f1 += cur_f1.item()
            if model_ema:
                ema_correct += (predicted_ema == targets).sum().item()
                cur_ema_f1 = f1_score(predicted_ema, targets)
                total_ema_f1 += cur_ema_f1.item()
            
            lr = optimizer.param_groups[0]['lr']

            if use_ema:
                pbar.set_postfix(loss=loss.item(), ema_loss=ema_loss.item(), 
                                 train_acc=correct / total, ema_train_acc=ema_correct / total, 
                                 f1=total_f1 / (i + 1), ema_f1=total_ema_f1 / (i + 1),
                                 lr=lr)
            else:
                pbar.set_postfix(loss=loss.item(), train_acc=correct / total, f1=total_f1 / (i + 1), lr=lr)
            
            if logger:
                logger.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
                logger.add_scalar('Accuracy/train', correct / total, epoch * len(train_loader) + i)
                logger.add_scalar('F1 score/train', total_f1 / (i + 1), epoch * len(train_loader) + i)
                logger.add_scalar('Learning rate', lr, epoch * len(train_loader) + i)
                
                if use_ema:
                    logger.add_scalar('EMA loss/train', ema_loss.item(), epoch * len(train_loader) + i)
                    logger.add_scalar('EMA accuracy/train', ema_correct / total, epoch * len(train_loader) + i)
                    logger.add_scalar('EMA F1 score/train', total_ema_f1 / (i + 1), epoch * len(train_loader) + i)
                
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_iters:
                    lr_scheduler.step()

        train_acc = correct / total
        train_f1 = total_f1 / len(train_loader)
        if use_ema:
            train_ema_acc = ema_correct / total
            train_ema_f1 = total_ema_f1 / len(train_loader)
        
        model.eval()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val epoch {epoch + 1}/{num_epochs}")
        total = 0
        correct = 0
        ema_correct = 0
        total_f1 = 0
        total_ema_f1 = 0
        val_loss = 0
        with torch.no_grad():
            for i, (images, targets) in pbar:
                images = images.to(device)
                targets = targets.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    if model_ema:
                        outputs_ema = model_ema.module(images)
                
                scores, predicted = torch.max(outputs.data, 1)
                if use_ema:
                    scores_ema, predicted_ema = torch.max(outputs_ema.data, 1)
                
                cur_f1 = f1_score(predicted, targets)
                if use_ema:
                    cur_ema_f1 = f1_score(predicted_ema, targets)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                total_f1 += cur_f1.item()
                if use_ema:
                    ema_correct += (predicted_ema == targets).sum().item()
                    total_ema_f1 += cur_ema_f1.item()
                
                if use_ema:
                    pbar.set_postfix(val_acc=correct / total, ema_val_acc=ema_correct / total, 
                                     val_f1=total_f1 / (i + 1), ema_val_f1=total_ema_f1 / (i + 1))
                else:
                    pbar.set_postfix(val_acc=correct / total, val_f1=total_f1 / (i + 1))

            if logger:
                logger.add_scalar('Loss/val', val_loss / len(val_loader), epoch + 1)
                logger.add_scalar('Accuracy/val', correct / total, epoch + 1)
                logger.add_scalar('F1 score/val', total_f1 / (i + 1), epoch + 1)
                if use_ema:
                    logger.add_scalar('EMA accuracy/val', ema_correct / total, epoch + 1)
                    logger.add_scalar('EMA F1 score/val', total_ema_f1 / (i + 1), epoch + 1)
        
        val_loss /= len(val_loader)
        acc = correct / total
        f1 = total_f1 / len(val_loader)
        if use_ema:
            ema_acc = ema_correct / total
            ema_f1 = total_ema_f1 / len(val_loader)
        
        if use_ema:
            df.loc[len(df)] = [epoch, f1, ema_f1, acc, ema_acc, train_f1, train_ema_f1, train_acc, train_ema_acc]
        else:
            df.loc[len(df)] = [epoch, f1, acc, train_f1, train_acc]
        
        if ckpt_path:
            df.to_csv(os.path.join(ckpt_path, 'train_logs.csv'), lineterminator='\n', index=False)
        else:
            df.to_csv('train_logs.csv', lineterminator='\n', index=False)
            
        cur_metric = {
            'f1': f1,
            'train_f1': train_f1,
            'acc': acc,
            'train_acc': train_acc
        }
        
        cnt_not_improve += 1
        if check_model_metric_is_best(best_metric, cur_metric):
            best_metric = cur_metric.copy()
            
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}.pth")
            else:
                model_path = f"epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            
            cnt_not_improve = 0
        
        if use_ema:
            cur_ema_metric = {
                'f1': ema_f1,
                'train_f1': train_ema_f1,
                'acc': ema_acc,
                'train_acc': train_ema_acc
            }
            
            if check_model_metric_is_best(best_metric, cur_ema_metric):
                best_metric = cur_ema_metric.copy()
                
                if ckpt_path:
                    model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}.pth")
                else:
                    model_path = f"epoch_{epoch + 1}.pth"
                torch.save(model_ema.module.state_dict(), model_path)
                
                cnt_not_improve = 0
            
        if patience and cnt_not_improve >= patience:
            print(f"Early stopping after {patience} of not improving!")
            break
            
    if ckpt_path:
        last_model_path = os.path.join(ckpt_path, "last.pth")
    else:
        last_model_path = "last.pth"
    torch.save(model.state_dict(), last_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train_config = config['train_config']
    model_config = config['model']
    dataset_config = config['dataset']
    image_config = config['image_config']
    device_config = config['device_config']
        
    train(
        train_config=train_config,
        model_config=model_config,
        dataset_config=dataset_config,
        image_config=image_config,
        device_config=device_config
    )
