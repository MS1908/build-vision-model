import argparse
import copy
import json
import math
import os
import random
import torch
import warnings
import yaml
import numpy as np
import pytorch_warmup as warmup
import torch.nn.functional as F
from timm.utils import ModelEmaV2
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from vision_classification import build_timm_model, model_summary, create_data_loader

warnings.filterwarnings("ignore")


def criterion_kd(outputs, teacher_outputs, targets, alpha, temperature):
    kd_loss = (nn.KLDivLoss()(F.log_softmax(outputs / temperature, dim=1), 
                              F.softmax(teacher_outputs / temperature, dim=1)) * alpha * temperature * temperature + 
               F.cross_entropy(outputs, targets) * (1 - alpha))
    return kd_loss


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(
    model,
    teacher_model,
    device,
    train_loader,
    val_loader,
    alpha,
    temperature,
    optimizer,
    scheduler,
    update_scheduler_by_iter,
    update_scheduler_with_metric,
    epochs,
    patience=None,
    ckpt_path=None,
    log_path=None
):
    best_acc = float('-inf')
    best_train_acc = float('-inf')
    
    if log_path:
        logger = SummaryWriter(log_path)
        logger.add_scalar('Accuracy/val', 0., 0)
    else:
        logger = None
    
    model_ema = ModelEmaV2(model=model, device=device, decay=0.998)
    
    cnt_not_improve = 0
    
    best_model = None
    teacher_model.to(device)
    teacher_model.eval()
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch + 1}/{epochs}")
        total = 0
        correct = 0
        correct_ema = 0
        for i, (images, targets) in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs_ema = model_ema.module(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            loss = criterion_kd(outputs, teacher_outputs, targets, alpha, temperature)
            
            loss.backward()
            optimizer.step()
            model_ema.update(model)

            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ema = torch.max(outputs_ema.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            correct_ema += (predicted_ema == targets).sum().item()
            
            lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix(loss=loss.item(), train_acc=correct / total, ema_train_acc=correct_ema / total, lr=lr)
            if logger:
                logger.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
                logger.add_scalar('Accuracy/train', correct / total, epoch * len(train_loader) + i)
                logger.add_scalar('EMA accuracy/train', correct_ema / total, epoch * len(train_loader) + i)
                logger.add_scalar('Learning rate', lr, epoch * len(train_loader) + i)
            
            if update_scheduler_by_iter:  # Update scheduler after each iteration
                scheduler.step()

        train_acc = correct / total
        train_acc_ema = correct_ema / total
        
        model.eval()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val epoch {epoch + 1}/{epochs}")
        total = 0
        correct = 0
        correct_ema = 0
        with torch.no_grad():
            for i, (images, targets) in pbar:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                outputs_ema = model_ema.module(images)
                
                scores, predicted = torch.max(outputs.data, 1)
                scores_ema, predicted_ema = torch.max(outputs_ema.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                correct_ema += (predicted_ema == targets).sum().item()
                pbar.set_postfix(val_acc=correct / total, val_acc_ema=correct_ema / total)

            if logger:
                logger.add_scalar('Accuracy/val', correct / total, epoch + 1)

        acc = correct / total
        acc_ema = correct_ema / total
        cnt_not_improve += 1
        if acc > best_acc:
            print(f"Val accuracy improved: {best_acc:.4f} ===> {acc:.4f}")
            best_acc = acc
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}_val_acc_{round(acc, 2)}.pth")
                torch.save(model.state_dict(), model_path)
            cnt_not_improve = 0
        
        if acc_ema > best_acc:
            print(f"EMA val accuracy improved: {best_acc:.4f} ===> {acc_ema:.4f}")
            best_acc = acc_ema
            best_train_acc = train_acc_ema
            best_model = copy.deepcopy(model_ema.module)
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}_ema_val_acc_{round(acc_ema, 2)}.pth")
                torch.save(model_ema.module.state_dict(), model_path)
            cnt_not_improve = 0
        
        elif math.isclose(acc, best_acc, abs_tol=1e-6) and best_train_acc < train_acc:
            print(f"Train accuracy improved: {best_train_acc:.4f} ===> {train_acc:.4f}")
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}_train_acc_{round(train_acc, 2)}.pth")
                torch.save(model.state_dict(), model_path)
            cnt_not_improve = 0
            
        elif math.isclose(acc_ema, best_acc, abs_tol=1e-6) and best_train_acc < train_acc_ema:
            print(f"EMA train accuracy improved: {best_train_acc:.4f} ===> {train_acc_ema:.4f}")
            best_train_acc = train_acc_ema
            best_model = copy.deepcopy(model)
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}_train_acc_{round(train_acc, 2)}.pth")
                torch.save(model.state_dict(), model_path)
            cnt_not_improve = 0
            
        if patience and cnt_not_improve >= patience:
            print(f"Early stopping after {patience} of not improving!")
            break
        
        if not update_scheduler_by_iter:  # Update scheduler after each epoch, probably ReduceLRByPlateau
            if update_scheduler_with_metric:
                scheduler.step(metrics=acc)  # On plateau scheduler
            else:
                scheduler.step()
            
    if ckpt_path:
        last_model_path = os.path.join(ckpt_path, f"last.pth")
        torch.save(model.state_dict(), last_model_path)
        
    return best_model, best_acc, best_train_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--log-path', type=str, default=None)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--n-worker', type=int, default=4)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train_config = config['train']
    model_config = config['model']
    teacher_model_config = config['teacher_model']
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    loss_config = config['loss']
    data_config = config['dataset']
    preprocess_config = config['preprocess']
    seed = train_config.get('seed', None)

    if seed:
        set_all_seeds(seed)

    use_gpu = torch.cuda.is_available() and args.device != -1
    device = torch.device(f"cuda:{args.device}" if use_gpu else "cpu")

    train_loader, n_classes = create_data_loader(
        dataset_config=data_config,
        image_config=preprocess_config,
        num_worker=args.n_worker,
        mode='train',
        return_classes=False,
        random_seed=args.seed
    )
    val_loader, _ = create_data_loader(
        dataset_config=data_config,
        image_config=preprocess_config,
        num_worker=args.n_worker,
        mode='val',
        return_classes=False,
        random_seed=args.seed
    )

    teacher_model = build_timm_model(config=teacher_model_config, n_classes=n_classes, device=device, phase='val')
    print('Teacher model')
    model_summary(teacher_model)
    
    model = build_timm_model(config=model_config, n_classes=n_classes, device=device, phase='train')
    print('Student model')
    model_summary(model)
    
    if loss_config is not None:
        smoothing = loss_config.get('smoothing', 0.)
        weight = loss_config.get('weight', None)
        if weight:
            weight = torch.Tensor(weight).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch_optimizer_factory(optimizer_config, model.parameters())
    scheduler = scheduler_factory(scheduler_config, optimizer)

    date_str = datetime.now().strftime('%Y%m%d-%H%M')
    if args.ckpt_path is not None:
        ckpt_path = os.path.join(args.ckpt_path, date_str)
        os.makedirs(ckpt_path, exist_ok=True)
    else:
        ckpt_path = None
    if args.log_path is not None:
        log_path = os.path.join(args.log_path, date_str)
    else:
        log_path = None

    config_to_save = {
        'arch': model_config['arch'],
        'n_classes': n_classes,
        'imgsz': preprocess_config['imgsz'],
        'mean': preprocess_config.get('mean', None),
        'std': preprocess_config.get('std', None)
    }
    json.dump(config_to_save, open(os.path.join(ckpt_path, 'model_config.json'), 'w'))
        
    best_model, best_acc, best_train_acc = train(
        model=model,
        teacher_model=teacher_model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        alpha=loss_config['alpha'],
        temperature=loss_config['temperature'],
        optimizer=optimizer,
        scheduler=scheduler,
        update_scheduler_by_iter=scheduler_config.get('update_by_iter', False),
        update_scheduler_with_metric=scheduler_config.get('use_metric', False),
        epochs=train_config['epochs'],
        patience=train_config.get('patience', None),
        ckpt_path=ckpt_path,
        log_path=log_path
    )
