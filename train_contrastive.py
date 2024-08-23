import argparse
import json
import os
import random
import torch
import warnings
import yaml
import numpy as np
import pandas as pd
import pytorch_warmup as warmup
from timm.utils import ModelEmaV2
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from deep_learning_utils import build_timm_model, ContrastiveLoss
from data_utils import create_data_loader
from misc_utils import model_summary

warnings.filterwarnings("ignore")


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
