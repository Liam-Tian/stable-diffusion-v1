import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torch import nn
#import torch

# target = Image.open('target.jpg')

# trans = transforms.Compose([transforms.ToTensor()])
# target = trans(target)
# print(target.shape)
# target = 255. * rearrange(target.cpu().detach().numpy(), 'c h w -> h w c')
# Image.fromarray(target.astype(np.uint8)).save('test.png')

target = torch.zeros(3, 512, 512)
target = 255. * rearrange(target.cpu().detach().numpy(), 'c h w -> h w c')
Image.fromarray(target.astype(np.uint8)).save('test.png')