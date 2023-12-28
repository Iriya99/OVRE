import os
import cv2
import torch
import random
import shutil
import pickle
import numpy as np
import ujson as json
from PIL import Image
from torchvision import transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('     '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")
    
def load_json(filename):
    with open(filename, "r") as f: 
        return json.load(f)

def load_pkl(filename):
    with open(filename, "rb") as f: 
        return pickle.load(f)

def load_frames_from_video(src, dst):
    cmd = 'ffmpeg -i '+ src +' -q 0 -r 30 '+ dst + src.split('/')[-1].split('.')[0] +'/%05d.jpg ' + '-loglevel quiet'
    os.system(cmd)

def load_frames(path, sample_type, num_frames):
    intervals = np.linspace(start=1, stop=len(os.listdir(path)), num=num_frames+1).astype(int)
    ranges = []
    frames = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample_type == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    elif sample_type == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    for idx in frame_idxs:
        idx = str(idx).zfill(5)
        frame = cv2.imread(os.path.join(path, idx + '.jpg'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = torch.from_numpy(frame)
        # frame = frame.permute(2, 0, 1)
        from PIL import Image
        frame = Image.fromarray(frame)
        frames.append(frame)
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    # frames = torch.stack(frames).float() / 255
    return frames

def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)

def deletedir(p):
    if os.path.exists(p):
        shutil.rmtree(p)

def generate_multihot_label(categories, num_classes):
    multi_hot = (torch.zeros(num_classes)).to(torch.int64)
    multi_hot[categories] = 1
    return multi_hot

import torchvision
from transforms import *

def train_augmentation(input_size, flip=True):
    if flip:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            GroupRandomHorizontalFlip(is_flow=False)])
    else:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            # GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip_sth()])

def get_augmentation(mode, args):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if args.input_size == 224 else args.input_size

    normalize = GroupNormalize(input_mean, input_std)
    groupscale = GroupScale(int(scale_size))

    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize])

    if mode != "test":
        train_aug = train_augmentation(
            args.input_size,
            flip=True)

        unique = torchvision.transforms.Compose([
            groupscale,
            train_aug,
            GroupRandomGrayscale(p=0.2),
        ])
            
        return torchvision.transforms.Compose([unique, common])

    else:
        unique = torchvision.transforms.Compose([
            groupscale,
            GroupCenterCrop(args.input_size)])
        return torchvision.transforms.Compose([unique, common])

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    # output = output.detach()
    # output.requires_grad_(True)
    return output

import numpy as np
from sklearn.metrics import average_precision_score

def calculate_mAP(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    values = []
    for i in range(len(y_pred)):
        values.append(average_precision_score(y_true[i], y_pred[i], average='macro'))
    return np.mean(values) * 100