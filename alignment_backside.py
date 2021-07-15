import os
import random
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import Adam, lr_scheduler
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from alignment_frontside import get_boundary, get_corner, warp_img

class SegmentationDataset(IterableDataset):
    def __init__(self, background_folder, template_path, training=True):
        self.template = cv2.imread(template_path)
        self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)
        h, w = self.template.shape[:-1]
        self.template_kp = [(0, 0), (0, h-1), (w-1, h-1), (w-1, 0)]

        self.background = []
        for filename in os.listdir(background_folder):
            img = cv2.imread(os.path.join(background_folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.background.append(img)
        
        self.foreground_augmentor = self.create_augmentors()
        self.img_augmentor = iaa.AdditiveGaussianNoise()

        self.training = training

    def create_augmentors(self):
        temperature = iaa.ChangeColorTemperature((3000, 9000))
        contrast = iaa.LinearContrast()
        saturation = iaa.AddToSaturation()
        color = iaa.Sequential([temperature, contrast, saturation])

        perspective = iaa.PerspectiveTransform(scale=0.05, fit_output=True)
        rotation = iaa.Rotate((-20, 20), fit_output=True)
        geometry = iaa.Sequential([perspective, rotation])

        return iaa.Sequential([color, geometry])
        # return perspective

    def create_foreground(self, shape):
        img, kp = self.foreground_augmentor.augment(image=self.template, keypoints=self.template_kp)
        foreground = np.zeros(shape, dtype='uint8')
        h_img, w_img = img.shape[:-1]
        h_fg, w_fg = foreground.shape[:-1]

        scale = min((h_fg-1)/h_img, (w_fg-1)/w_img)
        relative_size = random.uniform(0.9, 1)
        factor = scale * relative_size

        foreground_img = cv2.resize(img, None, fx=factor, fy=factor)
        h_img, w_img = foreground_img.shape[:2]
        top = random.randint(0, h_fg - h_img)
        left = random.randint(0, w_fg - w_img)
        foreground[top:top+h_img, left:left+w_img] = foreground_img

        kp = np.array(kp)
        kp = kp * factor + np.array([left, top])
        kp = kp.astype('int32')

        mask = np.zeros(foreground.shape[:-1], dtype='float32')
        mask = cv2.fillConvexPoly(mask, kp, 1)
        mask = cv2.blur(mask, (5, 5))

        return foreground, mask

    def blend(self, background, foreground, mask):
        alpha = np.expand_dims(mask, -1)
        return (alpha * foreground + (1-alpha) * background).astype('uint8')

    def __iter__(self):
        return self

    def __next__(self):
        background = random.choice(self.background)
        foreground, mask = self.create_foreground(background.shape)
        img = self.blend(background, foreground, mask)
        img = self.img_augmentor.augment_image(img)
        if self.training:
            img = process_img(img)
            mask = cv2.resize(mask, (224, 224))
        return {'img': img, 'mask': mask}


def process_img(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255
    img = np.transpose(img, (2, 0, 1))
    return img


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCNSegmentation(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super().__init__()
        cnn = torchvision.models.resnet18(pretrained_backbone)
        backbone_output = {'layer4': 'coarse', 'layer3': 'fine1', 'layer2': 'fine2'}
        self.backbone = IntermediateLayerGetter(cnn, backbone_output)
        self.clf_coarse = FCNHead(512, 1)
        self.upscale1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.clf_fine1 = FCNHead(256, 1)
        self.upscale2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.clf_fine2 = FCNHead(128, 1)

        bilinear_weight = get_upsampling_weight(1, 1, 4)
        self.upscale1.weight.data.copy_(bilinear_weight)
        self.upscale2.weight.data.copy_(bilinear_weight)

    def forward(self, x):
        features = self.backbone(x)
        score_coarse = self.clf_coarse(features['coarse'])
        
        score_upscale1 = self.upscale1(score_coarse)
        score_fine1 = self.clf_fine1(features['fine1'])
        score_fine1 = score_fine1 + score_upscale1

        score_upscale2 = self.upscale2(score_fine1)
        score_fine2 = self.clf_fine2(features['fine2'])
        score_fine2 = score_fine2 + score_upscale2

        out = F.interpolate(score_fine2, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.squeeze(out, dim=1)
        return out

        
class Trainer:
    def __init__(self, config):
        self.device = config['device']
        self.max_iters = config['iters']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.print_every = config['print_every']
        self.checkpoint_every = config['checkpoint_every']
        self.checkpoint = config['checkpoint']

        self.model = FCNSegmentation(config['pretrained'])
        self.model = self.model.to(self.device)

        self.dataset = SegmentationDataset(config['background'], config['template'])
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=int(self.batch_size / self.num_workers)
        )

        self.optimizer = Adam(self.model.parameters())
        self.lr_scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            total_steps=self.max_iters,
            max_lr=1e-3,
            pct_start=0.1)

        self.iter = 0

    def save_weight(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weight(self, path):
        weight = torch.load(path, map_location=self.device)
        self.model.load_state_dict(weight)

    def save_checkpoint(self):
        state = {
            'iter': self.iter,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(state, self.checkpoint)

    def load_checkpoint(self):
        state = torch.load(self.checkpoint)
        self.iter = state['iter']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

    def train_one_batch(self, batch):
        self.model.train()

        inputs, targets = batch['img'], batch['mask']
        inputs = inputs.to(self.device, dtype=torch.float32)
        targets = targets.to(self.device, dtype=torch.float32)

        outputs = self.model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean')
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def train(self):
        if os.path.exists(self.checkpoint):
            self.load_checkpoint()
        
        total_loss = 0
        data_iter = iter(self.dataloader)
        while self.iter < self.max_iters:
            self.iter += 1
            batch = next(data_iter)
            loss = self.train_one_batch(batch)
            total_loss += loss

            if self.iter % self.print_every == 0:
                info = "iter: {:06d} - train loss: {:.3f} - lr: {:.2e}"
                info = info.format(self.iter, total_loss / self.print_every, self.optimizer.param_groups[0]['lr'])
                print(info)
                total_loss = 0

            if self.iter % self.checkpoint_every == 0:
                self.save_checkpoint()
                # self.save_weight('drive/MyDrive/OCRmodel/alignment_model_weight.pth')
        
        return self.model


class BacksideAligner:
    def __init__(self, model_device='cpu'):
        self.device = model_device
        self.segment_model = FCNSegmentation().to(self.device)

    def load_weight(self, path):
        weight = torch.load(path, self.device)
        self.segment_model.load_state_dict(weight)

    def segment(self, img):
        w, h = img.shape[1], img.shape[0]
        img = process_img(img)
        img_tensor = torch.FloatTensor(img, device=self.device)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        self.segment_model.eval()
        with torch.no_grad():
            out = self.segment_model(img_tensor).squeeze()
            out = torch.sigmoid(out)
        mask = out.numpy()
        mask = cv2.resize(mask, (w, h))
        return mask
    
    def refine(self, img, mask):
        grabcut_mask = np.where(mask > 0.5, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        grabcut_mask[mask > 0.99] = cv2.GC_FGD
        grabcut_mask[:3, :] = cv2.GC_BGD
        grabcut_mask[-3:, :] = cv2.GC_BGD
        grabcut_mask[:, :3] = cv2.GC_BGD
        grabcut_mask[:, -3:] = cv2.GC_BGD

        fgdModel = np.zeros((1, 65), dtype='float')
        bgdModel = np.zeros((1, 65), dtype='float')
        grabcut_mask, bgdModel, fgdModel = cv2.grabCut(
            img, grabcut_mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
        
        output_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0)
        output_mask = output_mask.astype('uint8')
        return output_mask

    def warp(self, img, debug=False):
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        mask = self.segment(img)
        mask_refine = self.refine(img, mask)
        boundary_lines = get_boundary(mask_refine)
        corners = get_corner(*boundary_lines)
        img_aligned = warp_img(img, corners, (800, 500))
        if debug:
            return mask, mask_refine, boundary_lines, corners, img_aligned
        return img_aligned