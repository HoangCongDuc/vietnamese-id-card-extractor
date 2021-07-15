from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json
import os
import imgaug.augmenters as iaa
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from vietocr.tool.translate import build_model, translate
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss


class FrontsideTextDataset(IterableDataset):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

        # Create template background image
        template = Image.open('data/template2.jpg')
        template = template.resize((800, 500))
        self.patches = []
        self.patches.append(template.crop((375, 120, 700, 175)))
        self.patches.append(template.crop((325, 170, 750, 220)))
        self.patches.append(template.crop((290, 215, 750, 265)))
        self.patches.append(template.crop((370, 260, 750, 310)))
        self.patches.append(template.crop((410, 305, 750, 355)))
        self.patches.append(template.crop((250, 350, 750, 400)))
        self.patches.append(template.crop((500, 395, 750, 445)))
        self.patches.append(template.crop((250, 435, 750, 485)))

        # Declare fonts
        self.font_text = [ImageFont.truetype('pala.ttf', size=s) for s in [24, 25, 26]]
        self.font_name = [ImageFont.truetype('pala.ttf', size=s) for s in [34, 35, 36]]
        self.font_number = [ImageFont.truetype('arial.ttf', size=s) for s in [31, 32, 33]]

        self.text_color = (50, 70, 70)

        # Create lists of names and addresses
        self.first_names, self.last_names = self.load_names()
        self.addresses = self.load_addresses()

        # Declare augmentor
        self.augmentor = self.get_augmentor()

    def load_names(self):
        names = json.load(open('uit_member.json'))
        first_names = []
        last_names = []
        charset = set(iter(self.vocab.chars))
        for name in names:
            first_name = name['first_name']
            if set(iter(first_name)).issubset(charset):
                first_names.append(first_name)
            last_name = name['last_name']
            if set(iter(last_name)).issubset(charset):
                last_names.append(last_name)
        return first_names, last_names

    def load_addresses(self):
        addresses = json.load(open('dvhcvn.json'))
        addresses = addresses['data']
        for level1 in addresses:
            if level1['type'] == "Thành phố Trung ương":
                level1['name'] = 'TP' + level1['name'][9:]
            else:
                level1['name'] = level1['name'][5:]
            for level2 in level1['level2s']:
                typelength = len(level2['type'])
                level2['name'] = level2['name'][typelength+1:]
                level3s = level2.get('level3s')
                if isinstance(level3s, list):
                    for level3 in level3s:
                        typelength = len(level3['type'])
                        level3['name'] = level3['name'][typelength+1:]
        addresses[0]['name'] = 'Hà Nội'
        return addresses

    def get_augmentor(self):
        contrast = iaa.Sequential(children=[
            iaa.LinearContrast((0.5, 2)),
            iaa.GammaContrast((0.5, 2))
            ],
            random_order=True
        )
        hue_saturate = iaa.AddToHueAndSaturation(value_hue=(-18, 18), value_saturation=(-30, 30))
        temperature = iaa.ChangeColorTemperature((5000, 8000))
        color = iaa.Sequential([temperature, hue_saturate, contrast])
        blur = iaa.OneOf([
            iaa.GaussianBlur((0, 2)),
            iaa.MotionBlur(3, angle=90)
        ])
        blur = iaa.Sometimes(0.2, blur)
        noise = iaa.OneOf([
            iaa.AdditiveGaussianNoise(),
            iaa.AdditiveLaplaceNoise(),
            iaa.AdditivePoissonNoise()
        ])
        geometry = iaa.Sequential([
            iaa.PerspectiveTransform((0, 0.05)),
            iaa.Affine(translate_percent=(-0.05, 0.05), mode='edge')
        ])
        augmentor = iaa.Sequential([color, blur, noise, geometry])
        return augmentor

    def add_text(self, background, text, font):
        w, h = background.size
        x1, y1, x2, y2 = font.getbbox(text)
        
        x = random.randrange(-x1, w - x2)
        y = random.randrange(-y1, h - y2)

        img = background.copy()
        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, font=font, fill=self.text_color)

        return img

    def gen_num(self, background):
        num = [str(random.randrange(10)) for _ in range(10)]
        num_str = ''.join(num)

        return self.add_text(background, num_str, random.choice(self.font_number)), num_str

    def gen_name(self, background):
        first_name = random.choice(self.first_names).upper()
        last_name = random.choice(self.last_names).upper()
        full_name = ' '.join([last_name, first_name])
        w = background.size[0]
        font_name = random.choice(self.font_name)
        
        l = font_name.getsize(full_name)[0]
        p = random.uniform(0, 1)
        if p < 0.9 and l < w:
            return self.add_text(background, full_name, font_name), full_name
        
        l = font_name.getsize(last_name)[0]
        p = random.uniform(0, 1)
        if p < 0.9 and l < w:
            return self.add_text(background, last_name, font_name), last_name
        
        return self.add_text(background, first_name, font_name), first_name

    def gen_DoB(self, background):
        day = random.randrange(31) + 1
        month = random.randrange(12) + 1
        year = random.randrange(1920, 2005)
        
        day = '{:02d}'.format(day)
        month = '{:02d}'.format(month)
        year = str(year)
        
        DoB_text = '-'.join([day, month, year])
        return self.add_text(background, DoB_text, random.choice(self.font_text)), DoB_text

    def gen_address(self, background):
        level1 = random.choice(self.addresses)
        level2 = random.choice(level1['level2s'])
        try:
            level3 = random.choice(level2['level3s'])
        except Exception:
            level3 = ''
        
        level1 = level1['name']
        level2 = level2['name']
        if isinstance(level3, dict):
            level3 = level3['name']
        
        w, h = background.size
        font_text = random.choice(self.font_text)
        
        address = ', '.join([level3, level2, level1])
        l = font_text.getsize(address)[0]
        p = random.uniform(0, 1)
        if p < 0.8 and l < w:
            return self.add_text(background, address, font_text), address
        
        address = ', '.join([level2, level1])
        l = font_text.getsize(address)[0]
        p = random.uniform(0, 1)
        if p < 0.8 and l < w:
            return self.add_text(background, address, font_text), address
        
        l = font_text.getsize(level3)[0]
        if l < w:
            return self.add_text(background, level3, font_text), level3
        return background, ''

    def gen_empty(self, background):
        noise = random.random()
        if noise < 0.1:
            return background, ''
        
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        text = ' '.join([last_name, first_name])
        upper = random.random()
        if upper < 0.5:
            text = text.upper()

        font_text = random.choice(self.font_name)
        x = random.randint(0, int(background.size[0] / 2))
        y = random.randint(1, 6)
        place_top = random.random()
        if place_top < 0.5:
            y = y - font_text.getsize(text)[1]
        else:
            y = background.size[1] - y

        img = background.copy()
        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, font=font_text, fill=self.text_color)

        return img, ''
        

    def __iter__(self):
        return self

    def gen_data(self):
        p = random.uniform(0, 1)
        if p < 0.01:
            background = random.choice(self.patches)
            return self.gen_empty(background)
        patch_type = random.randint(0, 3)
        if patch_type == 0:
            background = self.patches[0]
            return self.gen_num(background)
        if patch_type == 1:
            background = random.choice(self.patches[1:3])
            return self.gen_name(background)
        if patch_type == 2:
            background = self.patches[3]
            return self.gen_DoB(background)
        background = random.choice(self.patches[4:])
        return self.gen_address(background)
    
    def __next__(self):
        img, word = self.gen_data()
        
        label = self.vocab.encode(word)
        
        new_height = 32
        w, h = img.size
        new_width = int(w * new_height / h)
        img = img.resize((new_width, new_height))
        img = self.augmentor.augment_image(np.asarray(img)) / 255
        
        return img, label


def collate_fn(batch):
    subbatches = defaultdict(lambda: {'imgs': [], 'labels': [], 'padding_mask': []})
    max_len = max(len(sample[1]) for sample in batch)
    for img, label in batch:
        w = img.shape[1]
        subbatch = subbatches[w]
        subbatch['imgs'].append(np.transpose(img, (2, 0, 1)))

        length = len(label)
        subbatch['labels'].append(np.pad(label, (0, max_len - length)))
        subbatch['padding_mask'].append(
            np.concatenate([np.zeros(length-1), np.ones(max_len-length+1)]))

    rs = []
    for subbatch in subbatches.values():
        imgs = torch.FloatTensor(subbatch['imgs'])
        tgt_input = torch.tensor(subbatch['labels']).T
        tgt_output = torch.roll(tgt_input.T, -1, 1)
        tgt_output[:, -1] = 0
        tgt_padding_mask = torch.BoolTensor(subbatch['padding_mask'])
        rs.append(dict(
            img=imgs,
            tgt_input=tgt_input,
            tgt_output=tgt_output,
            tgt_padding_mask=tgt_padding_mask
        ))

    return rs


class Trainer:
    def __init__(self, config):
        self.device = config['model']['device']
        self.num_iters = config['trainer']['iters']
        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.checkpoint_every = config['trainer']['checkpoint_every']
        self.checkpoint = config['trainer']['checkpoint']

        self.model, vocab = build_model(config['model'])

        self.dataset = FrontsideTextDataset(vocab)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=config['trainer']['num_workers'],
            prefetch_factor=int(self.batch_size / config['trainer']['num_workers']))

        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            total_steps=self.num_iters,
            max_lr=0.0003,
            pct_start=0.1)

        self.iter = 0
        self.train_losses = []
        self.criterion = LabelSmoothingLoss(len(vocab), padding_idx=vocab.pad, smoothing=0.1)
        
    def save_checkpoint(self):
        state = {
            'iter': self.iter,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses
        }
        torch.save(state, self.checkpoint)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.iter = checkpoint['iter']
        self.train_losses = checkpoint['train_losses']

    def save_weight(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weight(self, path):
        weight = torch.load(path, map_location=self.device)
        self.model.load_state_dict(weight)

    def train_one_batch(self, batch):
        self.model.train()

        outputs = []
        tgt_outputs = []
        for subbatch in batch:
            img = subbatch['img'].to(self.device)
            tgt_input = subbatch['tgt_input'].to(self.device)
            tgt_output = subbatch['tgt_output'].to(self.device)
            tgt_padding_mask = subbatch['tgt_padding_mask'].to(self.device)
            
            outputs.append(self.model(img, tgt_input, tgt_padding_mask))
            tgt_outputs.append(tgt_output)
        outputs = torch.cat(outputs, dim=0)
        tgt_outputs = torch.cat(tgt_outputs, dim=0)

        outputs = outputs.view(-1, outputs.size(2))
        tgt_outputs = tgt_outputs.view(-1)

        loss = self.criterion(outputs, tgt_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def train(self):
        if os.path.exists(self.checkpoint):
            self.load_checkpoint()
        
        total_loss = 0
        data_iter = iter(self.dataloader)
        while self.iter < self.num_iters:
            self.iter += 1
            batch = next(data_iter)
            loss = self.train_one_batch(batch)
            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = "iter: {:06d} - train loss: {:.3f} - lr: {:.2e}"
                info = info.format(self.iter, total_loss / self.print_every, self.optimizer.param_groups[0]['lr'])
                print(info)
                total_loss = 0

            if self.iter % self.checkpoint_every == 0:
                self.save_checkpoint()
                # self.save_weight('drive/MyDrive/OCRmodel/model_weights.pth')
        
        return self.model


class Predictor:
    def __init__(self, config, weight):
        self.device = config['device']
        self.max_seq_length = config['transformer']['max_seq_length']
        self.model, self.vocab = build_model(config)
        self.model.load_state_dict(torch.load(weight, map_location=self.device))

    def predict(self, img):
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img_tensor = torch.FloatTensor(img)
        img_tensor = img_tensor.to(self.device)

        s = translate(img_tensor, self.model, self.max_seq_length)[0]
        s = s[0].tolist()
        s = self.vocab.decode(s)
        return s