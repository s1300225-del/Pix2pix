import os
import argparse
from pathlib import Path
import re
import random

import numpy as np
import tqdm
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="学習(train)か推論(test)か")
parser.add_argument("--n_epoch", type=int, default=200, help="学習エポック数")
parser.add_argument("--batch_size", type=int, default=32, help="バッチサイズ")
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--size", type=int, default=256, help="画像サイズ")
parser.add_argument("--lambda_l1", type=float, default=10.0, help="L1損失自体の重み")
parser.add_argument("--lambda_dice", type=float, default=1.5, help="Dice損失の重み")
parser.add_argument("--road_weight", type=float, default=5.0, help="道路ピクセルに対する重み倍率")
parser.add_argument("--root_dir", type=str, default="./dataset/train", help="データセットのパス")
parser.add_argument("--result_dir", type=str, default="./result")
parser.add_argument("--params_dir", type=str, default="./params")
parser.add_argument("--load_model", type=str, default="", help="ロードする重みパス")
parser.add_argument("--num_workers", type=int, default=2, help="データ読み込みの並列数")
opt = parser.parse_args()

# ----------------------------
# Device Setup
# ----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ----------------------------
# Dice Loss Definition
# ----------------------------
def dice_loss(pred, target, smooth=1.0):
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# ----------------------------
# Dataset
# ----------------------------
class PairedDataset(Dataset):
    def __init__(self, root_dir, size=256, mode='train', suffix_input='_height', suffix_target='_road'):
        self.root_dir = Path(root_dir)
        self.size = size
        self.mode = mode
        self.suffix_input = suffix_input
        self.suffix_target = suffix_target

        files_input = []
        files_target = []
        dir_height = self.root_dir / "height"
        dir_road = self.root_dir / "road"

        if dir_height.exists() and dir_road.exists():
            files_input = [p for p in dir_height.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            files_target = [p for p in dir_road.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        else:
            all_files = [p for p in self.root_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            for p in all_files:
                if self.suffix_input.lower() in p.name.lower(): files_input.append(p)
                if self.suffix_target.lower() in p.name.lower(): files_target.append(p)

        def prefix_of(path, suffix):
            pattern = rf"^(.*?_\d+){re.escape(suffix)}\.[^.]+$"
            m = re.match(pattern, path.name, flags=re.IGNORECASE)
            return m.group(1) if m else None

        dict_in = {prefix_of(p, self.suffix_input): p for p in files_input if prefix_of(p, self.suffix_input)}
        dict_tar = {prefix_of(p, self.suffix_target): p for p in files_target if prefix_of(p, self.suffix_target)}
        common = sorted(set(dict_in.keys()).intersection(set(dict_tar.keys())))
        self.pairs = [(dict_in[k], dict_tar[k], k) for k in common]

        self.cached_data = []
        for path_A, path_B, prefix in tqdm.tqdm(self.pairs, desc=f"Loading {mode} dataset"):
            img_A = Image.open(path_A).convert("L")
            img_B = Image.open(path_B).convert("L")
            img_A.load(); img_B.load()
            self.cached_data.append((img_A, img_B, prefix))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        img_A_pil, img_B_pil, prefix = self.cached_data[index]
        img_A_pil = TF.resize(img_A_pil, (self.size, self.size))
        img_B_pil = TF.resize(img_B_pil, (self.size, self.size))

        if self.mode == 'train':
            if random.random() > 0.5:
                img_A_pil = TF.hflip(img_A_pil); img_B_pil = TF.hflip(img_B_pil)
            if random.random() > 0.5:
                img_A_pil = TF.vflip(img_A_pil); img_B_pil = TF.vflip(img_B_pil)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img_A_pil = TF.rotate(img_A_pil, angle); img_B_pil = TF.rotate(img_B_pil, angle)

        arr_A = np.array(img_A_pil)
        sobelx = cv2.Sobel(arr_A, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(arr_A, cv2.CV_64F, 0, 1, ksize=3)
        slope = np.sqrt(sobelx**2 + sobely**2)
        slope = cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        combined_A = np.stack([arr_A, slope], axis=-1)
        tensor_A = TF.to_tensor(combined_A)
        tensor_B = TF.to_tensor(img_B_pil)
        tensor_A = TF.normalize(tensor_A, [0.5, 0.5], [0.5, 0.5])
        tensor_B = TF.normalize(tensor_B, [0.5], [0.5])

        return {"A": tensor_A, "B": tensor_B, "name": prefix}

# ----------------------------
# Models
# ----------------------------
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)

        self.dilated_bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4); d6 = self.down6(d5); d7 = self.down7(d6)
        db = self.dilated_bottleneck(d7)
        u1 = self.up1(db, d6); u2 = self.up2(u1, d5); u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3); u5 = self.up5(u4, d2); u6 = self.up6(u5, d1)
        return self.final(u6)

class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        total_channels = in_channels + 1
        
        def block(in_f, out_f, norm=True):
            # 【変更点】Spectral Normalization を適用して学習を安定化
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, 4, 2, 1))]
            if norm: layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(total_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            # 【変更点】最終層にも Spectral Normalization を適用
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, padding=1, bias=False))
        )
    def forward(self, img_A, img_B):
        return self.model(torch.cat((img_A, img_B), 1))

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

# ----------------------------
# Main Process
# ----------------------------
def main():
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.params_dir, exist_ok=True)

    generator = GeneratorUNet(in_channels=2, out_channels=1).to(device)
    discriminator = Discriminator(in_channels=2).to(device)

    if opt.load_model and os.path.isfile(opt.load_model):
        generator.load_state_dict(torch.load(opt.load_model, map_location=device))
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr-3e-4, betas=(0.5, 0.999))
    criterion_GAN = nn.BCEWithLogitsLoss()

    dataset = PairedDataset(opt.root_dir, size=opt.size, mode=opt.mode)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=(opt.mode=="train"), num_workers=opt.num_workers, drop_last=True)

    if opt.mode == "test":
        generator.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                real_A = batch["A"].to(device)
                fake_B = generator(real_A)
                real_A_display = real_A[:, 0:1, :, :] 
                for k in range(real_A.size(0)):
                    vutils.save_image(torch.cat((real_A_display[k], fake_B[k]), 2), os.path.join(opt.result_dir, f"{batch['name'][k]}_gen.png"), normalize=True)
        return

    for epoch in range(opt.n_epoch):
        loop = tqdm.tqdm(dataloader, leave=True)
        for i, batch in enumerate(loop):
            real_A, real_B = batch["A"].to(device), batch["B"].to(device)

            # --- Train Generator (2回更新) ---
            for _ in range(2):
                optimizer_G.zero_grad()
                fake_B = generator(real_A)
                pred_fake = discriminator(real_A, fake_B)
                loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                
                # Weighted L1 Loss
                diff = torch.abs(fake_B - real_B)
                road_mask = (real_B > 0).float()
                weight_map = 1.0 + road_mask * (opt.road_weight - 1.0)
                loss_pixel = (diff * weight_map).mean()

                # Dice Loss
                loss_dice_val = dice_loss(fake_B, real_B)

                # --- 【追加】物理制約 (標高・勾配) ---
                # real_Aは -1~1 に正規化されているため 0~1 に戻して計算
                elevation = (real_A[:, 0:1, :, :] + 1) / 2
                slope = (real_A[:, 1:2, :, :] + 1) / 2
                fake_B_01 = (fake_B + 1) / 2  # 生成画像も 0~1 (黒~白) に変換
                
                # 標高ペナルティ: 標高 0.7 以上の白い場所に道を作ると罰金
                loss_elevation = (torch.relu(elevation - 0.7) * fake_B_01).mean()
                # 勾配ペナルティ: 勾配 0.5 以上の急斜面に道を作ると罰金
                loss_slope = (torch.relu(slope - 0.5) * fake_B_01).mean()

                # Total Loss (物理制約に重み 2.0 を適用)
                loss_G = loss_GAN + (opt.lambda_l1 * loss_pixel) + (opt.lambda_dice * loss_dice_val) + \
                         (2.0 * loss_elevation) + (2.0 * loss_slope)
                
                loss_G.backward()
                optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            pred_real = discriminator(real_A, real_B)
            loss_real = criterion_GAN(pred_real, torch.full_like(pred_real, 0.9))
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            loop.set_description(f"Epoch [{epoch+1}/{opt.n_epoch}]")
            loop.set_postfix(loss_G=f"{loss_G.item():.4f}", loss_D=f"{loss_D.item():.4f}", Dice=f"{loss_dice_val.item():.4f}")

        if (epoch + 1) % 1 == 0:
            torch.save(generator.state_dict(), os.path.join(opt.params_dir, f"g_{epoch+1:04d}.pth"))
            with torch.no_grad():
                generator.eval(); samples = []
                num_samples = min(4, real_A.size(0))
                test_input = real_A[:num_samples]; test_target = real_B[:num_samples]; test_output = generator(test_input)
                for idx in range(num_samples):
                    column = torch.cat((test_input[idx, 0:1], test_input[idx, 1:2], test_output[idx], test_target[idx]), 1)
                    samples.append(column)
                vutils.save_image(torch.cat(samples, 2), os.path.join(opt.result_dir, f"epoch_{epoch+1}.png"), normalize=True)

if __name__ == "__main__":
    main()