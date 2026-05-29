import os
import torch
import torch.nn as nn
import wandb
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Imports del teu dataset.py
from dataset import get_train_test_validation
from focal_loss import FocalDiceLoss

# ==========================================
# 1. ARQUITECTURA: ATTENTION U-NET
# ==========================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Attention_block(nn.Module):
    """Mòdul d'Atenció que filtra les característiques de la Skip Connection"""
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, dropout_rate=0.3):
        super(AttentionUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64, dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, dropout_rate))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024, dropout_rate))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.conv_up1 = DoubleConv(1024, 512, dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.conv_up2 = DoubleConv(512, 256, dropout_rate)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.conv_up3 = DoubleConv(256, 128, dropout_rate)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.conv_up4 = DoubleConv(128, 64, dropout_rate)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder (Down)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder (Up) + Attention
        g1 = self.up1(x5)
        x4_att = self.att1(g=g1, x=x4)
        u1 = self.conv_up1(torch.cat([g1, x4_att], dim=1))
        
        g2 = self.up2(u1)
        x3_att = self.att2(g=g2, x=x3)
        u2 = self.conv_up2(torch.cat([g2, x3_att], dim=1))
        
        g3 = self.up3(u2)
        x2_att = self.att3(g=g3, x=x2)
        u3 = self.conv_up3(torch.cat([g3, x2_att], dim=1))
        
        g4 = self.up4(u3)
        x1_att = self.att4(g=g4, x=x1)
        u4 = self.conv_up4(torch.cat([g4, x1_att], dim=1))
        
        return self.outc(u4)

# ==========================================
# 2. FUNCIÓ DE COST I MÈTRIQUES
# ==========================================

class JointLoss(nn.Module):
    def __init__(self, dice_weight=0.7):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        dice_loss = 1 - (2. * inter + 1e-7) / (probs.sum() + targets.sum() + 1e-7)
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss

def calculate_metrics(logits, true_masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    inter = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum() - inter
    dice = (2. * inter + 1e-7) / (preds.sum() + true_masks.sum() + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return dice.item(), iou.item()

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================

def main():
    config = {"lr": 1e-4, "epochs": 30, "batch_size": 32, "data_dir": "/home/datasets/BraTS2020/data_processed_12"}
    wandb.init(project="brats-uab-project-aa", config=config, name="Attention_Focal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.Affine(scale=(0.8, 1.2), rotate=(-20, 20), shear=(-10, 10), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(10, 30), hole_width_range=(10, 30), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ])

    train_ds, val_ds, _ = get_train_test_validation(config["data_dir"])
    train_ds.augmentations = train_transform
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # Nova arquitectura
    model = AttentionUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-3)
    criterion = FocalDiceLoss(dice_weight=0.7, gamma=2.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_dice = 0.0
    SAVE_PATH = "./checkpoints/best_attention_focaldicee.pth"

    for epoch in range(config["epochs"]):
        model.train()
        train_loss, train_dice = 0,0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            imgs, masks = imgs.to(device), masks.to(device).float()
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            d, _ = calculate_metrics(logits, masks)
            train_dice += d

        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                imgs, masks = imgs.to(device), masks.to(device).float()
                logits = model(imgs)
                loss = criterion(logits, masks)
                b_dice, b_iou = calculate_metrics(logits, masks)

                val_loss += loss.item()
                val_dice += b_dice
                val_iou += b_iou

        avg_t_loss = train_loss / len(train_loader)
        avg_t_dice = train_dice / len(train_loader)
        avg_v_loss = val_loss / len(val_loader)
        avg_v_dice = val_dice / len(val_loader)

        scheduler.step(avg_v_dice)
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_t_loss,
            "train_dice": avg_t_dice,
            "val_loss": avg_v_loss, 
            "val_dice": avg_v_dice,
            "val_iou": val_iou / len(val_loader)
            })
        
        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🌟 Nou millor model Attention! Dice: {best_val_dice:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()