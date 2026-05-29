import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import segmentation_models_pytorch as smp  # NOU IMPORT AFEGIT

# Importem el teu dataset.py (assegura't que el fitxer es digui dataset.py)
from dataset import get_train_val_datasets, get_train_test_validation

# ==========================================
# 1. ARQUITECTURA CUSTOM UNET
# ==========================================

class DoubleConv(nn.Module):
    """(convolució => [BN] => ReLU) * 2 amb Dropout opcional"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MyUNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, dropout_rate=0.3):
        super(MyUNet, self).__init__()
        
        # Encoder (Baixada)
        self.inc = DoubleConv(n_channels, 64, dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, dropout_rate))
        
        # Bridge / Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024, dropout_rate))

        # Decoder (Pujada)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512, dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256, dropout_rate)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128, dropout_rate)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64, dropout_rate)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)
        
        return self.outc(u4)

# ==========================================
# 2. FUNCIONS DE PÈRDUA I MÈTRIQUES
# ==========================================
# Mantinc les velles aquí definides perquè l'script sigui idèntic, encara que no les usem
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        iflat = probs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))

class JointLoss(nn.Module):
    def __init__(self, dice_weight=0.7):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dw = dice_weight
        self.bw = 1 - dice_weight
        
    def forward(self, out, target):
        return self.dw * self.dice(out, target) + self.bw * self.bce(out, target)

def calculate_metrics(logits, true_masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    inter = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum() - inter
    dice = (2. * inter + 1e-7) / (preds.sum() + true_masks.sum() + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return dice.item(), iou.item()

# ==========================================
# 3. ENTRENAMENT PRINCIPAL
# ==========================================

def main():
    config = {
        "lr": 1e-4,
        "epochs": 30,
        "batch_size": 32,  # Reduït per seguretat de memòria amb MyUNet (més filtres)
        "data_dir": os.path.abspath("/home/edxnG12/data_processed_12/"),
        "dropout": 0.3,
        "weight_decay": 1e-3
    }

    # CANVI NOM W&B PER EVITAR SOBRESCRIPCIÓ
    wandb.init(project="brats-uab-project-aa", config=config, name="prova_resnet_Tversky")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformacions corregides per a versions noves d'Albumentations
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

    train_ds, val_ds, test_ds = get_train_test_validation(config["data_dir"])

    train_ds.augmentations = train_transform
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # Instanciem el nostre model propi
    #model = MyUNet(n_channels=4, n_classes=1, dropout_rate=config["dropout"]).to(device)
    model = smp.Unet(encoder_name="resnet34", in_channels=4, classes=1).to(device)  # NOU MODEL AFEGIT PER COMPARAR
    
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # LA MÀGIA: Substituïm la JointLoss per la Tversky (alpha=0.3 FP, beta=0.7 FN)
    criterion = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, from_logits=True)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_dice = 0.0
    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(config["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        # Training
        model.train()
        train_loss, train_dice = 0.0, 0.0
        for imgs, masks in tqdm(train_loader, desc="Entrenant"):
            imgs, masks = imgs.to(device), masks.to(device).float()
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            d, _ = calculate_metrics(logits, masks)
            train_dice += d

        # Validation
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        viz_data = None

        with torch.no_grad():
            for i, (imgs, masks) in enumerate(tqdm(val_loader, desc="Validant")):
                imgs, masks = imgs.to(device), masks.to(device).float()
                logits = model(imgs)
                loss = criterion(logits, masks)
                b_dice, b_iou = calculate_metrics(logits, masks)
                
                val_loss += loss.item()
                val_dice += b_dice
                val_iou += b_iou

                if i == 0: 
                    # Agafem FLAIR (canal 3) per visualitzar
                    viz_data = (imgs[0, 3].cpu().numpy(), masks[0, 0].cpu().numpy(), 
                                (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy())

        avg_t_loss = train_loss / len(train_loader)
        avg_t_dice = train_dice / len(train_loader)
        avg_v_loss = val_loss / len(val_loader)
        avg_v_dice = val_dice / len(val_loader)
        
        scheduler.step(avg_v_dice)

        # Log a WandB mantenint noms originals
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_t_loss,
            "train_dice": avg_t_dice,
            "val_loss": avg_v_loss,
            "val_dice": avg_v_dice,
            "val_iou": val_iou / len(val_loader),
            "loss_gap": avg_v_loss - avg_t_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "prediction_viz": [wandb.Image(viz_data[0], caption="FLAIR"),
                               wandb.Image(viz_data[1], caption="Mask Real"),
                               wandb.Image(viz_data[2].astype(float), caption="Predicció")]
        })

        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            # CANVI NOM DEL FITXER PER NO SOBRESCRIURE L'ALTRE EXPERIMENT
            torch.save(model.state_dict(), "./checkpoints/prova_resnet_tversky.pth")
            print(f"🌟 Millora detectada! Dice: {best_val_dice:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()