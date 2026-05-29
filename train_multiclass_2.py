import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Importem el dataset amb la divisió estricta a 3 bandes per evitar Data Leaking
from dataset_mult import get_train_val_test_datasets

# ==========================================
# 1. ARQUITECTURA CUSTOM UNET (ADAPTADA A MULTICLASSE)
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
    def __init__(self, n_channels=4, n_classes=4, dropout_rate=0.3): 
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
# 2. FUNCIONS DE PÈRDUA I MÈTRIQUES MULTICLASSE
# ==========================================

class TverskyLoss(nn.Module):
    """Tversky Loss adaptada a Multiclasse (Ignorant el fons)"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-7, num_classes=4):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        # Convertim la màscara [B, H, W] a One-Hot [B, 4, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        tversky_loss = 0.0
        # Calculem per a cada classe (ignorant el fons 0)
        for c in range(1, self.num_classes):
            p_c = probs[:, c].reshape(-1)
            t_c = targets_one_hot[:, c].reshape(-1)
            
            TP = (p_c * t_c).sum()
            FP = ((1 - t_c) * p_c).sum()
            FN = (t_c * (1 - p_c)).sum()
            
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            tversky_loss += (1 - tversky)
            
        return tversky_loss / (self.num_classes - 1)

def calculate_multiclass_metrics(logits, true_masks, num_classes=4):
    """Calcula el Dice Score individual per a cada subregió tumoral de la llesca"""
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    dices = []
    for c in range(1, num_classes):
        p_c = (preds == c).float()
        t_c = (true_masks == c).float()
        
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum()
        
        # Escut per evitar falsos càstigs en llesques sense el teixit específic
        if union == 0:
            dices.append(1.0)
        else:
            dice = (2. * inter + 1e-7) / (union + 1e-7)
            dices.append(dice.item())
        
    return dices # Retorna una llista: [Dice_C1, Dice_C2, Dice_C3]

# ==========================================
# 3. ENTRENAMENT PRINCIPAL
# ==========================================

def main():
    config = {
        "lr": 1e-4,
        "epochs": 30,
        "batch_size": 16, 
        "data_dir": os.path.abspath("/home/datasets/BraTS2020/data_processed"),
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "num_classes": 4
    }

    wandb.init(project="brats-multiclass", config=config, name="MyUNet_Multiclass_Tversky_Pura")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Augmentacions completes
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

    # Càrrega de dades sense Data Leaking compartint Seed global
    train_ds, val_ds, test_ds = get_train_val_test_datasets(config["data_dir"], train_ratio=0.8, val_ratio=0.1, seed=42)

    train_ds.augmentations = train_transform
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # Instanciem el teu MyUNet configurant n_classes = 4
    model = MyUNet(n_channels=4, n_classes=config["num_classes"], dropout_rate=config["dropout"]).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # 🔥 TVERSKY PURA RECUPERADA
    criterion = TverskyLoss(alpha=0.3, beta=0.7, num_classes=config["num_classes"])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_macro_dice = 0.0
    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(config["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        # -------------------
        # PHASE: TRAINING
        # -------------------
        model.train()
        train_loss = 0.0
        train_dices_accum = np.zeros(3) # Per acumular [C1, C2, C3]
        
        for imgs, masks in tqdm(train_loader, desc="Entrenant"):
            imgs, masks = imgs.to(device), masks.to(device).long()
            optimizer.zero_grad()
            
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Càlcul del Dice de Train segur (amb .detach())
            d_list = calculate_multiclass_metrics(logits.detach(), masks, config["num_classes"])
            train_dices_accum += np.array(d_list)

        # -------------------
        # PHASE: VALIDATION
        # -------------------
        model.eval()
        val_loss = 0.0
        val_dices_accum = np.zeros(3)
        viz_data = None

        with torch.no_grad():
            for i, (imgs, masks) in enumerate(tqdm(val_loader, desc="Validant")):
                imgs, masks = imgs.to(device), masks.to(device).long()
                logits = model(imgs)
                
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                d_list = calculate_multiclass_metrics(logits, masks, config["num_classes"])
                val_dices_accum += np.array(d_list)

                # Generació automatitzada de la imatge de diagnòstic de WandB
                if i == 0: 
                    probs = torch.softmax(logits, dim=1)
                    pred_classes = torch.argmax(probs, dim=1)
                    # Forcem una escala normalitzada multiplicant per 85
                    viz_data = (
                        imgs[0, 3].cpu().numpy(),                     
                        (masks[0].cpu().numpy() * 85).astype(np.uint8),  
                        (pred_classes[0].cpu().numpy() * 85).astype(np.uint8) 
                    )

        # Mitjanes aritmètiques de l'època
        avg_t_loss = train_loss / len(train_loader)
        avg_t_dices = train_dices_accum / len(train_loader)
        
        avg_v_loss = val_loss / len(val_loader)
        avg_v_dices = val_dices_accum / len(val_loader)
        
        # El Macro Dice Score és la mitjana de l'encert clínic global
        train_macro_dice = np.mean(avg_t_dices)
        val_macro_dice = np.mean(avg_v_dices)
        
        scheduler.step(val_macro_dice)

        # Log a WandB amb els noms estandarditzats
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_t_loss,
            "val_loss": avg_v_loss,
            "train_dice": train_macro_dice,
            "val_dice": val_macro_dice,
            "val_dice_Necrotic (C1)": avg_v_dices[0],
            "val_dice_Edema (C2)": avg_v_dices[1],
            "val_dice_Actiu (C3)": avg_v_dices[2],
            "lr": optimizer.param_groups[0]['lr'],
            "prediction_viz": [
                wandb.Image(viz_data[0], caption="FLAIR (Estructural)"),
                wandb.Image(viz_data[1], caption="Màscara Multiclasse Real"),
                wandb.Image(viz_data[2], caption="Predicció Multiclasse IA")
            ]
        })

        print(f"📊 [Resultats] Train Dice: {train_macro_dice:.4f} | Val Dice: {val_macro_dice:.4f}")
        print(f"   ↳ C1 (Necròtic): {avg_v_dices[0]:.4f} | C2 (Edema): {avg_v_dices[1]:.4f} | C3 (Actiu): {avg_v_dices[2]:.4f}")

        if val_macro_dice > best_macro_dice:
            best_macro_dice = val_macro_dice
            torch.save(model.state_dict(), "./checkpoints/best_myunet_tversky.pth")
            print(f"🌟 RÈCORD! Nou millor model Tversky guardat ({best_macro_dice:.4f})")

    wandb.finish()

if __name__ == "__main__":
    main()