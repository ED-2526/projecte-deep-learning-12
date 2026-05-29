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
import segmentation_models_pytorch as smp

# Imports del teu sistema
from dataset import get_train_test_validation
from train_myunet import MyUNet

# ==========================================
# 1. LA MÀGIA MATEMÀTICA: FOCAL + DICE LOSS
# ==========================================
class FocalDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.7, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = 1.0 - dice_weight
        
        # Gamma=2.0: Ignora els casos fàcils (Dice alt) i centra tota la
        # penalització en els tumors on la xarxa dubta o falla estrepitosament.
        self.focal = smp.losses.FocalLoss(mode='binary', gamma=gamma)
        self.dice = smp.losses.DiceLoss(mode='binary')
        
    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.dice_weight * loss_dice) + (self.focal_weight * loss_focal)

def calculate_metrics(logits, true_masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    inter = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum() - inter
    dice = (2. * inter + 1e-7) / (preds.sum() + true_masks.sum() + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return dice.item(), iou.item()

# ==========================================
# 2. ENTRENAMENT PRINCIPAL
# ==========================================
def main():
    config = {
        "lr": 1e-4,
        "epochs": 30,
        "batch_size": 32, # Tornem a 32 perquè avaluem el cervell sencer de 240x240
        "data_dir": "/home/edxnG12/data_processed_12/",
        "dropout": 0.3,
        "weight_decay": 1e-3
    }

    # Nom nou a W&B per no barrejar gràfiques
    wandb.init(project="brats-uab-project-aa", config=config, name="prova_MyUNet_FocalLoss_Custom")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciant entrenament amb FOCAL LOSS (Gamma=2.0) a: {device}")

    # Transformacions idèntiques al model base
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

    # Inicialitzem la teva arquitectura
    model = MyUNet(n_channels=4, n_classes=1, dropout_rate=config["dropout"]).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # --- LA NOVA FUNCIÓ DE COST ---
    criterion = FocalDiceLoss(dice_weight=0.7, gamma=2.0)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_dice = 0.0
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Nou fitxer per no sobrescriure l'històric
    SAVE_PATH = "./checkpoints/prova_best_custom_focal.pth"

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
                    # Capturem la imatge per a W&B (Canal 3 sol ser el FLAIR)
                    viz_data = (imgs[0, 3].cpu().numpy(), masks[0, 0].cpu().numpy(), 
                                (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy())

        avg_t_loss = train_loss / len(train_loader)
        avg_t_dice = train_dice / len(train_loader)
        avg_v_loss = val_loss / len(val_loader)
        avg_v_dice = val_dice / len(val_loader)
        
        scheduler.step(avg_v_dice)

        # Log a WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_t_loss,
            "train_dice": avg_t_dice,
            "val_loss": avg_v_loss,
            "val_dice": avg_v_dice,
            "val_iou": val_iou / len(val_loader),
            "lr": optimizer.param_groups[0]['lr'],
            "prediction_viz": [wandb.Image(viz_data[0], caption="FLAIR"),
                               wandb.Image(viz_data[1], caption="Mask Real"),
                               wandb.Image(viz_data[2].astype(float), caption="Predicció Focal")]
        })

        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🌟 Millora detectada! Focal Dice Guardat: {best_val_dice:.4f}")

    wandb.finish()
    print(f"✅ Entrenament Finalitzat. Model guardat a {SAVE_PATH}")

if __name__ == "__main__":
    main()