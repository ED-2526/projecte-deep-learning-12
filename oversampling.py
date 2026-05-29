import os
import torch
import torch.nn as nn
import wandb
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Imports del teu sistema
from dataset import get_train_test_validation
from train_myunet import MyUNet
from focal_loss import FocalDiceLoss

# ==========================================
# 1. EL MOTOR DE DATA ENGINEERING (OVERSAMPLING)
# ==========================================
def create_hard_example_sampler(dataset):
    print("\n⚖️ [DATA ENGINEERING] Calculant pesos per a Hard Example Oversampling...")
    print("⏳ Això trigarà 1-2 minuts perquè estem analitzant milers de llesques prèviament...")
    
    weights = []
    
    # Iterem pel dataset per comptar quants píxels de tumor té cada imatge
    for i in tqdm(range(len(dataset)), desc="Analitzant mides de tumor"):
        # Agafem només la màscara (no ens cal la imatge per saber el pes)
        _, mask = dataset[i]
        tumor_pixels = mask.sum().item()
        
        # LÒGICA DE PESOS (El cor de l'experiment)
        if tumor_pixels == 0:
            weights.append(0.1)
        elif tumor_pixels < 50:
            weights.append(0.5)
        elif tumor_pixels < 300:
            weights.append(5.0)
        elif tumor_pixels < 1000:
            weights.append(2.0)
        else:
            weights.append(1.0)
            
    # Creem el sampler de PyTorch
    weights_tensor = torch.DoubleTensor(weights)
    # num_samples=len(weights) significa que l'època durarà el mateix de sempre
    # replacement=True permet que una mateixa imatge difícil surti diverses vegades
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
    
    print("✅ Pesos calculats! Preparant DataLoader...\n")
    return sampler

# ==========================================
# 2. FUNCIÓ DE COST ORIGINAL (Per aïllar l'experiment)
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

def calculate_metrics(logits, true_masks):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * true_masks).sum()
    dice = (2. * inter + 1e-7) / (preds.sum() + true_masks.sum() + 1e-7)
    return dice.item()

# ==========================================
# 3. ENTRENAMENT PRINCIPAL
# ==========================================
def main():
    config = {
        "lr": 1e-4,
        "epochs": 30, # Mantenim les 30 èpoques òptimes
        "batch_size": 32,
        "data_dir": "/home/datasets/BraTS2020/data_processed_12/",
        "dropout": 0.3
    }

    # Nom clar a W&B per diferenciar aquest experiment
    wandb.init(project="brats-uab-project-aa", config=config, name="Oversampling_Focal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciant Experiment 3: OVERSAMPLING a: {device}")

    # Càrrega de dades
    train_ds, val_ds, _ = get_train_test_validation(config["data_dir"])
    
    # ---------------------------------------------------------
    # APLICACIÓ DEL SAMPLER
    # ---------------------------------------------------------
    train_sampler = create_hard_example_sampler(train_ds)
    
    # CRÍTIC: Si uses 'sampler', NO pots posar 'shuffle=True' al DataLoader
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    # ---------------------------------------------------------

    # Utilitzem l'arquitectura BASE (sense Attention) per comparar justament
    model = MyUNet(n_channels=4, n_classes=1, dropout_rate=config["dropout"]).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-3)
    criterion = FocalDiceLoss(dice_weight=0.7, gamma=2.0) 
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_dice = 0.0
    os.makedirs("./checkpoints", exist_ok=True)
    SAVE_PATH = "./checkpoints/best_focal_oversampling.pth"

    for epoch in range(config["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        # Training
        model.train()
        train_dice = 0.0
        for imgs, masks in tqdm(train_loader, desc="Entrenant (Hard Cases)"):
            imgs, masks = imgs.to(device), masks.to(device).float()
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_dice += calculate_metrics(logits, masks)

        # Validation
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="Validant"):
                imgs, masks = imgs.to(device), masks.to(device).float()
                logits = model(imgs)
                val_dice += calculate_metrics(logits, masks)

        avg_t_dice = train_dice / len(train_loader)
        avg_v_dice = val_dice / len(val_loader)
        
        scheduler.step(avg_v_dice)

        wandb.log({
            "epoch": epoch + 1,
            "train_dice": avg_t_dice,
            "val_dice": avg_v_dice,
            "lr": optimizer.param_groups[0]['lr']
        })

        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🌟 Millora detectada! Model Guardat: {best_val_dice:.4f}")

    wandb.finish()
    print(f"✅ Entrenament Finalitzat. Model guardat a {SAVE_PATH}")

if __name__ == "__main__":
    main()