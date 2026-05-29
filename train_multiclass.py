import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Importem el dataset que acabem d'actualitzar a dalt
from dataset_mult import get_train_val_test_datasets

CONFIG = {
    "lr": 1e-4,
    "epochs": 30,
    "batch_size": 16,
    "num_classes": 4, # 0: Sa, 1: Necròtic, 2: Edema, 3: Actiu
    "data_dir": "/home/datasets/BraTS2020/data_processed"
}

# ==========================================
# 1. TVERSKY LOSS
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

# ==========================================
# 2. FUNCIONS DE MÈTRIQUES
# ==========================================
def calculate_multiclass_dice(logits, targets, num_classes=4):
    """Calcula el Dice Score per a cada classe de manera independent"""
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    dice_per_class = []
    # Ignorem la classe 0 (fons sa) per avaluar només el càncer real
    for c in range(1, num_classes):
        p_c = (preds == c).float()
        t_c = (targets == c).float()
        
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum()
        
        # Petit ajust de seguretat per evitar errors en llesques buides
        if union == 0:
            dice_per_class.append(1.0)
        else:
            dice = (2. * inter + 1e-7) / (union + 1e-7)
            dice_per_class.append(dice.item())
        
    return dice_per_class

# ==========================================
# 3. ENTRENAMENT PRINCIPAL
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Arrancant Motor Multiclasse amb TVERSKY LOSS en: {device}")
    
    # Inicialització de WandB actualitzada
    wandb.init(project="brats-multiclass", config=CONFIG, name="Unet_ResNet34_Tversky_Multi")
    
    # Carregar dades
    train_ds, val_ds, _ = get_train_val_test_datasets(
        CONFIG["data_dir"], 
        train_ratio=0.8, 
        val_ratio=0.1, 
        seed=42
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    # Instanciar arquitectura. NOTA: classes=4
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=4, 
        classes=CONFIG["num_classes"]
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    
    # ⚖️ Ús de Tversky Loss (alpha=0.3, beta=0.7)
    criterion = TverskyLoss(alpha=0.3, beta=0.7, num_classes=CONFIG["num_classes"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_macro_dice = 0.0
    checkpoint_path = "./checkpoints/best_multiclass_resnet_tversky.pth"
    os.makedirs("./checkpoints", exist_ok=True)
    
    for epoch in range(CONFIG["epochs"]):
        # --- TRAIN STAGE ---
        model.train()
        train_loss = 0.0
        train_dices = np.zeros(CONFIG["num_classes"] - 1) # Acumulador per [Dice_1, Dice_2, Dice_3]
        
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device).long()
            optimizer.zero_grad()
            
            logits = model(imgs)
            loss = criterion(logits, masks) 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculem el Dice del Train (fem .detach() perquè les mètriques no interfereixin amb la memòria de la gràfica)
            d_list_train = calculate_multiclass_dice(logits.detach(), masks, CONFIG["num_classes"])
            train_dices += np.array(d_list_train)
            
        # --- VALIDATION STAGE ---
        model.eval()
        val_loss = 0.0
        val_dices = np.zeros(CONFIG["num_classes"] - 1) 
        
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device).long()
                
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                d_list_val = calculate_multiclass_dice(logits, masks, CONFIG["num_classes"])
                val_dices += np.array(d_list_val)
                
        # --- MITJANES I LOGS ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        avg_train_dices = train_dices / len(train_loader)
        avg_val_dices = val_dices / len(val_loader)
        
        # Macro Dices: La mitjana dels encerts de les 3 subregions
        train_macro_dice = np.mean(avg_train_dices)
        val_macro_dice = np.mean(avg_val_dices)
        
        scheduler.step(val_macro_dice)
        
        # Enviem les dades amb els noms exactes a WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_dice": train_macro_dice,
            "val_dice": val_macro_dice,
            "val_dice_Necrotic (C1)": avg_val_dices[0],
            "val_dice_Edema (C2)": avg_val_dices[1],
            "val_dice_Actiu (C3)": avg_val_dices[2],
            "lr": optimizer.param_groups[0]['lr']
        })
        
        print(f"📊 [Resultats] Train Dice: {train_macro_dice:.4f} | Val Dice: {val_macro_dice:.4f}")
        print(f"   ↳ Necròtic: {avg_val_dices[0]:.4f} | Edema: {avg_val_dices[1]:.4f} | Actiu: {avg_val_dices[2]:.4f}")
        
        if val_macro_dice > best_macro_dice:
            best_macro_dice = val_macro_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🌟 RÈCORD! Nou millor model ResNet/Tversky guardat ({best_macro_dice:.4f})")
            
    wandb.finish()
    print("🏁 Entrenament multiclasse finalitzat amb èxit!")

if __name__ == "__main__":
    main()