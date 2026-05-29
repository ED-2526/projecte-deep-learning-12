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

from dataset_mult import get_train_val_test_datasets
from attention import AttentionUNet

# ==========================================
# 1. LA MÀGIA DEL RETALL (CROP DÍNAMIC)
# ==========================================
def crop_and_resize(img, mask, target_size=(128, 128), margin=16):
    """
    S'utilitza només a l'ENTRENAMENT. Retalla usant la màscara perfecta.
    """
    coords = torch.nonzero(mask > 0)
    
    if len(coords) == 0:
        c_img = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
        c_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest')[0, 0].long()
        return c_img, c_mask

    y_min, y_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    x_min, x_max = coords[:, 1].min().item(), coords[:, 1].max().item()

    y_min = max(0, y_min - margin)
    y_max = min(mask.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(mask.shape[1], x_max + margin)

    cropped_img = img[:, y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    c_img = F.interpolate(cropped_img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
    c_mask = F.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest')[0, 0].long()

    return c_img, c_mask

def crop_and_resize_eval(img, true_mask, binary_pred, target_size=(128, 128), margin=16):
    """
    🔥 NOU: S'utilitza a la VALIDACIÓ.
    Calcula la caixa usant 'binary_pred' (El model Binari), 
    però retalla la 'true_mask' perquè puguem calcular les mètriques d'error de l'Especialista.
    """
    coords = torch.nonzero(binary_pred > 0)
    
    if len(coords) == 0:
        c_img = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
        c_mask = F.interpolate(true_mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest')[0, 0].long()
        return c_img, c_mask

    y_min, y_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    x_min, x_max = coords[:, 1].min().item(), coords[:, 1].max().item()

    y_min = max(0, y_min - margin)
    y_max = min(true_mask.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(true_mask.shape[1], x_max + margin)

    cropped_img = img[:, y_min:y_max, x_min:x_max]
    cropped_mask = true_mask[y_min:y_max, x_min:x_max] 

    c_img = F.interpolate(cropped_img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
    c_mask = F.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest')[0, 0].long()

    return c_img, c_mask

# ==========================================
# 2. FUNCIONS DE PÈRDUA I MÈTRIQUES
# ==========================================
class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-7):
        super(MulticlassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dice_loss = 0.0
        for c in range(1, self.num_classes):
            iflat = probs[:, c].reshape(-1)
            tflat = targets_one_hot[:, c].reshape(-1)
            intersection = (iflat * tflat).sum()
            dice = (2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)
            dice_loss += (1 - dice)
        return dice_loss / (self.num_classes - 1)

class JointMulticlassLoss(nn.Module):
    def __init__(self, dice_weight=0.7, num_classes=4):
        super().__init__()
        self.dice = MulticlassDiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dw = dice_weight
        self.cw = 1.0 - dice_weight
    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.cw * self.ce(logits, targets)

def calculate_multiclass_metrics(preds, true_masks, num_classes=4):
    dices = []
    for c in range(1, num_classes):
        p_c = (preds == c).float()
        t_c = (true_masks == c).float()
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum()
        dice = (2. * inter + 1e-7) / (union + 1e-7)
        dices.append(dice.item())
    
    p_wt = (preds > 0).float()
    t_wt = (true_masks > 0).float()
    inter_wt = (p_wt * t_wt).sum()
    union_wt = p_wt.sum() + t_wt.sum()
    dice_wt = (2. * inter_wt + 1e-7) / (union_wt + 1e-7)
    
    return dices, dice_wt.item()

# ==========================================
# 3. ENTRENAMENT PRINCIPAL
# ==========================================
def main():
    config = {
        "lr": 1e-4,
        "epochs": 30,
        "batch_size": 16, 
        "data_dir": os.path.abspath("/home/datasets/BraTS2020/data_processed"),
        "weight_decay": 1e-3,
        "num_classes": 4,
        "target_size": 128,
        "binary_checkpoint": "./checkpoints/best_attention.pth" # 🔥 RUTA DEL TEU MODEL BINARI
    }

    wandb.init(project="brats-multiclass", config=config, name="Cascade_Crop_Train_Realistic")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(scale=(0.8, 1.2), rotate=(-20, 20), p=0.5),
    ])

    train_ds, val_ds, _ = get_train_val_test_datasets(config["data_dir"], train_ratio=0.8, val_ratio=0.1, seed=42)
    train_ds.augmentations = train_transform
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    print("🤖 Carregant el Model Binari de Rastreig...")
    binary_model = AttentionUNet(n_channels=4, n_classes=1).to(device)
    binary_model.load_state_dict(torch.load(config["binary_checkpoint"], map_location=device))
    binary_model.eval() 

    model = AttentionUNet(n_channels=4, n_classes=config["num_classes"]).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = JointMulticlassLoss(dice_weight=0.7, num_classes=config["num_classes"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_macro_dice = 0.0
    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(config["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        # --- TRAIN (Ideal: Amb Ground Truth) ---
        model.train()
        train_loss = 0.0
        
        # 🟢 AFEGIT: Acumuladors pel Train Dice
        train_dice_macro_accum = 0.0
        train_wt_accum = 0.0
        
        for imgs, masks in tqdm(train_loader, desc="Entrenant (Crop Ideal)"):
            cropped_imgs, cropped_masks = [], []
            for i in range(imgs.shape[0]):
                c_img, c_mask = crop_and_resize(imgs[i], masks[i], target_size=(config["target_size"], config["target_size"]))
                cropped_imgs.append(c_img)
                cropped_masks.append(c_mask)
            
            imgs = torch.stack(cropped_imgs).to(device)
            masks = torch.stack(cropped_masks).to(device).long()
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 🟢 AFEGIT: Càlcul de les mètriques d'entrenament amb detach()
            with torch.no_grad():
                probs = torch.softmax(logits.detach(), dim=1)
                preds = torch.argmax(probs, dim=1)
                for b in range(imgs.shape[0]):
                    d_list, wt_score = calculate_multiclass_metrics(preds[b], masks[b])
                    train_dice_macro_accum += np.mean(d_list)
                    train_wt_accum += wt_score

        # --- VALIDACIÓ (Realista: Amb Prediccions Binàries) ---
        model.eval()
        val_loss, val_dice_macro_accum, val_wt_accum = 0.0, 0.0, 0.0
        viz_data = None

        with torch.no_grad():
            for i, (imgs, true_masks) in enumerate(tqdm(val_loader, desc="Validant (Crop Realista)")):
                imgs_device = imgs.to(device)
                
                bin_logits = binary_model(imgs_device)
                bin_preds = (torch.sigmoid(bin_logits) > 0.5).long().squeeze(1).cpu()

                cropped_imgs, cropped_masks = [], []
                for j in range(imgs.shape[0]):
                    c_img, c_mask = crop_and_resize_eval(imgs[j], true_masks[j], bin_preds[j], target_size=(config["target_size"], config["target_size"]))
                    cropped_imgs.append(c_img)
                    cropped_masks.append(c_mask)
                
                imgs_crop = torch.stack(cropped_imgs).to(device)
                masks_crop = torch.stack(cropped_masks).to(device).long()
                
                logits = model(imgs_crop)
                loss = criterion(logits, masks_crop)
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                for b in range(imgs_crop.shape[0]):
                    d_list, wt_score = calculate_multiclass_metrics(preds[b], masks_crop[b])
                    val_dice_macro_accum += np.mean(d_list)
                    val_wt_accum += wt_score

                if i == 0: 
                    viz_data = (imgs_crop[0, 3].cpu().numpy(), (masks_crop[0].cpu().numpy() * 85).astype(np.uint8), (preds[0].cpu().numpy() * 85).astype(np.uint8))

        # --- CÀLCULS FINALS ---
        total_train_samples = len(train_loader.dataset)
        total_val_samples = len(val_loader.dataset)
        
        avg_t_loss = train_loss / len(train_loader)
        avg_v_loss = val_loss / len(val_loader)
        
        # 🟢 AFEGIT: Mitjanes del Train Dice
        final_train_macro_dice = train_dice_macro_accum / total_train_samples
        final_train_wt_dice = train_wt_accum / total_train_samples
        
        final_macro_dice = val_dice_macro_accum / total_val_samples
        final_wt_dice = val_wt_accum / total_val_samples
        
        scheduler.step(final_macro_dice)

        # 🟢 AFEGIT: Integració al WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_t_loss,
            "train_dice": final_train_macro_dice,     # Train Macro
            "train_dice_WT": final_train_wt_dice,     # Train Whole Tumor
            "val_loss": avg_v_loss,
            "val_dice": final_macro_dice,
            "val_dice_WT": final_wt_dice,
            "lr": optimizer.param_groups[0]['lr'],
            "crop_prediction_viz": [
                wandb.Image(viz_data[0], caption="FLAIR (Retall del Binari)"), 
                wandb.Image(viz_data[1], caption="Real"), 
                wandb.Image(viz_data[2], caption="Predicció")
            ]
        })

        # 🟢 AFEGIT: Mostrar-ho de forma clara pel terminal
        print(f"📊 TRAIN | Loss: {avg_t_loss:.4f} | Macro: {final_train_macro_dice:.4f} | WT: {final_train_wt_dice:.4f}")
        print(f"📊 VAL   | Loss: {avg_v_loss:.4f} | Macro: {final_macro_dice:.4f} | WT: {final_wt_dice:.4f}")

        if final_macro_dice > best_macro_dice:
            best_macro_dice = final_macro_dice
            torch.save(model.state_dict(), "./checkpoints/best_multiclass_CROP_realistic.pth")
            print(f"🌟 Model CROP Realista guardat!")

    wandb.finish()

if __name__ == "__main__":
    main()