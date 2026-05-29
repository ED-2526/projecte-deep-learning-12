import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset_mult import get_train_val_test_datasets

# ==========================================
# 1. ATTENTION GATE MULTICLASE
# ==========================================

class AttentionGate(nn.Module):
    """Mecanismo de atención para mejorar la precisión en bordes tumorales"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
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

class AttentionUNetMulticlass(nn.Module):
    """UNet con Attention Gates para segmentación multiclase de tumores cerebrales"""
    def __init__(self, n_channels=4, n_classes=4, dropout_rate=0.3):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(n_channels, 64,   dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64,   128,  dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128,  256,  dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256,  512,  dropout_rate))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512,  1024, dropout_rate))

        # Decoder con Attention Gates
        self.up1      = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1     = AttentionGate(512, 512, 256)
        self.conv_up1 = DoubleConv(1024, 512, dropout_rate)

        self.up2      = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2     = AttentionGate(256, 256, 128)
        self.conv_up2 = DoubleConv(512, 256, dropout_rate)

        self.up3      = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3     = AttentionGate(128, 128, 64)
        self.conv_up3 = DoubleConv(256, 128, dropout_rate)

        self.up4      = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4     = AttentionGate(64, 64, 32)
        self.conv_up4 = DoubleConv(128, 64, dropout_rate)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5)
        x4_att = self.att1(u1, x4)
        u1 = torch.cat([u1, x4_att], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        x3_att = self.att2(u2, x3)
        u2 = torch.cat([u2, x3_att], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        x2_att = self.att3(u3, x2)
        u3 = torch.cat([u3, x2_att], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        x1_att = self.att4(u4, x1)
        u4 = torch.cat([u4, x1_att], dim=1)
        u4 = self.conv_up4(u4)

        return self.outc(u4)

# ==========================================
# 2. LOSS FUNCTIONS MULTICLASE (RECUPERADES)
# ==========================================

class MulticlassDiceLoss(nn.Module):
    """Dice Loss adaptat a One-Hot encoding per a entorns multiclasse"""
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
    """Combina el CrossEntropyLoss amb el Dice Multiclasse regional"""
    def __init__(self, dice_weight=0.7, num_classes=4):
        super().__init__()
        self.dice = MulticlassDiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss() 
        self.dw = dice_weight
        self.cw = 1.0 - dice_weight
        
    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.cw * self.ce(logits, targets)

# ==========================================
# 3. MÉTRICAS MULTICLASE
# ==========================================

def calculate_multiclass_metrics(logits, targets, num_classes=4):
    """Calcula precisión, recall, F1 por clase"""
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)   # [B, H, W]

    metrics = {}
    for c in range(num_classes):
        pred_c = (preds == c).float().view(-1)
        true_c = (targets == c).float().view(-1)

        tp = (pred_c * true_c).sum()
        fp = (pred_c * (1 - true_c)).sum()
        fn = ((1 - pred_c) * true_c).sum()

        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f1        = 2 * (precision * recall) / (precision + recall + 1e-7)

        metrics[f"class_{c}"] = {
            "precision": precision.item(),
            "recall":    recall.item(),
            "f1":        f1.item()
        }

    return metrics

# ==========================================
# 4. VISUALIZACIÓN BEST / WORST
# ==========================================

CLASS_COLORS = {
    0: [0,   0,   0  ],   # Background
    1: [255, 0,   0  ],   # Necrosis/Core
    2: [0,   255, 0  ],   # Edema peritumoral
    3: [0,   0,   255],   # Tumor realzado
}
CLASS_NAMES = ["Background", "Necrosis/Core", "Edema peritumoral", "Tumor realzado"]

def masks_to_rgb(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask_np == cls] = color
    return rgb

def save_best_worst_predictions(model, val_loader, device, epoch, save_dir="./visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    samples = [] 

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device).long()

            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1) 

            for b in range(imgs.shape[0]):
                pred_b = preds[b].cpu().numpy() 
                mask_b = masks[b].cpu().numpy() 
                img_b  = imgs[b].cpu().numpy()  

                dices = []
                for c in [1, 2, 3]:
                    p     = (pred_b == c).astype(float).flatten()
                    t     = (mask_b == c).astype(float).flatten()
                    inter = (p * t).sum()
                    d     = (2 * inter + 1e-7) / (p.sum() + t.sum() + 1e-7)
                    dices.append(d)
                mean_dice = float(np.mean(dices))

                samples.append((mean_dice, img_b, mask_b, pred_b))

    if not samples:
        return

    samples.sort(key=lambda x: x[0])
    cases = [("WORST", samples[0]), ("BEST", samples[-1])]

    wandb_imgs = {}

    for label, (dice, img_np, true_mask, pred_mask) in cases:
        slice_img = img_np[1].copy()
        vmin, vmax = slice_img.min(), slice_img.max()
        if vmax > vmin:
            slice_img = (slice_img - vmin) / (vmax - vmin)
        else:
            slice_img = np.zeros_like(slice_img)

        true_rgb = masks_to_rgb(true_mask)
        pred_rgb = masks_to_rgb(pred_mask)

        overlay          = np.stack([slice_img] * 3, axis=-1)
        pred_norm        = pred_rgb.astype(float) / 255.0
        tumor_bool       = pred_mask > 0
        overlay[tumor_bool] = (
            0.45 * overlay[tumor_bool] +
            0.55 * pred_norm[tumor_bool]
        )
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor('#0d0d0d')

        panels = [
            ("T1ce (entrada)",  slice_img,  "gray"),
            ("Máscara real",    true_rgb,   None),
            ("Predicción",      pred_rgb,   None),
            ("Overlay",         overlay,    None),
        ]

        for ax, (title, data, cmap) in zip(axes, panels):
            ax.set_facecolor('#0d0d0d')
            if cmap:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(data)
            ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
            ax.axis('off')

        legend_patches = [
            mpatches.Patch(color=[c / 255 for c in col], label=f"Clase {cls}: {name}")
            for cls, (col, name) in enumerate(zip(CLASS_COLORS.values(), CLASS_NAMES))
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4, framealpha=0.25, fontsize=10, labelcolor='white', bbox_to_anchor=(0.5, -0.06), facecolor='#1a1a1a', edgecolor='#444444')

        emoji  = "🌟" if label == "BEST" else "❌"
        suptitle = f"{emoji}  Epoch {epoch:03d}  —  {label} case  |  Dice tumoral medio: {dice:.4f}"
        fig.suptitle(suptitle, color='white', fontsize=13, fontweight='bold', y=1.03)

        fname = os.path.join(save_dir, f"epoch{epoch:03d}_{label.lower()}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=130, facecolor='#0d0d0d', pad_inches=0.35)
        plt.close(fig)

        wandb_imgs[f"viz/{label.lower()}_epoch{epoch:03d}"] = wandb.Image(fname, caption=f"{label} | Dice={dice:.4f} | Epoch {epoch}")

    try:
        wandb.log(wandb_imgs, step=epoch)
    except Exception as e:
        pass


# ==========================================
# 5. ENTRENAMIENTO PRINCIPAL
# ==========================================

def main():
    config = {
        "lr":           1e-4,
        "epochs":       30,
        "batch_size":   8,
        "data_dir":     "/home/datasets/BraTS2020/data_processed/",
        "dropout":      0.3,
        "weight_decay": 1e-3,
        "num_classes":  4,
        "viz_every":    5,    
    }

    wandb.init(
        project="brats-multiclass",
        config=config,
        name="AttentionUNet_Multiclass_DiceCE" # Nom actualitzat per reflectir la Loss!
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Entrenando AttentionUNet Multiclase en: {device}")

    # ---- Transformaciones de augmentación ----
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
    ], additional_targets={'mask': 'mask'})

    # ---- Datasets ----
    train_ds, val_ds, test_ds = get_train_val_test_datasets(config["data_dir"], train_ratio=0.8, val_ratio=0.1, seed=42)
    train_ds.augmentations = train_transform

    print(
        f"Total pacients: {len(train_ds) + len(val_ds)} | "
        f"Train: {len(train_ds)} | Val: {len(val_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # ---- Modelo ----
    model = AttentionUNetMulticlass(
        n_channels=4,
        n_classes=config["num_classes"],
        dropout_rate=config["dropout"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])  
    
    # 🔥 RECUPERAT: Utilitzem el JointMulticlassLoss (70% Dice, 30% CE)
    criterion = JointMulticlassLoss(dice_weight=0.7, num_classes=config["num_classes"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_dice  = 0.0
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config["epochs"]):

        # ======== TRAIN ========
        model.train()
        train_loss    = 0.0
        train_metrics = {f"class_{c}": {"f1": 0.0} for c in range(config["num_classes"])}

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [TRAIN]"):
            imgs, masks = imgs.to(device), masks.to(device).long()

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            batch_metrics = calculate_multiclass_metrics(logits.detach(), masks, config["num_classes"])
            for c in range(config["num_classes"]):
                train_metrics[f"class_{c}"]["f1"] += batch_metrics[f"class_{c}"]["f1"]

        # ======== VALIDATION ========
        model.eval()
        val_loss    = 0.0
        val_metrics = {f"class_{c}": {"f1": 0.0, "recall": 0.0} for c in range(config["num_classes"])}

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [VAL]"):
                imgs, masks = imgs.to(device), masks.to(device).long()
                logits = model(imgs)
                loss   = criterion(logits, masks)
                val_loss += loss.item()

                batch_metrics = calculate_multiclass_metrics(logits, masks, config["num_classes"])
                for c in range(config["num_classes"]):
                    val_metrics[f"class_{c}"]["f1"]     += batch_metrics[f"class_{c}"]["f1"]
                    val_metrics[f"class_{c}"]["recall"] += batch_metrics[f"class_{c}"]["recall"]

        # ======== MÉTRICAS RESUMEN ========
        n_train = len(train_loader)
        n_val   = len(val_loader)

        avg_train_loss      = train_loss / n_train
        avg_val_loss        = val_loss   / n_val
        
        train_f1_core = train_metrics["class_1"]["f1"] / n_train
        train_f1_peri = train_metrics["class_2"]["f1"] / n_train
        train_f1_enh  = train_metrics["class_3"]["f1"] / n_train
        avg_train_dice = (train_f1_core + train_f1_peri + train_f1_enh) / 3.0
        
        val_f1_core = val_metrics["class_1"]["f1"] / n_val
        val_f1_peri = val_metrics["class_2"]["f1"] / n_val
        val_f1_enh  = val_metrics["class_3"]["f1"] / n_val
        avg_val_dice = (val_f1_core + val_f1_peri + val_f1_enh) / 3.0

        avg_val_recall_tumor = val_metrics["class_1"]["recall"] / n_val

        print(
            f"📊 Epoch {epoch+1:3d} | "
            f"Loss train={avg_train_loss:.4f} val={avg_val_loss:.4f} | "
            f"Train Dice={avg_train_dice:.4f} Val Dice={avg_val_dice:.4f} | "
            f"F1_Core={val_f1_core:.4f} | Recall_Core={avg_val_recall_tumor:.4f}"
        )

        scheduler.step(val_f1_core)

        # ======== LOG WANDB ========
        wandb.log({
            "epoch":               epoch + 1,
            "train_loss":          avg_train_loss,
            "val_loss":            avg_val_loss,
            "train_dice":          avg_train_dice, 
            "val_dice":            avg_val_dice,    
            "val_f1_core":         val_f1_core,
            "val_f1_peritumoral":  val_f1_peri,
            "val_f1_enhanced":     val_f1_enh,
            "val_recall_core":     avg_val_recall_tumor,
            "lr":                  optimizer.param_groups[0]['lr']
        }, step=epoch + 1)

        # ======== CHECKPOINT ========
        if val_f1_core > best_val_dice:
            best_val_dice = val_f1_core
            ckpt_path = os.path.join(CHECKPOINT_DIR, "attention_joint_multiclass_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"🌟 Mejora! F1 Tumor Core: {best_val_dice:.4f}  →  {ckpt_path}")

        # ======== VISUALIZACIÓN BEST / WORST ========
        if (epoch + 1) % config["viz_every"] == 0 or epoch == 0:
            save_best_worst_predictions(
                model      = model,
                val_loader = val_loader,
                device     = device,
                epoch      = epoch + 1,
                save_dir   = "./visualizations"
            )

    wandb.finish()
    print("✅ Entrenamiento finalizado.")

if __name__ == "__main__":
    main()