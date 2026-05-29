import os
import torch
import torch.nn as nn
import wandb
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_train_val_datasets, get_train_test_validation

class JointLoss(nn.Module):
    def __init__(self, dice_weight=0.7):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

        self.dw = dice_weight
        self.bw = 1 - dice_weight
        
    def forward(self, out, target):
        return self.dw * self.dice(out, target) + self.bw * self.bce(out, target)

def calculate_metrics(logits, true_masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().view(-1)
    true_masks = true_masks.view(-1)
    
    inter = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum() - inter
    dice = (2. * inter + 1e-7) / (preds.sum() + true_masks.sum() + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return dice.item(), iou.item()

def main():
    config = {
        "lr": 1e-4,
        "epochs": 30, 
        "batch_size": 32, 
        "data_dir": "/home/datasets/BraTS2020/data_processed_12",
        "backbone": "resnet34"
    }

    wandb.init(
        project="brats-uab-project-aa",
        entity="1710333-universitat-aut-noma-de-barcelona",
        config=config,
        name="prova_unet-resnet34-run"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Arrancando en: {device}")

    # 1. Definimos las aumentaciones (Data Augmentation)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    # 2. Obtenemos los datasets usando TU código original (sin pasarle augmentations)
    #train_ds, val_ds = get_train_val_datasets(config["data_dir"])
    train_ds, val_ds, test_ds = get_train_test_validation(config["data_dir"])
    
    # 3. EL TRUCO: Le inyectamos las transformaciones solo al dataset de entrenamiento
    train_ds.augmentations = train_transform
    
    # 4. Creamos los Loaders
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name=config["backbone"], encoder_weights="imagenet", in_channels=4, classes=1).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    criterion = JointLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_dice = 0.0
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        model.train()
        train_loss, train_dice = 0.0, 0.0
        for imgs, masks in tqdm(train_loader, desc="Entrenant"):
            imgs, masks = imgs.to(device), masks.to(device)
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
            for imgs, masks in tqdm(val_loader, desc="Validant"):
                imgs, masks = imgs.to(device), masks.to(device)
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
        avg_v_iou = val_iou / len(val_loader)
        
        scheduler.step(avg_v_dice)

        print(f"Train Loss: {avg_t_loss:.4f} | Train Dice: {avg_t_dice:.4f} | Val Loss: {avg_v_loss:.4f} | Val Dice: {avg_v_dice:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_t_loss,
            "train_dice": avg_t_dice,
            "val_loss": avg_v_loss,
            "val_dice": avg_v_dice,
            "val_iou": avg_v_iou,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "prova_unet_resnet34_bestt.pth"))
            print("🌟 Nou millor model guardat!")

    wandb.finish()

if __name__ == "__main__":
    main()