import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Imports del teu sistema
from dataset_mult import get_train_val_test_datasets

try:
    from medpy.metric.binary import sensitivity
except ImportError:
    os.system('pip install medpy')
    from medpy.metric.binary import sensitivity

# ==========================================
# 1. CONFIGURACIÓ GLOBAL
# ==========================================
DATA_DIR = "/home/datasets/BraTS2020/data_processed"
CHECKPOINT_PATH = "./checkpoints/best_multiclass_tversky_model.pth"
CSV_SAVE_PATH = "./test_multi/metrics_best_multiclass_tversky_model.csv"
DASHBOARD_SAVE_PATH = "./test_multi/terminalbest_multiclass_tversky_model.png"
BRAIN_SAVE_PATH = "./test_multi/cerverll_terminalbest_multiclass_tversky_model.png"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Etiquetes oficials de les 4 classes
CLASS_NAMES = ["0: Fons Sa", "1: Necròtic", "2: Edema", "3: Actiu"]

os.makedirs("./test_multi", exist_ok=True)

# ==========================================
# 2. PIPELINE D'AVALUACIÓ MULTICLASSE + TTA
# ==========================================
def evaluate_multiclass_pipeline_tta():
    print(f"🔮 Iniciant auditoria Multiclasse amb TTA (ResNet34-UNet) a {DEVICE}...")
    
    _, _, test_ds = get_train_val_test_datasets(DATA_DIR, train_ratio=0.8, val_ratio=0.1, seed=42)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Inicialitzem el model exacte usat a l'entrenament
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=4, 
        classes=4
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    metrics_log = []
    y_true_pixels, y_pred_pixels = [], []
    sample_img, sample_target, sample_pred = None, None, None
    max_tumor_pixels = 0
    slice_counter = 0

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Processant llesques (TTA activat)"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).long()
            
            # 🔥 TEST-TIME AUGMENTATION (TTA) 🔥
            # 1. Original
            logits_orig = model(imgs)
            
            # 2. Volteig Horitzontal
            imgs_h = torch.flip(imgs, dims=[3])
            logits_h = torch.flip(model(imgs_h), dims=[3])
            
            # 3. Volteig Vertical
            imgs_v = torch.flip(imgs, dims=[2])
            logits_v = torch.flip(model(imgs_v), dims=[2])
            
            # 4. Volteig Horitzontal + Vertical
            imgs_hv = torch.flip(imgs, dims=[2, 3])
            logits_hv = torch.flip(model(imgs_hv), dims=[2, 3])
            
            # Promig de les 4 prediccions per obtenir un mapa de calor més robust
            logits_ensemble = (logits_orig + logits_h + logits_v + logits_hv) / 4.0
            
            preds = torch.argmax(logits_ensemble, dim=1) 
            
            # Recollida de píxels per a la Matriu de Confusió 4x4 (subsampling)
            y_true_pixels.extend(masks.cpu().numpy().flatten()[::50])
            y_pred_pixels.extend(preds.cpu().numpy().flatten()[::50])
            
            for b in range(imgs.shape[0]):
                p_slice = preds[b]
                t_slice = masks[b]
                
                # Només mesurem llesques on hi ha algun tipus de tumor
                if t_slice.sum() > 0:
                    slice_metrics = {"Slice_ID": slice_counter}
                    
                    # 1. CÀLCUL DEL SCORE GLOBAL (Whole Tumor)
                    p_wt = (p_slice > 0).long()
                    t_wt = (t_slice > 0).long()
                    
                    inter_wt = (p_wt.float() * t_wt.float()).sum().item()
                    union_wt = p_wt.sum().item() + t_wt.sum().item()
                    dice_wt = (2. * inter_wt + 1e-7) / (union_wt + 1e-7)
                    
                    slice_metrics["Dice_WT"] = dice_wt
                    
                    # 2. CÀLCUL PER SUBREGIONS (Classes 1, 2 i 3)
                    for c in range(1, 4):
                        p_c = (p_slice == c).long()
                        t_c = (t_slice == c).long()
                        
                        inter = (p_c.float() * t_c.float()).sum().item()
                        union = p_c.sum().item() + t_c.sum().item()
                        
                        dice_c = (2. * inter + 1e-7) / (union + 1e-7)
                        
                        if p_c.sum() == 0 or t_c.sum() == 0:
                            sens_c = 0.0 if t_c.sum() > 0 else np.nan
                        else:
                            sens_c = sensitivity(p_c.cpu().numpy(), t_c.cpu().numpy())
                            
                        slice_metrics[f"Dice_C{c}"] = dice_c
                        slice_metrics[f"Sens_C{c}"] = sens_c
                    
                    metrics_log.append(slice_metrics)
                    
                    # Guardar la mostra amb més tumor
                    tumor_size = t_wt.sum().item()
                    if tumor_size > max_tumor_pixels:
                        max_tumor_pixels = tumor_size
                        sample_img = imgs[b, 3].cpu().numpy()
                        sample_target = t_slice.cpu().numpy()
                        sample_pred = p_slice.cpu().numpy()
                
                slice_counter += 1

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(CSV_SAVE_PATH, index=False)
    
    # Càlcul de mitjanes finals per al títol
    mean_wt = df_metrics['Dice_WT'].mean()
    mean_c1 = df_metrics['Dice_C1'].mean()
    mean_c2 = df_metrics['Dice_C2'].mean()
    mean_c3 = df_metrics['Dice_C3'].mean()

    # ==========================================
    # 3. GENERACIÓ DEL DASHBOARD MULTICLASSE + GLOBAL
    # ==========================================
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(22, 15))
    
    titol_dashboard = (
        f"🏆 PUNTUACIÓ FINAL AMB TTA (DICE SCORE) 🏆\n"
        f"🌍 GLOBAL (WT): {mean_wt:.4f}  |  🔴 Necròtic (C1): {mean_c1:.4f}  |  🟢 Edema (C2): {mean_c2:.4f}  |  🟠 Actiu (C3): {mean_c3:.4f}"
    )
    fig.suptitle(titol_dashboard, fontsize=22, fontweight='bold', y=0.98, color='#333333')

    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true_pixels, y_pred_pixels, labels=[0, 1, 2, 3])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_percent, annot=True, fmt=".1%", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1, cbar=False, annot_kws={"size": 12})
    ax1.set_title("Matriu de Confusió 4x4 (Subregions) - TTA", fontweight="bold", fontsize=15, pad=12)

    ax2 = fig.add_subplot(gs[0, 1])
    df_dice = df_metrics[["Dice_C1", "Dice_C2", "Dice_C3", "Dice_WT"]].melt(var_name="Regió", value_name="Dice Score")
    df_dice["Regió"] = df_dice["Regió"].map({
        "Dice_C1": "1: Necròtic", 
        "Dice_C2": "2: Edema", 
        "Dice_C3": "3: Actiu",
        "Dice_WT": "Global (WT)"
    })
    
    palette_colors = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
    sns.boxplot(data=df_dice, x="Regió", y="Dice Score", palette=palette_colors, ax=ax2, width=0.5)
    sns.stripplot(data=df_dice, x="Regió", y="Dice Score", color=".25", size=2, alpha=0.15, ax=ax2, jitter=0.2)
    
    ax2.set_title("Distribució del Dice Score per Llesca", fontweight="bold", fontsize=15, pad=12)
    ax2.set_ylim(-0.05, 1.05)

    ax3 = fig.add_subplot(gs[1, :])
    sns.kdeplot(data=df_metrics["Dice_C1"].dropna(), label="Necròtic", fill=True, color="#d62728", alpha=0.3, ax=ax3)
    sns.kdeplot(data=df_metrics["Dice_C2"].dropna(), label="Edema", fill=True, color="#2ca02c", alpha=0.3, ax=ax3)
    sns.kdeplot(data=df_metrics["Dice_C3"].dropna(), label="Actiu", fill=True, color="#ff7f0e", alpha=0.3, ax=ax3)
    sns.kdeplot(data=df_metrics["Dice_WT"].dropna(), label="Global (Whole Tumor)", fill=False, color="black", linewidth=2.5, linestyle="--", ax=ax3)
    
    ax3.set_title("Densitat de Precisió per Tipus de Teixit i Global", fontweight="bold", fontsize=15, pad=12)
    ax3.set_xlim(0, 1)
    ax3.legend(fontsize=12)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88) 
    plt.savefig(DASHBOARD_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 4. IMATGE INDEPENDENT: COMPARATIVA VISUAL MULTICLASSE
    # ==========================================
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    cmap_multi = ListedColormap(['none', '#d62728', '#2ca02c', '#ff7f0e'])
    
    fig_brain, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig_brain.suptitle("Verificació de Diagnòstic Multiclasse amb TTA (ResNet34)", fontsize=20, fontweight="bold", y=1.02)

    target_viz = np.where(sample_target == 0, np.nan, sample_target)
    pred_viz = np.where(sample_pred == 0, np.nan, sample_pred)

    axes[0].imshow(sample_img, cmap='gray')
    axes[0].imshow(target_viz, cmap=cmap_multi, alpha=0.6, vmin=0, vmax=3)
    axes[0].set_title("Màscara Real de Subregions (GT)", fontsize=15, pad=12)
    axes[0].axis('off')

    axes[1].imshow(sample_img, cmap='gray')
    axes[1].imshow(pred_viz, cmap=cmap_multi, alpha=0.6, vmin=0, vmax=3)
    axes[1].set_title("Predicció ResNet34-UNet (Ensemble TTA)", fontsize=15, pad=12)
    axes[1].axis('off')
    
    patches = [
        mpatches.Patch(color='#d62728', label='Necròtic (C1)'),
        mpatches.Patch(color='#2ca02c', label='Edema (C2)'),
        mpatches.Patch(color='#ff7f0e', label='Tumor Actiu (C3)')
    ]
    axes[1].legend(handles=patches, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(BRAIN_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # 🔥 EL NÚMERO FINAL IMPRESO EN TERMINAL 🔥
    print("\n" + "="*50)
    print("🏆 PUNTUACIÓN FINAL DEL MODELO CON TTA (DICE SCORE) 🏆")
    print("="*50)
    print(f"🌍 GLOBAL (Whole Tumor) : {mean_wt:.4f}")
    print("-" * 50)
    print(f"🔴 Necrótico (C1)       : {mean_c1:.4f}")
    print(f"🟢 Edema (C2)           : {mean_c2:.4f}")
    print(f"🟠 Activo (C3)          : {mean_c3:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_multiclass_pipeline_tta()