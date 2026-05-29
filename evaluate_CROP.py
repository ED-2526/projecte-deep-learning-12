import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Imports del teu sistema
from dataset_mult import get_train_val_test_datasets

try:
    from medpy.metric.binary import sensitivity
except ImportError:
    os.system('pip install medpy')
    from medpy.metric.binary import sensitivity

# 🚀 IMPORTACIÓ DE LES TEVES DUES ARQUITECTURES D'ATENCIÓ
try:
    from attention import AttentionUNet 
    from train_multiclass_crop import AttentionUNetMulticlass 
except ImportError:
    from attention import AttentionUNet
    AttentionUNetMulticlass = AttentionUNet 

# ==========================================
# 1. CONFIGURACIÓ GLOBAL
# ==========================================
DATA_DIR = "/home/datasets/BraTS2020/data_processed"

CHECKPOINT_BINARI = "./checkpoints/best_attention.pth"
CHECKPOINT_CROP = "./checkpoints/best_multiclass_CROP_realistic.pth"

CSV_SAVE_PATH = "./test_multi/metrics_cascade_crop_advanced.csv"
DASHBOARD_SAVE_PATH = "./test_multi/dashboard_cascade_crop_advanced.png"
BRAIN_SAVE_PATH = "./test_multi/cervell_comparativa_cascade_crop_advanced.png"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["0: Fons", "1: Necròtic", "2: Edema", "3: Actiu"]

os.makedirs("./test_multi", exist_ok=True)

# ==========================================
# 2. FUNCIONS GEOMÈTRIQUES I DE MÈTRIQUES
# ==========================================
def extract_bbox_coor(pred_binary_mask, margin=16, orig_shape=(240, 240)):
    """Troba les coordenades de la Bounding Box de la predicció binària"""
    coords = torch.nonzero(pred_binary_mask > 0)
    if len(coords) == 0:
        return None
        
    y_min, y_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    x_min, x_max = coords[:, 1].min().item(), coords[:, 1].max().item()
    
    y_min = max(0, y_min - margin)
    y_max = min(orig_shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(orig_shape[1], x_max + margin)
    
    return y_min, y_max, x_min, x_max

def calculate_metrics_pipeline(pred, target):
    """Calcula Dice i Sensibilitat de forma robusta"""
    dice_scores = []
    sens_scores = []
    
    # Mètriques de subregions pures (1, 2, 3)
    for c in range(1, 4):
        p_c = (pred == c).long()
        t_c = (target == c).long()
        
        inter = (p_c.float() * t_c.float()).sum().item()
        union = p_c.sum().item() + t_c.sum().item()
        
        dice = (2. * inter + 1e-7) / (union + 1e-7)
        
        if p_c.sum() == 0 or t_c.sum() == 0:
            sens = 0.0 if t_c.sum() > 0 else np.nan
        else:
            sens = sensitivity(p_c.cpu().numpy(), t_c.cpu().numpy())
            
        dice_scores.append(dice)
        sens_scores.append(sens)
        
    # Mètrica unificada Whole Tumor (WT)
    p_wt = (pred > 0).long()
    t_wt = (target > 0).long()
    
    inter_wt = (p_wt.float() * t_wt.float()).sum().item()
    union_wt = p_wt.sum().item() + t_wt.sum().item()
    dice_wt = (2. * inter_wt + 1e-7) / (union_wt + 1e-7)
    
    if p_wt.sum() == 0 or t_wt.sum() == 0:
        sens_wt = 0.0 if t_wt.sum() > 0 else np.nan
    else:
        sens_wt = sensitivity(p_wt.cpu().numpy(), t_wt.cpu().numpy())
        
    dice_scores.append(dice_wt)
    sens_scores.append(sens_wt)
        
    return dice_scores, sens_scores

# ==========================================
# 3. PIPELINE D'AVALUACIÓ PRINCIPAL
# ==========================================
def evaluate_cascade_crop():
    print(f"🔮 Iniciant pipeline en CASCADA + CROP a {DEVICE}...")
    
    _, _, test_ds = get_train_val_test_datasets(DATA_DIR, train_ratio=0.8, val_ratio=0.1, seed=42)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("🧠 Carregant Model Rastrejador Binari...")
    model_binari = AttentionUNet(n_channels=4, n_classes=1).to(DEVICE)
    model_binari.load_state_dict(torch.load(CHECKPOINT_BINARI, map_location=DEVICE, weights_only=False), strict=False)
    model_binari.eval()

    print("🧠 Carregant Model Especialista Crop Multiclasse...")
    model_crop_multi = AttentionUNetMulticlass(n_channels=4, n_classes=4).to(DEVICE)
    model_crop_multi.load_state_dict(torch.load(CHECKPOINT_CROP, map_location=DEVICE, weights_only=False), strict=False)
    model_crop_multi.eval()
    
    metrics_log = []
    y_true_pixels, y_pred_pixels = [], []
    sample_img, sample_target, sample_pred = None, None, None
    max_tumor_pixels = 0
    slice_counter = 0

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Processant cascada complexa"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).long()
            
            logits_bin = model_binari(imgs)
            preds_bin = (torch.sigmoid(logits_bin) > 0.5).long().squeeze(1) 
            
            for b in range(imgs.shape[0]):
                orig_img_slice = imgs[b]       
                orig_mask_slice = masks[b]     
                bin_pred_slice = preds_bin[b]  
                
                reconstructed_pred = torch.zeros_like(orig_mask_slice).to(DEVICE)
                bbox = extract_bbox_coor(bin_pred_slice, margin=16)
                
                if bbox is not None:
                    y_min, y_max, x_min, x_max = bbox
                    crop_img = orig_img_slice[:, y_min:y_max, x_min:x_max]
                    h_crop, w_crop = crop_img.shape[1], crop_img.shape[2]
                    
                    if h_crop > 4 and w_crop > 4: 
                        crop_img_input = F.interpolate(crop_img.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
                        logits_crop = model_crop_multi(crop_img_input)
                        pred_crop = torch.argmax(logits_crop, dim=1).squeeze(0) 
                        pred_crop_rescaled = F.interpolate(pred_crop.unsqueeze(0).unsqueeze(0).float(), 
                                                           size=(h_crop, w_crop), mode='nearest')[0, 0].long()
                        reconstructed_pred[y_min:y_max, x_min:x_max] = pred_crop_rescaled
                
                if orig_mask_slice.sum() > 0:
                    dices, senses = calculate_metrics_pipeline(reconstructed_pred, orig_mask_slice)
                    metrics_log.append({
                        "Slice_ID": slice_counter,
                        "Dice_C1": dices[0], "Dice_C2": dices[1], "Dice_C3": dices[2], "Dice_WT": dices[3],
                        "Sens_C1": senses[0], "Sens_C2": senses[1], "Sens_C3": senses[2], "Sens_WT": senses[3],
                        "Macro_Dice": np.mean(dices[:3])
                    })
                    
                    tumor_size = orig_mask_slice.sum().item()
                    if tumor_size > max_tumor_pixels:
                        max_tumor_pixels = tumor_size
                        sample_img = orig_img_slice[3].cpu().numpy() 
                        sample_target = orig_mask_slice.cpu().numpy()
                        sample_pred = reconstructed_pred.cpu().numpy()
                        
                y_true_pixels.extend(orig_mask_slice.cpu().numpy().flatten()[::50])
                y_pred_pixels.extend(reconstructed_pred.cpu().numpy().flatten()[::50])
                slice_counter += 1

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(CSV_SAVE_PATH, index=False)
    
    mean_wt = df_metrics['Dice_WT'].mean()
    mean_c1 = df_metrics['Dice_C1'].mean()
    mean_c2 = df_metrics['Dice_C2'].mean()
    mean_c3 = df_metrics['Dice_C3'].mean()

    # ==========================================
    # 4. GENERACIÓ DEL DASHBOARD ANALÍTIC AVANÇAT
    # ==========================================
    print("🎨 Generant Dashboard de control d'alta qualitat...")
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(22, 15))
    
    titol_dashboard = (
        f"🏆 PUNTUACIÓ FINAL (CASCADE + CROP) 🏆\n"
        f"🌍 GLOBAL (WT): {mean_wt:.4f}  |  🔴 Necròtic (C1): {mean_c1:.4f}  |  🟢 Edema (C2): {mean_c2:.4f}  |  🟠 Actiu (C3): {mean_c3:.4f}"
    )
    fig.suptitle(titol_dashboard, fontsize=22, fontweight='bold', y=0.98, color='#333333')

    gs = fig.add_gridspec(2, 2)

    # 1. Matriu de Confusió
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true_pixels, y_pred_pixels, labels=[0, 1, 2, 3])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_percent, annot=True, fmt=".1%", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1, cbar=False, annot_kws={"size": 12})
    ax1.set_title("Matriu de Confusió 4x4 (Subregions)", fontweight="bold", fontsize=15, pad=12)

    # 2. Boxplot (Utilitzant Melt i format Llarg)
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

    # 3. Gràfic de Densitat KDE (Elegant)
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
    # 5. IMATGE INDEPENDENT: COMPARATIVA VISUAL MULTICLASSE
    # ==========================================
    print("🧠 Dibuixant control visual de llesques...")
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    cmap_multi = ListedColormap(['none', '#d62728', '#2ca02c', '#ff7f0e'])
    
    fig_brain, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig_brain.suptitle("Verificació del Pipeline en Cascada + Crop (Espai Final 240x240)", fontsize=20, fontweight="bold", y=1.02)

    target_viz = np.where(sample_target == 0, np.nan, sample_target)
    pred_viz = np.where(sample_pred == 0, np.nan, sample_pred)

    axes[0].imshow(sample_img, cmap='gray')
    axes[0].imshow(target_viz, cmap=cmap_multi, alpha=0.6, vmin=0, vmax=3)
    axes[0].set_title("Màscara Real de Subregions (GT)", fontsize=15, pad=12)
    axes[0].axis('off')

    axes[1].imshow(sample_img, cmap='gray')
    axes[1].imshow(pred_viz, cmap=cmap_multi, alpha=0.6, vmin=0, vmax=3)
    axes[1].set_title("Predicció Reconstruïda (Cascade + Crop)", fontsize=15, pad=12)
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

    # 🔥 EL NÚMERO FINAL IMPRÈS EN TERMINAL 🔥
    print("\n" + "="*50)
    print("🏆 PUNTUACIÓ FINAL (CASCADE + CROP) 🏆")
    print("="*50)
    print(f"🌍 GLOBAL (Whole Tumor) : {mean_wt:.4f}")
    print("-" * 50)
    print(f"🔴 Necròtic (C1)       : {mean_c1:.4f}")
    print(f"🟢 Edema (C2)          : {mean_c2:.4f}")
    print(f"🟠 Actiu (C3)          : {mean_c3:.4f}")
    print("="*50 + "\n")
    print(f"🖼️ Resultats guardats a la carpeta test_multi/\n🏁 Pipeline completat amb èxit!")

if __name__ == "__main__":
    evaluate_cascade_crop()