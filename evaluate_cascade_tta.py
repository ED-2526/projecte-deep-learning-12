import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Imports del teu sistema
from dataset_mult import get_train_val_test_datasets
from train_multiclass_2 import MyUNet  
import segmentation_models_pytorch as smp # Per si el binari era un ResNet
from attention import AttentionUNet
import torch.nn.functional as F

try:
    from medpy.metric.binary import sensitivity
except ImportError:
    os.system('pip install medpy')
    from medpy.metric.binary import sensitivity

# ==========================================
# 1. CONFIGURACIÓ GLOBAL
# ==========================================
DATA_DIR = "/home/datasets/BraTS2020/data_processed"

# 🚀 LES DUES CLAUS DE L'ÈXIT: Posa aquí les rutes dels teus dos millors models
CHECKPOINT_BINARI = "./checkpoints/best_attention.pth"       # <-- Canvia pel teu nom real
CHECKPOINT_MULTI = "./checkpoints/prova_best_custom_unet.pth"   # <-- Canvia pel teu nom real

CSV_SAVE_PATH = "./test_multi/metrics_cascade.csv"
DASHBOARD_SAVE_PATH = "./test_multi/dashboard_cascade_tta.png"
BRAIN_SAVE_PATH = "./test_multi/cervell_comparativa_cascade_tta.png"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["0: Fons", "1: Necròtic", "2: Edema", "3: Actiu"]
GRAPH_LABELS = ["Necròtic", "Edema", "Actiu", "Whole Tumor"] 

os.makedirs("./test_multi", exist_ok=True)

def calculate_metrics_pipeline(pred, target):
    """Calcula Dice i Sensibilitat per a C1, C2, C3 i la unió Whole Tumor"""
    dice_scores = []; sens_scores = []
    
    # 1. Subregions (1, 2, 3)
    for c in range(1, 4):
        p_c = (pred == c)
        t_c = (target == c)
        
        if t_c.sum() == 0 and p_c.sum() == 0:
            dice_scores.append(1.0); sens_scores.append(1.0)
            continue
        elif t_c.sum() == 0 or p_c.sum() == 0:
            dice_scores.append(0.0); sens_scores.append(0.0)
            continue
            
        inter = (p_c.float() * t_c.float()).sum()
        dice = (2. * inter + 1e-7) / (p_c.sum() + t_c.sum() + 1e-7)
        sens = sensitivity(p_c.cpu().numpy(), t_c.cpu().numpy())
        dice_scores.append(dice.item()); sens_scores.append(sens)
        
    # 2. Whole Tumor (WT)
    p_wt = (pred > 0); t_wt = (target > 0)
    if t_wt.sum() == 0 and p_wt.sum() == 0:
        dice_scores.append(1.0); sens_scores.append(1.0)
    elif t_wt.sum() == 0 or p_wt.sum() == 0:
        dice_scores.append(0.0); sens_scores.append(0.0)
    else:
        inter_wt = (p_wt.float() * t_wt.float()).sum()
        dice_wt = (2. * inter_wt + 1e-7) / (p_wt.sum() + t_wt.sum() + 1e-7)
        sens_wt = sensitivity(p_wt.cpu().numpy(), t_wt.cpu().numpy())
        dice_scores.append(dice_wt.item()); sens_scores.append(sens_wt)
        
    return dice_scores, sens_scores

def evaluate_cascade_pipeline():
    print(f"🔮 Iniciant avaluació en CASCADA a {DEVICE}...")
    
    _, _, test_ds = get_train_val_test_datasets(DATA_DIR, train_ratio=0.8, val_ratio=0.1, seed=42)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ==========================================
    # CARREGAR MODEL 1: EL RASTREJADOR (BINARI)
    # ==========================================
    print("🧠 Carregant Model Especialista Binari...")
    # ATENCIÓ: Si el teu model binari el vas fer amb MyUNet(n_classes=1), canvia aquesta línia!
    model_binari = AttentionUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(DEVICE)
    model_binari.load_state_dict(torch.load(CHECKPOINT_BINARI, map_location=DEVICE))
    model_binari.eval()

    # ==========================================
    # CARREGAR MODEL 2: EL CLASSIFICADOR (MULTICLASSE)
    # ==========================================
    print("🧠 Carregant Model Detallista Multiclasse...")
    model_multi = MyUNet(n_channels=4, n_classes=4, dropout_rate=0.3).to(DEVICE)
    model_multi.load_state_dict(torch.load(CHECKPOINT_MULTI, map_location=DEVICE))
    model_multi.eval()
    
    metrics_log = []
    y_true_pixels, y_pred_pixels = [], []
    sample_img, sample_target, sample_pred = None, None, None
    max_tumor_pixels = 0
    slice_counter = 0

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Avaluant Cascada en Test"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).long()
            
            # --- PAS A: PREDICCIÓ BINÀRIA ---
            logits_bin = model_binari(imgs)
            probs_bin = torch.sigmoid(logits_bin)
            preds_bin = (probs_bin > 0.5).long().squeeze(1) # [Batch, H, W] (0 o 1)
            
            # --- PAS B: PREDICCIÓ MULTICLASSE AMB TTA ---
            # 1. Pasada normal
            logits_multi = model_multi(imgs)
            probs_multi_1 = F.softmax(logits_multi, dim=1)
            
            # 2. Pasada con la imagen volteada horizontalmente (eje W, que es -1)
            imgs_flipped = torch.flip(imgs, dims=[-1])
            logits_multi_flipped = model_multi(imgs_flipped)
            probs_multi_2_flipped = F.softmax(logits_multi_flipped, dim=1)
            
            # 3. Deshacemos el volteo a las predicciones para que encajen
            probs_multi_2 = torch.flip(probs_multi_2_flipped, dims=[-1])
            
            # 4. Promediamos y sacamos la clase definitiva
            probs_multi_avg = (probs_multi_1 + probs_multi_2) / 2.0
            preds_multi = torch.argmax(probs_multi_avg, dim=1).long() # [Batch, H, W]
            
            # --- PAS C: 🔥 LA FUSIÓ MÀGICA (CASCADA) 🔥 ---
            # Multipliquem. Si el binari diu 0 (Fons), qualsevol color que hagi posat el multi es torna 0.
            # Si el binari diu 1 (Tumor), el color del multi es respecta i es manté.
            # --- PAS C: 🔥 FUSIÓ EN CASCADA AMB DILATACIÓ (SOFT ENSEMBLING) 🔥 ---
            
            # 1. Donem el format correcte a la màscara [Batch, Canals, H, W]
            preds_bin_float = preds_bin.float().unsqueeze(1)
            
            # 2. INFLAR EL GLOBUS (Dilatació Morfològica a la GPU)
            # Un kernel_size=9 amb padding=4 inflarà el perímetre exacte 4 píxels cap enfora 
            # en totes les direccions. Aporta un "marge de seguretat" d'uns 2 mil·límetres.
            preds_bin_dilated = F.max_pool2d(preds_bin_float, kernel_size=9, stride=1, padding=4)
            
            # 3. Tornem al format original [Batch, H, W]
            preds_bin_dilated = preds_bin_dilated.squeeze(1).long()
            
            # 4. Multiplicació amb la màscara tolerant
            preds_final = preds_multi * preds_bin_dilated
            
            # A partir d'aquí, avaluem usant la nostra obra d'art fusionada (preds_final)
            y_true_pixels.extend(masks.cpu().numpy().flatten()[::50])
            y_pred_pixels.extend(preds_final.cpu().numpy().flatten()[::50])
            
            for b in range(imgs.shape[0]):
                pred_img = preds_final[b]
                mask_img = masks[b]
                
                if mask_img.sum() > 0:
                    dices, senses = calculate_metrics_pipeline(pred_img, mask_img)
                    metrics_log.append({
                        "Slice_ID": slice_counter,
                        "Dice_Necrotic": dices[0], "Dice_Edema": dices[1], "Dice_Actiu": dices[2], "Dice_WholeTumor": dices[3],
                        "Sens_Necrotic": senses[0], "Sens_Edema": senses[1], "Sens_Actiu": senses[2], "Sens_WholeTumor": senses[3],
                        "Macro_Dice": np.mean(dices[:3])
                    })
                    
                    tumor_size = mask_img.sum().item()
                    if tumor_size > max_tumor_pixels:
                        max_tumor_pixels = tumor_size
                        sample_img = imgs[b, 3].cpu().numpy()
                        sample_target = mask_img.cpu().numpy()
                        sample_pred = pred_img.cpu().numpy()
                
                slice_counter += 1

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(CSV_SAVE_PATH, index=False)
    print(f"💾 CSV llesca a llesca guardat.")
    
    # ==========================================
    # DASHBOARD I IMATGES (CODI RESUMIT PER ESPAI, ÉS EL MATEIX D'ABANS)
    # ==========================================
    print("🎨 Generant Dashboard Final de la Cascada...")
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2)

    # 1. Matriu
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true_pixels, y_pred_pixels, labels=[0, 1, 2, 3])
    sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt=".1%", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1, cbar=False)
    ax1.set_title("Matriu de Confusió (Model en Cascada)", fontweight="bold")

    # 2. Boxplot Dice
    ax2 = fig.add_subplot(gs[0, 1])
    dice_data = df_metrics[["Dice_Necrotic", "Dice_Edema", "Dice_Actiu", "Dice_WholeTumor"]]
    sns.boxplot(data=dice_data, palette="Set2", ax=ax2); sns.stripplot(data=dice_data, color=".25", size=1.5, alpha=0.2, ax=ax2)
    ax2.set_xticklabels(GRAPH_LABELS); ax2.set_title("Distribució del Dice Score", fontweight="bold"); ax2.set_ylim(-0.05, 1.05)

    # 3. Violin Sensibilitat
    ax3 = fig.add_subplot(gs[1, 0])
    sens_data = df_metrics[["Sens_Necrotic", "Sens_Edema", "Sens_Actiu", "Sens_WholeTumor"]]
    sns.violinplot(data=sens_data, palette="Pastel1", ax=ax3)
    ax3.set_xticklabels(GRAPH_LABELS); ax3.set_title("Sensibilitat / Recall", fontweight="bold"); ax3.set_ylim(-0.05, 1.05)

    # 4. Histograma
    ax4 = fig.add_subplot(gs[1, 1])
    sns.histplot(data=df_metrics, x="Macro_Dice", bins=30, kde=True, color="purple", ax=ax4)
    ax4.axvline(df_metrics["Macro_Dice"].mean(), color='red', linestyle='dashed', label=f'Mitjana: {df_metrics["Macro_Dice"].mean():.3f}')
    ax4.set_title("Histograma del Macro Dice Global", fontweight="bold"); ax4.legend()

    plt.tight_layout(); plt.savefig(DASHBOARD_SAVE_PATH, dpi=300); plt.close()
    
    # 5. Cervell Comparatiu
    from matplotlib.colors import ListedColormap; from matplotlib.patches import Patch
    cmap = ListedColormap(['black', 'red', 'green', 'blue'])
    fig_brain, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig_brain.suptitle("Diagnòstic en Cascada (Binari + Multiclasse)", fontsize=20, fontweight="bold", y=1.02)

    axes[0].imshow(sample_img, cmap='gray'); axes[0].imshow(sample_target, cmap=cmap, alpha=0.5, vmin=0, vmax=3); axes[0].set_title("Veritat Terreny", fontsize=16); axes[0].axis('off')
    axes[1].imshow(sample_img, cmap='gray'); axes[1].imshow(sample_pred, cmap=cmap, alpha=0.5, vmin=0, vmax=3); axes[1].set_title("Predicció CASCADA", fontsize=16); axes[1].axis('off')
    
    axes[1].legend(handles=[Patch(facecolor='red', label='Necròtic'), Patch(facecolor='green', label='Edema'), Patch(facecolor='blue', label='Actiu')], loc='upper right', fontsize=12)
    plt.tight_layout(); plt.savefig(BRAIN_SAVE_PATH, dpi=300); plt.close()
    
    print(f"🏁 Procés finalitzat! Obre {DASHBOARD_SAVE_PATH} per veure l'espectacle.")

if __name__ == "__main__":
    evaluate_cascade_pipeline()