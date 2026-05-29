import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_myunet import MyUNet
from attention import AttentionUNet

# Intentem importar les mètriques clíniques de medpy
try:
    from medpy.metric.binary import hd95, sensitivity, specificity, dc
except ImportError:
    print("⚠️ medpy no està instal·lat. Instal·lant...")
    os.system('pip install medpy')
    from medpy.metric.binary import hd95, sensitivity, specificity, dc

# 🔥 CANVI: Importem la funció que divideix en Train, Validation i Test
from dataset import get_train_test_validation

def calculate_clinical_metrics(pred, truth):
    """Calcula mètriques i també extreu els píxels absoluts per a la matriu de confusió"""
    
    # 1. Càlcul de píxels
    tp = np.logical_and(pred == 1, truth == 1).sum()
    fp = np.logical_and(pred == 1, truth == 0).sum()
    fn = np.logical_and(pred == 0, truth == 1).sum()
    tn = np.logical_and(pred == 0, truth == 0).sum()

    # 2. Casos buits (evitar divisions per zero)
    if truth.sum() == 0 and pred.sum() == 0:
        return 1.0, 0.0, 1.0, 1.0, tp, fp, fn, tn 
    
    if truth.sum() == 0 or pred.sum() == 0:
        return 0.0, 100.0, 0.0, 0.0, tp, fp, fn, tn 
    
    # 3. Càlcul normal de mètriques
    dice = dc(pred, truth)
    sens = sensitivity(pred, truth)
    spec = specificity(pred, truth)
    try:
        hausdorff = hd95(pred, truth)
    except:
        hausdorff = 100.0 
        
    return dice, hausdorff, sens, spec, tp, fp, fn, tn

def main():
    # 1. Rutes
    DATA_DIR = "/home/datasets/BraTS2020/data_processed_12"
    MODEL_PATH = "./checkpoints/prova_best_custom_tversky.pth"
    WORST_CASES_DIR = "./Grafics_Finals/hola"
    os.makedirs(WORST_CASES_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🩺 Iniciant Avaluació Clínica (TEST SET) a: {device}")
    
    # 2. Carregar Model
    model = MyUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    #model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=1).to(device)
    #model = AttentionUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model carregat correctament.")
    else:
        print(f"❌ Error: No s'ha trobat el model a {MODEL_PATH}")
        return
        
    model.eval()

    # 3. Carregar Dades (TEST SET)
    _, _, test_dataset = get_train_test_validation(DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    results = []
    
    # Comptadors de PÍXELS (Volum geomètric)
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    # Comptadors de LLESQUES (Capacitat de triatge/detecció)
    slice_tp, slice_fp, slice_fn, slice_tn = 0, 0, 0, 0

    # 4. Avaluar cada llesca del Test
    with torch.no_grad():
        for i, (image, true_mask) in enumerate(tqdm(test_loader, desc="Analitzant pacients (Test)")):
            img_tensor = image.to(device)
            
                        # --- 🚀 INICI DEL TEST-TIME AUGMENTATION (TTA) 🚀 ---
            
            # 1. Predicció Original
            logits_orig = model(img_tensor)
            probs_orig = torch.sigmoid(logits_orig)
            
            # 2. Predicció Horizontal (Mirall)
            img_hflip = torch.flip(img_tensor, dims=[3]) # Girem l'amplada
            logits_hflip = model(img_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)
            probs_hflip_back = torch.flip(probs_hflip, dims=[3]) # Tornem a girar la resposta
            
            # 3. Predicció Vertical (Cap per avall)
            img_vflip = torch.flip(img_tensor, dims=[2]) # Girem l'alçada
            logits_vflip = model(img_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)
            probs_vflip_back = torch.flip(probs_vflip, dims=[2]) # Tornem a girar la resposta
            
            # 4. ENSAMBLATGE: Fem la mitjana de les 3 opinions
            probs_mean = (probs_orig + probs_hflip_back + probs_vflip_back) / 3.0
            
            # --- FINAL DEL TTA ---
            pred_mask = (probs_mean > 0.5).cpu().numpy().astype(bool)[0, 0]
            truth_mask = true_mask.cpu().numpy().astype(bool)[0, 0]
            
            # Recollim mètriques i píxels
            d, hd, s, sp, tp, fp, fn, tn = calculate_clinical_metrics(pred_mask, truth_mask)
            
            # Acumulem píxels globals
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
            # ==========================================
            # Avaluació Llesca a Llesca (Presència de tumor)
            # ==========================================
            has_tumor_true = truth_mask.sum() > 0
            has_tumor_pred = pred_mask.sum() > 0
            
            if has_tumor_true and has_tumor_pred:
                slice_tp += 1
            elif not has_tumor_true and has_tumor_pred:
                slice_fp += 1
            elif has_tumor_true and not has_tumor_pred:
                slice_fn += 1
            else:
                slice_tn += 1
            # ==========================================
            
            results.append({
                "index": i,
                "dice": d,
                "hd95": hd,
                "sens": s,
                "spec": sp,
                "image": image[0].numpy(), 
                "pred": pred_mask,
                "truth": truth_mask
            })

    # 5. Resultats Finals
    avg_dice = np.mean([r["dice"] for r in results])
    avg_hd95 = np.mean([r["hd95"] for r in results if r["hd95"] < 100])
    avg_sens = np.mean([r["sens"] for r in results])
    avg_spec = np.mean([r["spec"] for r in results])
    
    print("\n" + "="*50)
    print(" 🏥 RESULTATS CLÍNICS FINALS (TEST) 🏥 ")
    print("="*50)
    print(f"  Dice Score Mitjà : {avg_dice:.4f}")
    print(f"  Hausdorff (HD95) : {avg_hd95:.2f} px")
    print(f"  Sensibilitat     : {avg_sens:.4f}")
    print(f"  Especificitat    : {avg_spec:.4f}")
    print("="*50)

    # ---------------------------------------------------------
    # BLOC VISUAL (Gràfics per a la memòria del TFG)
    # ---------------------------------------------------------
    print("\n🎨 Generant gràfics analítics...")

    # 6.1 Histograma
    dice_scores = [r["dice"] for r in results if r["truth"].sum() > 0]
    plt.figure(figsize=(10, 5))
    plt.hist(dice_scores, bins=40, color='skyblue', edgecolor='black')
    plt.axvline(avg_dice, color='red', linestyle='dashed', label=f'Mitjana: {avg_dice:.2f}')
    plt.title("Distribució del Dice Score (Test Set - Casos amb tumor)")
    plt.xlabel("Dice Score")
    plt.ylabel("Freqüència")
    plt.legend()
    plt.savefig(os.path.join(WORST_CASES_DIR, "histograma.png"))
    plt.close()

    # 6.2A Matriu de Confusió Global (PÍXELS)
    cm_px = np.array([[total_tn, total_fp], 
                      [total_fn, total_tp]])
    cm_px_perc = cm_px / np.sum(cm_px) * 100

    plt.figure(figsize=(8, 6))
    labels_px = np.array([[(f"True Negatives\n{total_tn:,}\n({cm_px_perc[0,0]:.2f}%)"), 
                           (f"False Positives\n{total_fp:,}\n({cm_px_perc[0,1]:.4f}%)")],
                          [(f"False Negatives\n{total_fn:,}\n({cm_px_perc[1,0]:.4f}%)"), 
                           (f"True Positives\n{total_tp:,}\n({cm_px_perc[1,1]:.2f}%)")]])

    sns.heatmap(cm_px, annot=labels_px, fmt="", cmap="Blues", cbar=False, 
                xticklabels=["Negatiu (Sa)", "Positiu (Tumor)"], 
                yticklabels=["Negatiu (Sa)", "Positiu (Tumor)"])
    plt.title('Matriu de Confusió en Test (Píxel a Píxel)', fontsize=14, pad=20)
    plt.xlabel('Predicció de la IA', fontsize=12)
    plt.ylabel('Realitat (Metge)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(WORST_CASES_DIR, "matriu_confusio_pixels.png"), dpi=300)
    plt.close()

    # 6.2B Matriu de Confusió Global (LLESQUES/IMATGES)
    cm_slice = np.array([[slice_tn, slice_fp], 
                         [slice_fn, slice_tp]])
    cm_slice_perc = cm_slice / np.sum(cm_slice) * 100

    plt.figure(figsize=(8, 6))
    labels_slice = np.array([[(f"True Negatives\n{slice_tn:,}\n({cm_slice_perc[0,0]:.1f}%)"), 
                              (f"False Positives\n{slice_fp:,}\n({cm_slice_perc[0,1]:.1f}%)")],
                             [(f"False Negatives\n{slice_fn:,}\n({cm_slice_perc[1,0]:.1f}%)"), 
                              (f"True Positives\n{slice_tp:,}\n({cm_slice_perc[1,1]:.1f}%)")]])

    sns.heatmap(cm_slice, annot=labels_slice, fmt="", cmap="Greens", cbar=False, 
                xticklabels=["Llesca Sana", "Llesca amb Tumor"], 
                yticklabels=["Llesca Sana", "Llesca amb Tumor"])
    plt.title('Matriu de Confusió en Test (Llesca a Llesca)', fontsize=14, pad=20)
    plt.xlabel('Predicció de la IA', fontsize=12)
    plt.ylabel('Realitat (Metge)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(WORST_CASES_DIR, "matriu_confusio_llesques.png"), dpi=300)
    plt.close()
        
    print(f"✅ Execució finalitzada! Totes les analítiques del Test estan a la carpeta: {WORST_CASES_DIR}")

if __name__ == "__main__":
    main()