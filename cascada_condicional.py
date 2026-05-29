import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
from train_myunet import MyUNet 
from dataset import get_train_test_validation
from attention import AttentionUNet

try:
    from medpy.metric.binary import hd95, sensitivity, specificity, dc
except ImportError:
    os.system('pip install medpy')
    from medpy.metric.binary import hd95, sensitivity, specificity, dc

# 🔥 NOU: Funció recuperada amb el càlcul de píxels
def calculate_clinical_metrics(pred, truth):
    # 1. Càlcul de píxels
    tp = np.logical_and(pred == 1, truth == 1).sum()
    fp = np.logical_and(pred == 1, truth == 0).sum()
    fn = np.logical_and(pred == 0, truth == 1).sum()
    tn = np.logical_and(pred == 0, truth == 0).sum()

    if truth.sum() == 0 and pred.sum() == 0: 
        return 1.0, 0.0, 1.0, 1.0, tp, fp, fn, tn 
    if truth.sum() == 0 or pred.sum() == 0: 
        return 0.0, 100.0, 0.0, 0.0, tp, fp, fn, tn 
    
    dice = dc(pred, truth)
    sens = sensitivity(pred, truth)
    spec = specificity(pred, truth)
    try: 
        hausdorff = hd95(pred, truth)
    except: 
        hausdorff = 100.0 
        
    return dice, hausdorff, sens, spec, tp, fp, fn, tn

def predict_with_tta(model, img_tensor):
    logits_orig = model(img_tensor)
    probs_orig = torch.sigmoid(logits_orig)
    img_hflip = torch.flip(img_tensor, dims=[3])
    probs_hflip_back = torch.flip(torch.sigmoid(model(img_hflip)), dims=[3])
    img_vflip = torch.flip(img_tensor, dims=[2])
    probs_vflip_back = torch.flip(torch.sigmoid(model(img_vflip)), dims=[2])
    return (probs_orig + probs_hflip_back + probs_vflip_back) / 3.0

def main():
    DATA_DIR = "/home/datasets/BraTS2020/data_processed_12/"
    MODEL_BASE_PATH = "./checkpoints/best_attention_focaldicee.pth" 
    MODEL_ESPECIALISTA_PATH = "./checkpoints/best_focal_oversampling.pth"
    
    # --- PARÀMETRE CLAU DE L'EXPERIMENT ---
    MIN_PIXELS = 50 
    # --------------------------------------

    OUTPUT_DIR = "./estudi_triatge_amb_filtree"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🩺 Iniciant Cascada Condicional amb FILTRE DE MIDA ({MIN_PIXELS} px) a: {device}")
    
    model_base = AttentionUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    model_base.load_state_dict(torch.load(MODEL_BASE_PATH, map_location=device))
    model_base.eval()

    model_especialista = MyUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    model_especialista.load_state_dict(torch.load(MODEL_ESPECIALISTA_PATH, map_location=device))
    model_especialista.eval()

    _, _, test_dataset = get_train_test_validation(DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    results_base, results_combinat = [], []
    rescats_exitosos, falsos_positius, rescats_descartats_per_mida = 0, 0, 0
    
    # Comptadors de llesques i píxels
    slice_tp, slice_fp, slice_tn, slice_fn = 0, 0, 0, 0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0 # 🔥 NOU

    with torch.no_grad():
        for i, (image, true_mask) in enumerate(tqdm(test_loader, desc="Avaluant Cascada")):
            img_tensor = image.to(device)
            truth_mask = true_mask.cpu().numpy().astype(bool)[0, 0]
            
            # 1. Base (Sense TTA actiu en aquest codi per defecte)
            probs_base = torch.sigmoid(model_base(img_tensor))
            pred_base = (probs_base > 0.5).cpu().numpy().astype(bool)[0, 0]
            
            # Recollim només el Dice del model Base per comparar-lo al final
            d_base, _, _, _, _, _, _, _ = calculate_clinical_metrics(pred_base, truth_mask)
            results_base.append({"dice": d_base})
            
            # 2. Lògica de Rescat (Cascada Condicional)
            pred_final = pred_base
            if pred_base.sum() == 0:
                # probs_esp = predict_with_tta(model_especialista, img_tensor)
                probs_esp = torch.sigmoid(model_especialista(img_tensor))
                pred_esp = (probs_esp > 0.5).cpu().numpy().astype(bool)[0, 0]
                
                # --- APLIQUEM EL FILTRE MORFOLÒGIC ---
                num_pixels = pred_esp.sum()
                if num_pixels >= MIN_PIXELS:
                    pred_final = pred_esp
                    if truth_mask.sum() > 0: rescats_exitosos += 1
                    else: falsos_positius += 1
                elif num_pixels > 0:
                    rescats_descartats_per_mida += 1
                    # Es queda amb el pred_base (que és 0)
            
            # 🔥 NOU: Obtenim TOTES les mètriques i píxels de la predicció final combinada
            d_final, hd_final, s_final, sp_final, p_tp, p_fp, p_fn, p_tn = calculate_clinical_metrics(pred_final, truth_mask)
            
            # Guardem les mètriques clíniques completes de l'arquitectura combinada
            results_combinat.append({
                "dice": d_final,
                "hd95": hd_final,
                "sens": s_final,
                "spec": sp_final
            })

            # Sumem píxels globals
            total_tp += p_tp
            total_fp += p_fp
            total_fn += p_fn
            total_tn += p_tn

            # Matriu de Llesques
            has_tumor_real = truth_mask.sum() > 0
            has_tumor_pred = pred_final.sum() > 0
            
            if has_tumor_real and has_tumor_pred:       slice_tp += 1
            elif not has_tumor_real and has_tumor_pred: slice_fp += 1
            elif has_tumor_real and not has_tumor_pred: slice_fn += 1
            else:                                       slice_tn += 1

    # Càlcul de Mitjanes Globals
    avg_dice_base = np.mean([r["dice"] for r in results_base])
    avg_dice_comb = np.mean([r["dice"] for r in results_combinat])
    
    # 🔥 NOU: Mètriques extra
    avg_hd95_comb = np.mean([r["hd95"] for r in results_combinat if r["hd95"] < 100])
    avg_sens_comb = np.mean([r["sens"] for r in results_combinat])
    avg_spec_comb = np.mean([r["spec"] for r in results_combinat])

    # ==========================================
    # RESUM DE CONSOLA
    # ==========================================
    print("\n" + "="*60)
    print(f" 🔍 FILTRE APLICAT: > {MIN_PIXELS} píxels")
    print(f" 📉 Rescats descartats per ser massa petits: {rescats_descartats_per_mida}")
    print(f" ✅ Rescats vàlids aconseguits: {rescats_exitosos}")
    print(f" ❌ Falses alarmes restants: {falsos_positius}")
    print("-" * 60)
    print(" 🏥 MÈTRIQUES DEL SISTEMA EN CASCADA 🏥 ")
    print(f" 🏆 DICE Base       : {avg_dice_base:.4f}")
    print(f" 🏆 DICE Cascada    : {avg_dice_comb:.4f}")
    print(f"  Hausdorff (HD95) : {avg_hd95_comb:.2f} px")
    print(f"  Sensibilitat     : {avg_sens_comb:.4f}")
    print(f"  Especificitat    : {avg_spec_comb:.4f}")
    print("="*60)

    # ==========================================
    # BLOC VISUAL
    # ==========================================
    print("\n🎨 Generant gràfics analítics...")

    # --- 1. HISTOGRAMA ---
    dice_base_tot = [r["dice"] for r in results_base]
    dice_comb_tot = [r["dice"] for r in results_combinat]
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(dice_base_tot, bins=40, color='skyblue', edgecolor='black')
    plt.axvline(avg_dice_base, color='red', linestyle='dashed', label=f'Mitjana Global: {avg_dice_base:.2f}')
    plt.title('Abans (Model Base Únic)')
    plt.xlabel('Dice Score')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(dice_comb_tot, bins=40, color='lightgreen', edgecolor='black')
    plt.axvline(avg_dice_comb, color='red', linestyle='dashed', label=f'Mitjana Global: {avg_dice_comb:.2f}')
    plt.title('Després (Cascada Condicional)')
    plt.xlabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograma_cascada.png"))
    plt.close()

    # --- 🔥 NOU: 2. MATRIU DE CONFUSIÓ (PÍXELS) ---
    cm_px = np.array([[total_tn, total_fp], 
                      [total_fn, total_tp]])
    cm_px_perc = cm_px / (np.sum(cm_px) + 1e-8) * 100

    plt.figure(figsize=(8, 6))
    labels_px = np.array([[(f"True Negatives\n{total_tn:,}\n({cm_px_perc[0,0]:.2f}%)"), 
                           (f"False Positives\n{total_fp:,}\n({cm_px_perc[0,1]:.4f}%)")],
                          [(f"False Negatives\n{total_fn:,}\n({cm_px_perc[1,0]:.4f}%)"), 
                           (f"True Positives\n{total_tp:,}\n({cm_px_perc[1,1]:.2f}%)")]])

    sns.heatmap(cm_px, annot=labels_px, fmt="", cmap="Blues", cbar=False, 
                xticklabels=["Negatiu (Sa)", "Positiu (Tumor)"], 
                yticklabels=["Negatiu (Sa)", "Positiu (Tumor)"])
    plt.title('Matriu de Confusió Cascada (Píxel a Píxel)', fontsize=14, pad=20)
    plt.xlabel('Predicció del Sistema Final', fontsize=12)
    plt.ylabel('Realitat (Metge)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriu_pixels_cascada.png"), dpi=300)
    plt.close()

    # --- 3. MATRIU DE CONFUSIÓ (LLESQUES) ---
    cm_slice = np.array([[slice_tn, slice_fp], 
                         [slice_fn, slice_tp]])
    cm_slice_perc = cm_slice / (np.sum(cm_slice) + 1e-8) * 100

    plt.figure(figsize=(8, 6))
    labels_slice = np.array([[(f"True Negatives\n{slice_tn:,}\n({cm_slice_perc[0,0]:.1f}%)"), 
                              (f"False Positives\n{slice_fp:,}\n({cm_slice_perc[0,1]:.1f}%)")],
                             [(f"False Negatives\n{slice_fn:,}\n({cm_slice_perc[1,0]:.1f}%)"), 
                              (f"True Positives\n{slice_tp:,}\n({cm_slice_perc[1,1]:.1f}%)")]])

    sns.heatmap(cm_slice, annot=labels_slice, fmt="", cmap="Greens", cbar=False, 
                xticklabels=["Llesca Sana", "Llesca amb Tumor"], 
                yticklabels=["Llesca Sana", "Llesca amb Tumor"])
    plt.title(f'Matriu de Confusió Cascada (Filtre {MIN_PIXELS}px)', fontsize=14, pad=20)
    plt.xlabel('Predicció del Sistema Final', fontsize=12)
    plt.ylabel('Realitat (Metge)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriu_llesques_cascada.png"), dpi=300)
    plt.close()

    print(f"✅ Procés acabat! S'han desat els gràfics a la carpeta: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()