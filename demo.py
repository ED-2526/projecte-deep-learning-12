import os
import random
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np

from dataset import get_train_val_datasets

def show_prediction():
    # Rutes de la teva màquina
    DATA_DIR = "/home/edxnG12/data_processed_12"
    MODEL_PATH = "./checkpoints/unet_resnet34_best.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carregar el model (ResNet34 com l'entrenament)
    model = smp.Unet(encoder_name="resnet34", in_channels=4, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Agafar un pacient a l'atzar de validació
    _, val_dataset = get_train_val_datasets(DATA_DIR)
    idx = random.randint(0, len(val_dataset) - 1)
    image, true_mask = val_dataset[idx]
    
    # 3. Predicció
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        prediction = (probs > 0.5).float().cpu().numpy()[0, 0]

    # 4. Preparar visualització
    img_flair = image[3].numpy() 
    img_t1ce = image[1].numpy()
    truth = true_mask[0].numpy()

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title(f"FLAIR (Original)\nIdx: {idx}")
    plt.imshow(img_flair, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("T1ce (Contrast)")
    plt.imshow(img_t1ce, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Metge (Ground Truth)")
    plt.imshow(img_flair, cmap='gray')
    plt.imshow(truth, cmap='Reds', alpha=0.5) # Tumor real en vermell
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("IA (Predicció)")
    plt.imshow(img_flair, cmap='gray')
    plt.imshow(prediction, cmap='Greens', alpha=0.5) # Tumor IA en verd
    plt.axis('off')

    plt.tight_layout()

    # Guardem la imatge per poder-la descarregar
    OUTPUT_PATH = "/home/edxnG12/Grafics/demo_prediccio.png"
    plt.savefig(OUTPUT_PATH)
    print(f"✅ Predicció guardada com a '{OUTPUT_PATH}' (Idx del cas: {idx})")
    plt.show()

if __name__ == "__main__":
    show_prediction()