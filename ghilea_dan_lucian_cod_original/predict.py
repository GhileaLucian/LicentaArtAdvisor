# predict_licenta.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import torchvision.models.vision_transformer as vit
import tkinter as tk
from tkinter import filedialog

# --- Pasul 1: Recreăm arhitectura modelului (identică cu cea de la antrenament) ---
class EmotionEnsemble(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(EmotionEnsemble, self).__init__()
        # Model 1: EfficientNet-B2
        self.effnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features_effnet = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Identity()
        # Model 2: Vision Transformer
        self.vit = vit.vit_b_16(weights=vit.ViT_B_16_Weights.DEFAULT)
        in_features_vit = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        # Clasificator comun
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features_effnet + in_features_vit),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features_effnet + in_features_vit, num_classes)
        )
    def forward(self, x):
        # Pentru predicție, nu avem nevoie de gradient, putem opri calculul pentru viteză
        with torch.no_grad():
            eff_features = self.effnet(x)
            vit_features = self.vit(x)
        combined_features = torch.cat((eff_features, vit_features), dim=1)
        output = self.classifier(combined_features)
        return output

# --- Pasul 2: Funcția de predicție ---
def predict_emotion(image_path, model_path):
    # Definirea celor 14 emoții, în aceeași ordine ca la antrenament
    EMOTIONS = ['Sadness', 'Trust', 'Fear', 'Disgust', 'Anger', 'Anticipation', 'Happiness', 
                'Love', 'Surprise', 'Optimism', 'Gratitude', 'Pessimism', 'Regret', 'Agreeableness']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Se folosește dispozitivul: {device}")
    
    # Inițiem modelul și încărcăm greutățile salvate
    model = EmotionEnsemble(num_classes=len(EMOTIONS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Foarte important: setăm modelul în modul de evaluare

    # Pregătim imaginea
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Facem predicția
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().squeeze()

    # Creăm o listă de rezultate (emoție, probabilitate)
    results = []
    for i, emotion in enumerate(EMOTIONS):
        prob = probabilities[i].item()
        results.append((emotion, prob))

    # Sortăm rezultatele după probabilitate, de la cea mai mare la cea mai mică
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Afișăm rezultatele
    print("\n--- Rezultatele Predicției ---")
    print("\nEmoțiile detectate, sortate după probabilitate:")
    for emotion, prob in results:
        print(f"- {emotion:<15}: {prob:.3f}")
    print("---------------------------------")
    print("\nNotă: Pentru o decizie finală (DA/NU), compară aceste probabilități cu pragurile optime găsite la finalul antrenamentului.")


# --- Pasul 3: Rularea scriptului cu interfață grafică ---
if __name__ == '__main__':
    # Calea către modelul antrenat pentru licență
    MODEL_PATH = "models/model_licenta_best.pth"

    if not os.path.exists(MODEL_PATH):
        print(f"EROARE: Fișierul model nu a fost găsit la calea: {MODEL_PATH}")
        print("Asigură-te că ai rulat antrenamentul final și că modelul este în folderul 'models'.")
    else:
        root = tk.Tk()
        root.withdraw()
        print("Se deschide fereastra pentru a selecta o imagine...")
        image_path = filedialog.askopenfilename(
            title="Selectează o imagine pentru analiză",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if image_path:
            print(f"\nAi selectat imaginea: {os.path.basename(image_path)}")
            predict_emotion(image_path, MODEL_PATH)
        else:
            print("\nNicio imagine nu a fost selectată. Programul se închide.")