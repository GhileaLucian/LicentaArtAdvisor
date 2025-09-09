# predictors/emotion_predictor.py
# VERSIUNEA FINALĂ, CORECTATĂ ȘI OPTIMIZATĂ

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torchvision.models.vision_transformer as vit
import streamlit as st
import operator # MODIFICARE: Am adăugat importul pentru sortare

# --- CONSTANTE ȘI CONFIGURARE ---
MODEL_PATH = "models/model_licenta_definitiv.pth"
# Traducem numele emoțiilor pentru o afișare mai elegantă în interfață
EMOTIONS_MAP = {
    'Sadness': 'Tristețe', 'Trust': 'Încredere', 'Fear': 'Frică', 'Disgust': 'Dezgust', 
    'Anger': 'Furie', 'Anticipation': 'Anticipare', 'Happiness': 'Fericire', 
    'Love': 'Iubire', 'Surprise': 'Surpriză', 'Optimism': 'Optimism', 
    'Gratitude': 'Recunoștință', 'Pessimism': 'Pesimism', 'Regret': 'Regret', 
    'Agreeableness': 'Amabilitate'
}
# Folosim cheile din map pentru a se potrivi cu modelul antrenat
EMOTIONS_KEYS = list(EMOTIONS_MAP.keys())


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEFINIȚIA ARHITECTURII MODELULUI ---
class EmotionEnsemble(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(EmotionEnsemble, self).__init__()
        self.effnet = models.efficientnet_b2(weights=None)
        in_features_effnet = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Identity()
        self.vit = vit.vit_b_16(weights=None)
        in_features_vit = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features_effnet + in_features_vit),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features_effnet + in_features_vit, num_classes)
        )
    def forward(self, x):
        eff_features = self.effnet(x)
        vit_features = self.vit(x)
        combined_features = torch.cat((eff_features, vit_features), dim=1)
        output = self.classifier(combined_features)
        return output
        
# --- FUNCȚIA DE ÎNCĂRCARE A MODELULUI CU CACHING ---
@st.cache_resource
def load_emotion_model():
    print("Încărcare model EMOȚIE... (rulează o singură dată)")
    model = EmotionEnsemble(num_classes=len(EMOTIONS_KEYS))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()
    return model

# --- DEFINIȚIA TRANSFORMĂRILOR ---
def get_prediction_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- FUNCȚIA PRINCIPALĂ DE PREDICTIE ---
def predict_emotion(image_path: str):
    model = load_emotion_model()
    try:
        transform = get_prediction_transforms()
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            # Acest model este multi-label, deci folosim sigmoid
            probabilities = torch.sigmoid(outputs).cpu().squeeze()
        
        # Folosim map-ul pentru a traduce emoțiile în română
        results = {EMOTIONS_MAP[EMOTIONS_KEYS[i]]: prob.item() for i, prob in enumerate(probabilities)}
        
        # --- MODIFICARE: Sortăm și formatăm rezultatul pentru app.py ---
        predictions_sorted = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        
        return {"predictions_sorted": predictions_sorted}

    except Exception as e:
        print(f"Eroare detaliată în emotion_predictor: {e}")
        return {"error": f"Eroare în emotion_predictor: {e}", "predictions_sorted": []}