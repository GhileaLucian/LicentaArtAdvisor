# 1_curata_imagini.py (Versiunea Corectată)
import os
from PIL import Image
from tqdm import tqdm
import logging

# Configurare logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Căi către foldere
sursa_dir = os.path.join("data", "imagini_initiale")
dest_dir = os.path.join("data", "imagini_curate")
corupt_dir = os.path.join("data", "imagini_corupte")

# Creare directoare de destinație
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(corupt_dir, exist_ok=True)

valid_images = 0
corrupt_images = 0

image_files = [f for f in os.listdir(sursa_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for file in tqdm(image_files, desc="Verificare imagini"):
    file_path_sursa = os.path.join(sursa_dir, file)
    file_path_corupt = os.path.join(corupt_dir, file)
    file_path_dest = os.path.join(dest_dir, file)
    
    img = None  # Inițializăm variabila pentru imagine

    try:
        img = Image.open(file_path_sursa)
        img.verify()  # Verifică integritatea fișierului
        img.close()   # Eliberăm fișierul după verificare

        # Re-deschidem imaginea pentru a o procesa și salva
        img = Image.open(file_path_sursa)
        img.load()
        img.convert('RGB').save(file_path_dest, "JPEG")
        valid_images += 1
        img.close() # Eliberăm fișierul și aici

    except Exception as e:
        logging.warning(f"Imagine coruptă detectată: {file} - Eroare: {e}")
        
        # AICI ESTE CORECȚIA CHEIE:
        # Ne asigurăm că fișierul este închis înainte de a-l muta.
        if img:
            img.close()
            
        # Acum mutăm fișierul în siguranță
        os.rename(file_path_sursa, file_path_corupt)
        corrupt_images += 1

logging.info(f"Procesare finalizată. Imagini valide: {valid_images}. Imagini corupte: {corrupt_images}.")