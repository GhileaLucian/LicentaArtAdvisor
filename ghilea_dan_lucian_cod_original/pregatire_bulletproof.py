# pregatire_bulletproof.py
import pandas as pd
import os
import shutil
import random
import logging
from tqdm import tqdm
import csv

# --- Configurare ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log_pregatire.log", mode='w', encoding='utf-8'), 
                              logging.StreamHandler()])
random.seed(42)

# --- Căi ---
METADATA_INITIAL_FILE = os.path.join("data", "WikiArt_Organized_Emotions_Metadata.csv")
MISSING_FILES_LIST = "missing_files.txt"
METADATA_CURAT_FILE = os.path.join("data", "metadata_curat.csv")
IMAGINI_SURSA_DIR = os.path.join("data", "imagini_curate")
TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "validation")
TEST_DIR = os.path.join("data", "test")

# --- PASUL 1: Parsare manuală și robustă a fișierului CSV ---
logging.info("--- START: Pasul 1 - Parsare manuală a CSV-ului ---")

try:
    with open(MISSING_FILES_LIST, 'r', encoding='utf-8') as f:
        missing_files_set = {line.strip() for line in f if line.strip()}
    logging.info(f"Am încărcat {len(missing_files_set)} fișiere din lista de fișiere lipsă.")

    # Citim fișierul linie cu linie folosind modulul `csv`
    with open(METADATA_INITIAL_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        all_rows = list(reader)

    # Procesăm header-ul
    header = [col.replace(';', '').strip() for col in all_rows[0]]
    num_columns = len(header)
    logging.info(f"Header detectat cu {num_columns} coloane.")

    # Procesăm datele, păstrând doar rândurile valide
    data_rows = []
    for i, row in enumerate(all_rows[1:]):
        if len(row) == num_columns:
            # Curățăm ultima coloană de semicolons
            row[-1] = row[-1].replace(';', '').strip()
            data_rows.append(row)
        else:
            logging.warning(f"Rândul {i+2} a fost ignorat (are {len(row)} coloane, se așteptau {num_columns}).")

    # Creăm DataFrame-ul din datele curate
    df = pd.DataFrame(data_rows, columns=header)
    logging.info(f"S-au procesat {len(df)} rânduri valide din CSV.")
    
    # Filtrăm pe baza listei de fișiere lipsă
    df_curat = df[~df['ImageName'].isin(missing_files_set)]
    logging.info(f"Fișierul CSV curățat are acum {len(df_curat)} rânduri.")

    df_curat.to_csv(METADATA_CURAT_FILE, index=False, sep=',')
    logging.info(f"Fișierul CSV curățat a fost salvat ca: {METADATA_CURAT_FILE}")

except Exception as e:
    logging.error(f"A apărut o eroare neașteptată la Pasul 1: {e}", exc_info=True)
    exit()

logging.info("--- FINAL: Pasul 1 - Curățarea CSV a fost finalizată cu succes. ---")


# --- PASUL 2: Împărțirea imaginilor (neschimbat) ---
logging.info("--- START: Pasul 2 - Împărțirea imaginilor în seturi ---")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

imagini_valide = df_curat['ImageName'].dropna().str.strip().tolist()
random.shuffle(imagini_valide)

logging.info(f"Am găsit {len(imagini_valide)} imagini valide de împărțit.")

train_split = 0.7
val_split = 0.15
num_imagini = len(imagini_valide)
train_end = int(num_imagini * train_split)
val_end = train_end + int(num_imagini * val_split)
train_files = imagini_valide[:train_end]
val_files = imagini_valide[train_end:val_end]
test_files = imagini_valide[val_end:]

def copy_files(files, dest_dir):
    copied_count = 0
    for f in tqdm(files, desc=f"Copiere în {os.path.basename(dest_dir)}"):
        sursa_path = os.path.join(IMAGINI_SURSA_DIR, f.strip())
        if os.path.exists(sursa_path):
            shutil.copy(sursa_path, os.path.join(dest_dir, f.strip()))
            copied_count += 1
        else:
            logging.warning(f"Fișierul '{f}' nu a fost găsit în sursă și nu a fost copiat.")
    return copied_count

train_count = copy_files(train_files, TRAIN_DIR)
val_count = copy_files(val_files, VAL_DIR)
test_count = copy_files(test_files, TEST_DIR)

logging.info("--- REZUMAT FINAL ---")
logging.info(f"Imagini de antrenament copiate: {train_count}")
logging.info(f"Imagini de validare copiate: {val_count}")
logging.info(f"Imagini de test copiate: {test_count}")

total_copied = train_count + val_count + test_count
if total_copied > 0:
    logging.info("--- FELICITĂRI! Pregătirea datelor s-a finalizat cu succes! ---")
else:
     logging.error("EROARE: Niciun fișier nu a fost copiat.")