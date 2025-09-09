# verifica_coloane.py (Versiune îmbunătățită)
import pandas as pd
import os

csv_file_path = os.path.join("data", "WikiArt_Organized_Emotions_Metadata.csv")

try:
    # Citim fișierul cu separatorul virgulă și gestionăm rândurile cu erori
    df = pd.read_csv(csv_file_path, delimiter=',', on_bad_lines='skip')

    print("--- Numele Coloanelor Găsite ---")
    print(list(df.columns))
    print("\n" + "="*50 + "\n")

    print("--- Primele 3 Rânduri din Tabel (pentru a vedea conținutul) ---")
    # Afișăm primele 3 rânduri pentru a vedea ce conține fiecare coloană
    print(df.head(3))

except Exception as e:
    print(f"A apărut o eroare: {e}")