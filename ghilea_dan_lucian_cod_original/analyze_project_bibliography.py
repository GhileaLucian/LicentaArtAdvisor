import os
import re
import pandas as pd
from pathlib import Path

def analyze_project_structure():
    """Analizează structura proiectului și extrage informații relevante"""
    
    base_paths = [
        r"D:\LicentaArtWorks\ProiectLicenta",
        r"D:\LicentaArtWorks\InterfataArtAdvisor", 
        r"D:\LicentaArtWorks\Research"
    ]
    
    project_info = {
        'datasets': [],
        'models': [],
        'frameworks': [],
        'academic_sources': [],
        'web_sources': [],
        'code_files': []
    }
    
    for base_path in base_paths:
        if os.path.exists(base_path):
            analyze_directory(base_path, project_info)
    
    return project_info

def analyze_directory(directory, project_info):
    """Analizează recursiv un director și extrage informații"""
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Analizează fișiere Python
            if file_ext == '.py':
                analyze_python_file(file_path, project_info)
            
            # Analizează fișiere Jupyter
            elif file_ext == '.ipynb':
                analyze_notebook_file(file_path, project_info)
            
            # Analizează fișiere de configurație
            elif file in ['requirements.txt', 'environment.yml', 'setup.py']:
                analyze_config_file(file_path, project_info)
            
            # Analizează fișiere README
            elif 'readme' in file.lower() or file_ext in ['.md', '.txt']:
                analyze_text_file(file_path, project_info)

def analyze_python_file(file_path, project_info):
    """Analizează fișiere Python pentru import-uri și referințe"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Detectează framework-uri
        frameworks = detect_frameworks(content)
        project_info['frameworks'].extend(frameworks)
        
        # Detectează modele
        models = detect_models(content)
        project_info['models'].extend(models)
        
        # Detectează dataset-uri
        datasets = detect_datasets(content)
        project_info['datasets'].extend(datasets)
        
        # Adaugă info despre fișier
        project_info['code_files'].append({
            'file': os.path.basename(file_path),
            'path': file_path,
            'type': 'Python Script'
        })
        
    except Exception as e:
        print(f"Eroare la citirea {file_path}: {e}")

def detect_frameworks(content):
    """Detectează framework-uri ML/DL din cod"""
    
    frameworks = []
    framework_patterns = {
        'TensorFlow': r'import tensorflow|from tensorflow',
        'Keras': r'import keras|from keras',
        'PyTorch': r'import torch|from torch',
        'Scikit-learn': r'from sklearn|import sklearn',
        'OpenCV': r'import cv2|from cv2',
        'Pandas': r'import pandas|from pandas',
        'NumPy': r'import numpy|from numpy',
        'Matplotlib': r'import matplotlib|from matplotlib',
        'Seaborn': r'import seaborn|from seaborn',
        'PIL/Pillow': r'from PIL|import PIL',
        'Transformers': r'from transformers|import transformers'
    }
    
    for framework, pattern in framework_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            frameworks.append(framework)
    
    return frameworks

def detect_models(content):
    """Detectează modele ML din cod"""
    
    models = []
    model_patterns = {
        'EfficientNet': r'EfficientNet|efficientnet',
        'ResNet': r'ResNet|resnet',
        'VGG': r'VGG|vgg',
        'Vision Transformer': r'ViT|vision.*transformer',
        'BERT': r'BERT|bert',
        'CNN': r'Conv2D|Convolution',
        'Random Forest': r'RandomForest',
        'SVM': r'SVC|SVM|support.*vector',
        'Logistic Regression': r'LogisticRegression'
    }
    
    for model, pattern in model_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            models.append(model)
    
    return models

def detect_datasets(content):
    """Detectează dataset-uri din cod"""
    
    datasets = []
    dataset_patterns = {
        'WikiArt': r'WikiArt|wikiart',
        'ArtEmis': r'ArtEmis|artemis',
        'IMDB': r'IMDB|imdb',
        'CIFAR': r'CIFAR|cifar',
        'ImageNet': r'ImageNet|imagenet',
        'Custom Dataset': r'custom.*dataset|proprietary.*data'
    }
    
    for dataset, pattern in dataset_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            datasets.append(dataset)
    
    return datasets

def analyze_notebook_file(file_path, project_info):
    """Analizează fișiere Jupyter Notebook"""
    
    project_info['code_files'].append({
        'file': os.path.basename(file_path),
        'path': file_path,
        'type': 'Jupyter Notebook'
    })

def analyze_config_file(file_path, project_info):
    """Analizează fișiere de configurație"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Detectează dependințe
        dependencies = extract_dependencies(content, file_path)
        project_info['frameworks'].extend(dependencies)
        
    except Exception as e:
        print(f"Eroare la citirea {file_path}: {e}")

def extract_dependencies(content, file_path):
    """Extrage dependințele din fișiere de configurație"""
    
    dependencies = []
    
    if 'requirements.txt' in file_path:
        lines = content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                dep = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                dependencies.append(dep)
    
    return dependencies

def analyze_text_file(file_path, project_info):
    """Analizează fișiere text pentru referințe academice"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Caută referințe academice
        academic_refs = find_academic_references(content)
        project_info['academic_sources'].extend(academic_refs)
        
        # Caută surse web
        web_refs = find_web_references(content)
        project_info['web_sources'].extend(web_refs)
        
    except Exception as e:
        print(f"Eroare la citirea {file_path}: {e}")

def find_academic_references(content):
    """Caută referințe academice în text"""
    
    references = []
    
    # Paterne pentru referințe academice
    patterns = [
        r'([A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\))',
        r'([A-Z][a-z]+\s*\(\d{4}\))',
        r'(doi:\s*10\.\d+/[^\s]+)',
        r'(arxiv:\s*\d+\.\d+)',
        r'(https?://arxiv\.org/[^\s]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        references.extend(matches)
    
    return references

def find_web_references(content):
    """Caută referințe web în text"""
    
    web_refs = []
    
    # Paterne pentru surse web
    patterns = [
        r'(https?://[^\s]+)',
        r'(www\.[^\s]+)',
        r'(github\.com/[^\s]+)',
        r'(kaggle\.com/[^\s]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        web_refs.extend(matches)
    
    return web_refs

def generate_bibliography_report(project_info):
    """Generează raportul bibliografic"""
    
    print("=" * 80)
    print("📋 FORMULAR STUDIU BIBLIOGRAFIC - GENERAT AUTOMAT")
    print("=" * 80)
    
    # 1. Framework-uri și tehnologii detectate
    print("\n🔹 1. FRAMEWORK-URI ȘI TEHNOLOGII DETECTATE:")
    unique_frameworks = list(set(project_info['frameworks']))
    for i, fw in enumerate(unique_frameworks, 1):
        print(f"{i}. {fw}")
    
    # 2. Modele detectate
    print("\n🔹 2. MODELE ML/DL DETECTATE:")
    unique_models = list(set(project_info['models']))
    for i, model in enumerate(unique_models, 1):
        print(f"{i}. {model}")
    
    # 3. Dataset-uri detectate
    print("\n🔹 3. DATASET-URI DETECTATE:")
    unique_datasets = list(set(project_info['datasets']))
    for i, dataset in enumerate(unique_datasets, 1):
        print(f"{i}. {dataset}")
    
    # 4. Fișiere de cod
    print("\n🔹 4. FIȘIERE DE COD ANALIZATE:")
    for i, code_file in enumerate(project_info['code_files'], 1):
        print(f"{i}. {code_file['file']} ({code_file['type']})")
    
    # 5. Referințe academice găsite
    print("\n🔹 5. REFERINȚE ACADEMICE GĂSITE:")
    unique_academic = list(set(project_info['academic_sources']))
    for i, ref in enumerate(unique_academic, 1):
        print(f"{i}. {ref}")
    
    # 6. Surse web găsite
    print("\n🔹 6. SURSE WEB GĂSITE:")
    unique_web = list(set(project_info['web_sources']))
    for i, ref in enumerate(unique_web[:10], 1):  # Limitez la primele 10
        print(f"{i}. {ref}")

def create_excel_bibliography():
    """Creează un template Excel pentru bibliografia"""
    
    # Template pentru surse academice
    academic_template = pd.DataFrame({
        'Nr.': [1, 2, 3, 4, 5],
        'Titlu lucrare/articol': [
            'WikiArt: A Large-scale Dataset for Art Analysis',
            'ArtEmis: Affective Language for Art',
            'Deep Learning for Emotion Recognition in Art',
            'EfficientNet: Rethinking Model Scaling',
            'Vision Transformer for Image Classification'
        ],
        'Autori': [
            'Saleh et al.',
            'Achlioptas et al.',
            'Să completezi',
            'Tan et al.',
            'Dosovitskiy et al.'
        ],
        'An': [2019, 2021, '', 2019, 2020],
        'Tip': ['Dataset Paper', 'Conference Paper', '', 'Conference Paper', 'Conference Paper'],
        'Ce ai folosit': [
            'Dataset principal pentru imagini artistice',
            'Metodologie pentru adnotarea emoțiilor',
            '',
            'Arhitectura modelului de clasificare',
            'Alternativă la CNN-uri clasice'
        ],
        'Link/PDF': [
            'https://arxiv.org/abs/1906.02874',
            'https://arxiv.org/abs/2101.07396',
            '',
            'https://arxiv.org/abs/1905.11946',
            'https://arxiv.org/abs/2010.11929'
        ]
    })
    
    # Template pentru dataset-uri
    dataset_template = pd.DataFrame({
        'Nr.': [1, 2, 3],
        'Numele datasetului': ['WikiArt', 'Dataset propriu curățat', 'Metadata emoțional'],
        'Link sursă': [
            'https://www.wikiart.org/',
            'Procesat local',
            'WikiArt_Organized_Emotions_Metadata.csv'
        ],
        'Structura datelor': [
            'Imagini artistice organizate pe stiluri/emoții',
            'Imagini filtrate și validate',
            'Metadata cu etichete emoționale'
        ],
        'Cum l-ai folosit': [
            'Sursa primară de date pentru antrenare',
            'Dataset curat pentru train/val/test',
            'Asocierea imaginilor cu emoțiile'
        ]
    })
    
    # Template pentru modele
    model_template = pd.DataFrame({
        'Nr.': [1, 2, 3],
        'Model utilizat': ['EfficientNetB0', 'CNN personalizat', 'Focal Loss'],
        'Framework': ['TensorFlow/Keras', 'TensorFlow', 'Custom implementation'],
        'Sursa modelului': [
            'tf.keras.applications.EfficientNetB0',
            'Arhitectură proprie',
            'Implementare pentru date dezbalansate'
        ],
        'Adaptare/antrenare': [
            'Fine-tuning cu date artistice',
            'Antrenare de la zero',
            'Pentru îmbunătățirea clasificării'
        ]
    })
    
    # Salvare în Excel
    with pd.ExcelWriter('bibliografia_proiect.xlsx', engine='openpyxl') as writer:
        academic_template.to_excel(writer, sheet_name='Surse Academice', index=False)
        dataset_template.to_excel(writer, sheet_name='Dataset-uri', index=False)
        model_template.to_excel(writer, sheet_name='Modele', index=False)
    
    print("\n✅ Fișierul 'bibliografia_proiect.xlsx' a fost creat cu template-uri!")

if __name__ == "__main__":
    print("🔍 Analizez structura proiectului...")
    project_info = analyze_project_structure()
    
    print("📊 Generez raportul bibliografic...")
    generate_bibliography_report(project_info)
    
    print("\n📝 Creez template-ul Excel...")
    create_excel_bibliography()
    
    print("\n🎉 Analiza completă! Verifică fișierul Excel generat.")
