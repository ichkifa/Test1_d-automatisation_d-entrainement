import os
from pathlib import Path

print("==================================================")
print("EXPLORATION DES DOSSIERS DISPONIBLES SUR KAGGLE")
print("==================================================")

input_path = Path("/kaggle/input")

if not input_path.exists():
    print("❌ Le dossier racine /kaggle/input n'existe pas.")
else:
    print("Contenu du dossier /kaggle/input :")
    # Liste tous les sous-dossiers directement rattachés
    subfolders = [f.name for f in input_path.iterdir()]
    print(subfolders)
    
    print("\n--- RECHERCHE DE VOTRE DOSSIER TOMATE ---")
    # Recherche automatique de tous les dossiers nommés 'tomato' ou 'dataset'
    for root, dirs, files in os.walk("/kaggle/input"):
        if "tomato" in root.lower() or "dataset" in root.lower():
            print(f"Trouvé : {root}")
            # Si on trouve des fichiers à l'intérieur, on liste les 3 premiers
            items = os.listdir(root)
            if items:
                print(f"   -> Contenu (3 max) : {items[:3]}")

print("==================================================")
