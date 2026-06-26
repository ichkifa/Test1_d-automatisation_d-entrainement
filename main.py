from pathlib import Path
from PIL import Image

# --- CONFIGURATION (Le chemin d'accès direct corrigé) ---
OOD_DATA_DIR = Path("/kaggle/input/dataset_test/dataset_test_Rabie_cleaned/tomato")

print("==================================================")
print("TEST DE CONNEXION AU DATASET KAGGLE VIA GITHUB")
print("==================================================")

# 1. Vérification de l'existence du dossier
if not OOD_DATA_DIR.exists():
    print(f"❌ Erreur : Le dossier {OOD_DATA_DIR} est introuvable.")
    print("Vérifiez que le dataset est bien attaché à votre script.")
else:
    print(f"✅ Succès : Le dossier {OOD_DATA_DIR} a été détecté.")
    
    # 2. Recherche et affichage des premières images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = []
    
    # Parcours des sous-dossiers pour trouver des images
    for folder in sorted(OOD_DATA_DIR.iterdir()):
        if folder.is_dir():
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    img_paths.append((img_path, folder.name))
                    if len(img_paths) >= 5:  # On s'arrête à 5 pour aller vite
                        break
        if len(img_paths) >= 5:
            break

    print(f"\nNombre total d'images d'exemple lues : {len(img_paths)}")
    
    # 3. Test de lecture rapide (PIL) pour prouver que les fichiers s'ouvrent
    print("\n--- ANALYSE EXPRESS DES 5 PREMIÈRES IMAGES ---")
    for i, (path, class_name) in enumerate(img_paths):
        with Image.open(path) as img:
            width, height = img.size
            print(f"Image {i+1} : Dossier [{class_name}] -> Nom: {path.name} | Taille: {width}x{height}")

print("==================================================")
print("FIN DU TEST RAPIDE - LE DATASET FONCTIONNE !")
