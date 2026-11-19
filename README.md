# VLMs légers pour juger des images

Projet réalisé dans Google Colab avec une limite de 12 Go de VRAM.

## Objectif

Comparer deux petits modèles vision-langage (VLM), **BLIP** et **GIT**, sur un même
sous-ensemble de Flickr8k. Chaque modèle est testé en précision normale (fp16), en
8 bits (int8) et en 4 bits (int4). Le projet compare ensuite leurs jugements avec des
notations humaines et fournit une démonstration interactive complète.

## Organisation

```text
.
├── configs/     Paramètres et configurations des expériences
├── data/        Scripts de téléchargement et documentation des données
├── demo/        Démonstration interactive Gradio (script et interface web)
├── docs/        Documentation et réflexion éthique
├── notebooks/   Notebooks Google Colab numérotés (01 à 05)
├── report/      Rapport final et présentation
├── results/     Figures, tableaux et exemples produits
└── src/         Code Python réutilisable (chargement, inférence, métriques)
```

Les images, modèles téléchargés et autres fichiers volumineux ne doivent pas
être ajoutés au dépôt GitHub.

## Feuille de route et notebooks

Les notebooks doivent être exécutés dans l'ordre suivant :

1. `01_preparation_flickr8k.ipynb` : Préparer l'environnement Google Colab et le sous-ensemble Flickr8k (500 images).
2. `02_baseline.ipynb` : Créer une baseline simple (alignement TF-IDF / hasard).
3. `03_modeles_blip_git.ipynb` : Exécuter BLIP et GIT avec les trois niveaux de précision (fp16, int8, int4).
4. `04_evaluation.ipynb` : Calculer les mesures chiffrées, produire les figures et comparer avec les annotations humaines.
5. `05_demonstration.ipynb` : Démonstration interactive avec interface Gradio (légendage, score de cohérence et comparaison A/B).

## Lancement de la démonstration

### Option A : Dans Google Colab (avec GPU T4)
Ouvrez le notebook [`notebooks/05_demonstration.ipynb`](notebooks/05_demonstration.ipynb) dans Colab, activez le GPU T4, puis cliquez sur **Exécution → Tout exécuter**. Une interface Gradio interactive s'affichera directement dans le notebook avec un lien public `gradio.live`.

### Option B : En local
```bash
pip install -r demo/requirements.txt
python demo/app.py
```
L'interface sera disponible sur `http://127.0.0.1:7860`.

## Exécution prévue

Les expériences lourdes sont lancées dans Google Colab. Les notebooks montent
automatiquement Google Drive et utilisent le dossier suivant :

```text
MyDrive/ia_visuelle/
├── data/
└── results/
```

Les fichiers intermédiaires et les résultats (`.csv`, `.json`, figures, etc.)
sont enregistrés dans ce dossier Drive. Une autorisation Google est demandée la première fois que le notebook monte Drive.
