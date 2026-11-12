# VLMs légers pour juger des images

Projet réalisé dans Google Colab avec une limite de 12 Go de VRAM.

## Objectif

Comparer deux petits modèles vision-langage (VLM), BLIP et GIT, sur un même
sous-ensemble de Flickr8k. Chaque modèle sera testé en précision normale, en
8 bits et en 4 bits. Le projet comparera ensuite leurs jugements avec des
notations humaines.

## Organisation

```text
.
├── configs/     Paramètres des expériences
├── data/        Scripts et documentation des données
├── demo/        Démonstration interactive
├── docs/        Documentation et réflexion éthique
├── notebooks/   Notebooks Colab numérotés
├── report/      Rapport final et présentation
├── results/     Figures, tableaux et exemples produits
└── src/         Code Python réutilisable
```

Les images, modèles téléchargés et autres fichiers volumineux ne doivent pas
être ajoutés au dépôt GitHub.

## Feuille de route

1. Préparer l'environnement Google Colab et le sous-ensemble Flickr8k.
2. Créer une baseline simple.
3. Exécuter BLIP et GIT avec les trois niveaux de précision.
4. Recueillir les jugements humains.
5. Calculer les mesures et produire des exemples visuels.
6. Préparer la démonstration, le rapport et la soutenance.

## Exécution prévue

Les expériences seront lancées dans Google Colab. Les notebooks montent
automatiquement Google Drive et utilisent le dossier suivant :

```text
MyDrive/ia_visuelle/
├── data/
└── results/
```

Les fichiers intermédiaires et les résultats (`.csv`, `.json`, figures, etc.)
sont enregistrés dans ce dossier Drive. Une autorisation Google est demandée la première fois que le notebook monte Drive.
