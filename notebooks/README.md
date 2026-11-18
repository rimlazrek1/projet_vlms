# Notebooks Google Colab

Les notebooks seront numérotés et exécutés dans cet ordre :

1. Préparation des données et de l'environnement
2. Baseline sans modèle
3. Jugement image-légende avec BLIP et GIT (`03_modeles_blip_git.ipynb`)
4. Évaluation et comparaison avec les humains (`04_evaluation.ipynb`)
5. Démonstration

Le notebook 04 fonctionne sur CPU et lit les CSV produits par les notebooks
précédents. Il sauvegarde les mesures et figures sur Drive. La comparaison
humaine nécessite de compléter `results/human_annotations.csv` à partir du
modèle d'annotation généré par le notebook.

## Lancer le notebook 04

1. Ouvrir [04_evaluation.ipynb dans Colab](https://colab.research.google.com/github/rimlazrek1/projet_vlms/blob/main/notebooks/04_evaluation.ipynb).
2. Utiliser une session CPU : cette étape ne recharge pas BLIP et GIT.
3. Vérifier les fichiers dans `MyDrive/ia_visuelle/results/` :
   `baseline_alignment_pairs.csv`, les fichiers `judgements_*.csv`,
   `model_judgement_summary.csv` et `flickr8k_references.csv` pour les images.
4. Choisir **Exécution → Tout exécuter**, puis autoriser la connexion à Drive.
5. Retrouver les tableaux dans `results/` et les images dans `results/figures/`.

Si les CSV sont sur le Drive d'un autre membre du groupe, les copier dans ce
dossier sur son propre Drive avant de lancer le notebook. L'accès au dépôt
GitHub ne donne pas automatiquement accès aux fichiers Google Drive.

Les configurations manquantes sont signalées et les configurations disponibles
sont évaluées. Sans fichier `judgements_*.csv`, il faut d'abord lancer le
notebook 03. La comparaison humaine peut être complétée plus tard, sans
bloquer le calcul des autres mesures.

## Sauvegarde Google Drive

Les notebooks montent automatiquement `MyDrive/ia_visuelle`. Les fichiers
créés sont sauvegardés dans `data/` ou `results/` sur Google Drive, afin que
les notebooks suivants puissent les réutiliser après un redémarrage du PC ou
une nouvelle session Colab.
