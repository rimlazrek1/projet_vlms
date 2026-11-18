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

## Sauvegarde Google Drive

Les notebooks montent automatiquement `MyDrive/ia_visuelle`. Les fichiers
créés sont sauvegardés dans `data/` ou `results/` sur Google Drive, afin que
les notebooks suivants puissent les réutiliser après un redémarrage du PC ou
une nouvelle session Colab.
