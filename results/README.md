# Résultats

Les tableaux, figures et exemples visuels produits par les notebooks seront
enregistrés ici. Les résultats régénérables peuvent être exclus du dépôt si
leur taille est importante.

## Fichiers produits par l'évaluation

Le [notebook 04](../notebooks/04_evaluation.ipynb) sauvegarde les fichiers
suivants dans `MyDrive/ia_visuelle/results/` lors d'une exécution dans Colab.

| Fichier | Contenu |
| --- | --- |
| `evaluation_metrics.csv` | Mesures de qualité, temps et mémoire pour chaque configuration disponible. |
| `evaluation_predictions.csv` | Décisions détaillées et labels automatiques pour toutes les paires évaluées. |
| `evaluation_humans.csv` | Accord avec les annotations humaines et corrélation avec leurs notes, si elles sont fournies. |
| `evaluation_examples.csv` | Paires choisies pour illustrer des réussites et des erreurs. |
| `human_annotations_template.csv` | Tableau vide à copier pour préparer les annotations humaines. |
| `figures/comparaison_configurations.png` | Comparaison de l'accuracy, du temps et de la mémoire. |
| `figures/matrices_confusion.png` | Nombre de bonnes et de mauvaises décisions par classe. |
| `figures/exemples_<modele>_<precision>.png` | Images et légendes illustrant les décisions de chaque configuration. |

## Lire les mesures pour le rapport

- L'accuracy, le F1 et l'AUC sont compris entre 0 et 1 ; par exemple, une
  accuracy de `0.90` correspond à 90 % de décisions correctes.
- `accuracy_delta_vs_fp16_pp` donne l'écart avec la version fp16 du même
  modèle, en points de pourcentage. Une valeur négative indique une baisse.
- `seconds_per_pair` donne le temps moyen par paire et `peak_memory_mb`
  le pic de mémoire allouée par PyTorch pendant l'expérience.
- `tied_images` compte les images pour lesquelles les deux légendes ont
  exactement la même loss. Le notebook 03 les départage selon l'ordre des lignes.
- `NaN` indique une mesure absente ou indéfinie ; cette valeur ne signifie pas zéro.

Les exemples visuels sont sélectionnés pour montrer des réussites et des
erreurs. Leurs proportions ne représentent pas celles de tout le jeu évalué.

## Compléter la comparaison humaine

Copier `human_annotations_template.csv` vers `human_annotations.csv`, puis
regarder les images et remplir `human_label` (0 ou 1) et éventuellement
`human_score` (note de 1 à 5). Garder les identifiants et les légendes inchangés.
Les lignes non annotées peuvent rester vides. Relancer ensuite le notebook 04.

Sans annotations remplies, `evaluation_humans.csv` ne contient que les noms
des colonnes. Les labels automatiques du notebook 02 ne remplacent pas les
jugements humains.
