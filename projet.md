## Ce qu'on doit produire

- Un dépôt GitHub organisé et mis à jour chaque semaine. 
- Un modèle qui fonctionne sur Google Colab ou Kaggle avec 12 Go de VRAM maximum. 
- Une évaluation avec des mesures chiffrées et des exemples visuels. 
- Une petite démonstration interactive. 
- Un rapport de 6 à 8 pages et une soutenance (présentation pptx) de 10 minutes. 

## Github
Le dépôt est le carnet de bord du projet. Il doit montrer ce que vous avez fait, qui l’a fait et comment relancer les expériences.


| Dossier | Contenu |
| --- | --- |
| README.md | Présentation, installation, commande de lancement et résultats principaux. |
| notebooks/ | Notebooks numérotés : données, baseline, modèle, évaluation, démo. |
| src/  | Code réutilisable du projet. |
| configs/  | Paramètres des expériences. |
| data/ | Script de téléchargement et fiche des données, sans données lourdes.  |
| results/  | Figures, tableaux et exemples générés. |
| demo/ | Application simple et courte vidéo.|
| report/ | Rapport final.  |
| docs/ | Licences, fiche du modèle et réflexion éthique.  |


- Ne déposez pas les gros fichiers de données ou de modèles dans Git. 
- Faites des commits réguliers avec des messages précis. 
- Utilisez des issues pour répartir les tâches. 
- Chaque membre doit contribuer de manière visible


# Projet • VLMs légers pour légender ou juger des images

En une phrase Vous utilisez un petit modèle qui comprend à la fois l’image et le texte, soit pour créer des légendes, soit pour évaluer des images générées.

## Votre mission

- Tester deux petits VLMs sur le même jeu d’images.
- Comparer leurs versions en précision normale, 8 bits et 4 bits.
- Option A : créer des légendes pour entraîner une LoRA.
- Option B : comparer le jugement du VLM avec celui d’humains.

## Jeux de données conseillés et format raisonnable

| Élément | Recommandation |
| --- | --- |
| Jeu principal recommandé | Flickr8k : environ 8 000 images, chacune associée à cinq légendes. Pour l’évaluation seulement, un sous-ensemble de 500 à 2 000 images suffit. |
| Alternative légère  | Construire 500 à 1 000 images libres depuis Wikimedia Commons, avec une légende de référence vérifiée manuellement.  |
| Préparation | Redimensionner les images à la taille attendue par le VLM ; réutiliser les mêmes images pour comparer précision normale, 8 bits et 4 bits.  |
| À éviter | Ne pas affiner complètement un VLM. Faire principalement de l’inférence ou, si nécessaire, une adaptation légère sur un petit échantillon. |

## Notions techniques à connaître

- VLM : modèle vision-langage.
- Quantification : réduction de la mémoire utilisée par le modèle.
- Corrélation : niveau d’accord entre deux systèmes de notation.

## Démo finale possible 

Charger une image pour obtenir une légende ou un score, avec choix du niveau de quantification.