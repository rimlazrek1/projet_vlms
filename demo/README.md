# Démonstration Interactive (BLIP & GIT)

Cette application interactive permet de tester et visualiser les capacités des modèles vision-langage légers **BLIP** (`Salesforce/blip-image-captioning-base`) et **GIT** (`microsoft/git-base`).

## Fonctionnalités

1. **Génération de légende (Captioning)** : Obtenez une description automatique générée par le VLM pour n'importe quelle image importée.
2. **Jugement image-légende (Scoring)** : Évaluez la cohérence d'une légende candidate à l'aide de la perte *cross-entropy* (score de cohérence).
3. **Comparaison A/B de légendes** : Comparez deux légendes pour une même image et découvrez laquelle est préférée par le modèle (reproduit le protocole d'évaluation des notebooks 02, 03 et 04).
4. **Niveaux de quantification** : Comparez les inférences en **fp16**, **int8** (8 bits) et **int4** (4 bits avec `bitsandbytes`).

---

## 1. Lancement dans Google Colab (Recommandé)

Pour exécuter la démonstration avec accélération GPU (T4 12 Go de VRAM) et quantification 8-bit / 4-bit :

1. Ouvrez le notebook [`notebooks/05_demonstration.ipynb`](../notebooks/05_demonstration.ipynb) dans Google Colab.
2. Activez l'accélérateur GPU T4 (**Exécution → Modifier le type d'exécution → T4 GPU**).
3. Exécutez les cellules : l'interface Gradio s'affichera directement dans le notebook et fournira un lien public `gradio.live`.

---

## 2. Lancement en local

### Prérequis

Installez les dépendances nécessaires :

```bash
pip install -r demo/requirements.txt
```

> **Note :** Sur Windows sans GPU CUDA compatible, les modes `int8` et `int4` nécessitent des configurations spécifiques de `bitsandbytes`. Le mode `fp32` ou `fp16` sur CPU/GPU fonctionne directement.

### Lancement

Exécutez le script depuis la racine du dépôt :

```bash
python demo/app.py
```

L'application sera accessible dans votre navigateur à l'adresse [http://127.0.0.1:7860](http://127.0.0.1:7860).
