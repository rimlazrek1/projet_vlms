"""Application de démonstration interactive pour les modèles vision-langage légers (BLIP & GIT).

Cette interface Gradio permet de :
1. Générer automatiquement une légende pour une image importée.
2. Évaluer et noter la cohérence d'une légende candidate (calcul de la perte et du score).
3. Comparer deux légendes pour une même image (détection de la meilleure description).
4. Tester l'impact du niveau de quantification (fp16, int8, int4, fp32).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image

# Ajouter le répertoire racine au PYTHONPATH pour importer src
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vlm_judge import (
    clear_device_cache,
    compare_two_captions,
    compute_alignment_loss,
    generate_caption,
    get_default_device,
    judge_caption,
    load_vlm,
)

# Cache global du modèle chargé
CURRENT_MODEL = None
CURRENT_PROCESSOR = None
CURRENT_CONFIG: Tuple[Optional[str], Optional[str]] = (None, None)


def get_or_load_model(model_name: str, precision: str):
    """Charge le modèle demandé ou réutilise l'instance en cache."""
    global CURRENT_MODEL, CURRENT_PROCESSOR, CURRENT_CONFIG

    target_config = (model_name.lower(), precision.lower())
    if CURRENT_MODEL is not None and CURRENT_CONFIG == target_config:
        return CURRENT_MODEL, CURRENT_PROCESSOR, "Modèle déjà en mémoire (cache)"

    # Libérer la mémoire précédente
    if CURRENT_MODEL is not None:
        del CURRENT_MODEL
        del CURRENT_PROCESSOR
        clear_device_cache()

    start = time.perf_counter()
    model, processor = load_vlm(model_name=model_name, precision=precision)
    load_time = time.perf_counter() - start

    CURRENT_MODEL = model
    CURRENT_PROCESSOR = processor
    CURRENT_CONFIG = target_config
    return model, processor, f"Modèle chargé en {load_time:.2f} s"


def handle_caption_generation(image: Image.Image, model_name: str, precision: str):
    """Génère une légende pour l'image fournie."""
    if image is None:
        return "Veuillez charger une image.", ""

    try:
        model, processor, status = get_or_load_model(model_name, precision)
        start = time.perf_counter()
        caption = generate_caption(model, processor, image, model_name=model_name)
        elapsed = time.perf_counter() - start
        info = f"{status} | Inférence : {elapsed:.2f} s | Précision : {precision} | Modèle : {model_name.upper()}"
        return caption, info
    except Exception as exc:
        return f"Erreur lors de la génération : {exc}", ""


def handle_caption_judgment(image: Image.Image, caption: str, model_name: str, precision: str):
    """Évalue la cohérence d'une légende candidate."""
    if image is None:
        return "Veuillez charger une image.", "", "", ""
    if not caption or not caption.strip():
        return "Veuillez saisir une légende à évaluer.", "", "", ""

    try:
        model, processor, status = get_or_load_model(model_name, precision)
        start = time.perf_counter()
        result = judge_caption(model, processor, image, caption.strip())
        elapsed = time.perf_counter() - start

        loss_str = f"{result['loss']:.4f}"
        score_str = f"{result['score']:.4f} (Perplexité : {result['perplexity']:.2f})"
        verdict_str = f"{result['verdict']} [Confiance : {result['confidence']}]"
        info = f"{status} | Calcul : {elapsed:.2f} s"
        return loss_str, score_str, verdict_str, info
    except Exception as exc:
        return f"Erreur : {exc}", "", "", ""


def handle_caption_comparison(
    image: Image.Image,
    caption_a: str,
    caption_b: str,
    model_name: str,
    precision: str,
):
    """Compare deux légendes et détermine la meilleure."""
    if image is None:
        return "Veuillez charger une image.", "", "", "", ""
    if not caption_a or not caption_b:
        return "Veuillez renseigner les deux légendes.", "", "", "", ""

    try:
        model, processor, status = get_or_load_model(model_name, precision)
        start = time.perf_counter()
        comp = compare_two_captions(model, processor, image, caption_a.strip(), caption_b.strip())
        elapsed = time.perf_counter() - start

        res_a = f"Loss : {comp['loss_a']:.4f} (Score : {comp['score_a']:.4f})"
        res_b = f"Loss : {comp['loss_b']:.4f} (Score : {comp['score_b']:.4f})"
        decision = f"Gagnant : {comp['winner']} (Écart de perte : {comp['loss_difference']:.4f})"
        pref = comp["winner_text"]
        info = f"{status} | Comparaison en {elapsed:.2f} s"
        return res_a, res_b, decision, pref, info
    except Exception as exc:
        return f"Erreur : {exc}", "", "", "", ""


# Construction de l'interface Gradio
device_label = get_default_device().upper()
title = "Démonstration Interactive : VLMs Légers (BLIP & GIT)"
description = (
    "Cette application permet de tester les modèles vision-langage légers **BLIP** "
    "(`Salesforce/blip-image-captioning-base`) et **GIT** (`microsoft/git-base`) "
    "avec différents niveaux de quantification (**fp16**, **int8**, **int4**). "
    f"Dispositif matériel détecté : **{device_label}**."
)

with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Radio(
                choices=["blip", "git"],
                value="blip",
                label="Modèle VLM",
                info="Sélectionnez l'architecture à utiliser",
            )
            precision_selector = gr.Dropdown(
                choices=["fp16", "int8", "int4", "fp32"],
                value="fp16" if device_label == "CUDA" else "fp32",
                label="Niveau de quantification",
                info="int8 et int4 réduisent l'empreinte VRAM via bitsandbytes",
            )
            input_image = gr.Image(
                type="pil",
                label="Image d'entrée (Glisser-déposer ou sélectionner)",
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                # Onglet 1 : Génération de légende
                with gr.TabItem("1. Générer une légende"):
                    gr.Markdown("### Génération automatique de description")
                    btn_generate = gr.Button("Générer la légende", variant="primary")
                    out_caption = gr.Textbox(label="Légende prédite par le VLM", lines=2)
                    out_gen_info = gr.Markdown()
                    btn_generate.click(
                        fn=handle_caption_generation,
                        inputs=[input_image, model_selector, precision_selector],
                        outputs=[out_caption, out_gen_info],
                    )

                # Onglet 2 : Jugement d'une légende
                with gr.TabItem("2. Juger une légende"):
                    gr.Markdown("### Évaluation de la cohérence d'une légende")
                    candidate_caption = gr.Textbox(
                        label="Légende candidate à évaluer",
                        placeholder="Ex: A dog playing in the grass with a red ball",
                        lines=2,
                    )
                    btn_judge = gr.Button("Évaluer la cohérence", variant="primary")
                    with gr.Row():
                        out_loss = gr.Textbox(label="Perte (Cross-Entropy Loss)")
                        out_score = gr.Textbox(label="Score de cohérence (-Loss)")
                    out_verdict = gr.Textbox(label="Verdict d'alignement")
                    out_judge_info = gr.Markdown()
                    btn_judge.click(
                        fn=handle_caption_judgment,
                        inputs=[
                            input_image,
                            candidate_caption,
                            model_selector,
                            precision_selector,
                        ],
                        outputs=[out_loss, out_score, out_verdict, out_judge_info],
                    )

                # Onglet 3 : Comparaison de deux légendes
                with gr.TabItem("3. Comparer 2 légendes"):
                    gr.Markdown("### Détection de la légende la plus adaptée (A vs B)")
                    caption_a_input = gr.Textbox(
                        label="Légende A",
                        placeholder="Ex: A cat sitting on a couch",
                    )
                    caption_b_input = gr.Textbox(
                        label="Légende B",
                        placeholder="Ex: An airplane flying in the sky",
                    )
                    btn_compare = gr.Button("Comparer les légendes", variant="primary")
                    with gr.Row():
                        out_comp_a = gr.Textbox(label="Résultat Légende A")
                        out_comp_b = gr.Textbox(label="Résultat Légende B")
                    out_comp_winner = gr.Textbox(label="Décision du modèle")
                    out_comp_text = gr.Textbox(label="Légende retenue")
                    out_comp_info = gr.Markdown()
                    btn_compare.click(
                        fn=handle_caption_comparison,
                        inputs=[
                            input_image,
                            caption_a_input,
                            caption_b_input,
                            model_selector,
                            precision_selector,
                        ],
                        outputs=[
                            out_comp_a,
                            out_comp_b,
                            out_comp_winner,
                            out_comp_text,
                            out_comp_info,
                        ],
                    )

                # Onglet 4 : À propos
                with gr.TabItem("4. Synthèse du projet"):
                    gr.Markdown(
                        """
                        ### Contexte et Objectifs du Projet
                        - **Objectif** : Comparer deux petits modèles vision-langage (**BLIP** et **GIT**) pour évaluer l'alignement image-texte.
                        - **Quantification** : Évaluation des performances et de l'empreinte mémoire en **fp16**, **int8** et **int4**.
                        - **Méthodologie** : Utilisation de la *cross-entropy loss* du modèle conditionné par l'image pour mesurer la vraisemblance du texte.
                        - **Évaluation** : Comparaison avec la baseline (TF-IDF/hasard) et avec les annotations humaines (Flickr8k).
                        """
                    )

if __name__ == "__main__":
    demo.launch(share=False)
