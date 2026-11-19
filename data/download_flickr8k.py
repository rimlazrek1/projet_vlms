"""Script utilitaire de téléchargement et d'extraction de l'échantillon Flickr8k."""

import argparse
from pathlib import Path
from datasets import load_dataset
import pandas as pd

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = 42


def download_flickr8k_sample(output_dir: Path, sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED):
    """Télécharge l'échantillon Flickr8k et sauvegarde la table des références."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du jeu de données Flickr8k (taille demandée : {sample_size})...")
    dataset = load_dataset("jxie/flickr8k", split="train")
    sample = dataset.shuffle(seed=seed).select(range(min(sample_size, len(dataset))))

    rows = []
    print("Sauvegarde des images et extraction des légendes de référence...")
    for idx, item in enumerate(sample):
        image = item["image"]
        image_path = images_dir / f"{idx}.jpg"
        if not image_path.exists():
            image.save(image_path)

        rows.append({
            "id": idx,
            "caption_0": item.get("caption_0", ""),
            "caption_1": item.get("caption_1", ""),
            "caption_2": item.get("caption_2", ""),
            "caption_3": item.get("caption_3", ""),
            "caption_4": item.get("caption_4", ""),
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "flickr8k_references.csv"
    df.to_csv(csv_path, index=False)
    print(f"Extraction terminée avec succès. Fichier sauvegardé : {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharger l'échantillon Flickr8k")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Dossier de sortie")
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Nombre d'images")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Graine aléatoire")
    args = parser.parse_args()

    download_flickr8k_sample(args.output_dir, args.size, args.seed)
