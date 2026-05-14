"""
seed_studio_speakers.py — register XTTS-v2 built-in studio speakers into the SpeakerStore.

Reads speakers_xtts.pth directly from MODEL_PATH — the full model is NOT loaded, so
this runs in seconds even on CPU.  Writes one directory per speaker under SPEAKERS_DIR:

    SPEAKERS_DIR/
        {slug}/
            latents.npz   (gpt_cond_latent, speaker_embedding as float32 numpy)
            meta.json     (slug, original_name, created_at, source: studio)

No ref.wav is written; the SpeakerStore treats its absence as a studio speaker.

Usage
-----
    # from project root, with .env or env vars set:
    python seed_studio_speakers.py

    # explicit paths:
    python seed_studio_speakers.py --model-path ./model --speakers-dir ./xtts_server/speakers

    # overwrite already-registered speakers:
    python seed_studio_speakers.py --force

    # dry run — print what would be registered without writing:
    python seed_studio_speakers.py --dry-run
"""

import argparse
import json
import os
import sys
import unicodedata
from datetime import UTC, datetime


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """
    Convert a studio speaker display name to a SpeakerStore-safe slug.

    Rules: alphanumeric + hyphens + underscores only, max 64 chars.
    Spaces → underscores; accented chars → ASCII equivalents.

    "Claribel Dervla"  → "Claribel_Dervla"
    "Alma María"       → "Alma_Maria"
    "Camilla Holmström"→ "Camilla_Holmstrom"
    """
    # NFKD decomposition strips combining diacritics (accents become separate chars)
    normalized = unicodedata.normalize("NFKD", name)
    ascii_bytes = normalized.encode("ascii", errors="ignore")
    ascii_str = ascii_bytes.decode("ascii")

    # Replace spaces and any remaining non-word chars with underscores
    slug = ""
    for ch in ascii_str:
        if ch.isalnum() or ch in ("-", "_"):
            slug += ch
        elif ch == " ":
            slug += "_"
        # silently drop anything else

    return slug[:64]


# ---------------------------------------------------------------------------
# Shape normalisation
# ---------------------------------------------------------------------------


def _to_numpy_latent(tensor) -> "np.ndarray":
    """Convert gpt_cond_latent tensor to float32 numpy, ensuring shape (1, T, 1024)."""
    import numpy as np

    arr = tensor.numpy().astype(np.float32) if hasattr(tensor, "numpy") else tensor.astype(np.float32)
    if arr.ndim == 2:        # (T, 1024) → (1, T, 1024)
        arr = arr[None]
    return arr


def _to_numpy_embedding(tensor) -> "np.ndarray":
    """Convert speaker_embedding tensor to float32 numpy, ensuring shape (1, 512)."""
    import numpy as np

    arr = tensor.numpy().astype(np.float32) if hasattr(tensor, "numpy") else tensor.astype(np.float32)
    arr = arr.squeeze()      # (1, 512, 1) or (512,) or (1, 512) → (512,)
    if arr.ndim == 1:        # (512,) → (1, 512)
        arr = arr[None]
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="Seed XTTS-v2 studio speakers into SpeakerStore")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "./model"),
        help="Directory containing speakers_xtts.pth (default: $MODEL_PATH or ./model)",
    )
    parser.add_argument(
        "--speakers-dir",
        default=os.environ.get("SPEAKERS_DIR", "./xtts_server/speakers"),
        help="SpeakerStore root directory (default: $SPEAKERS_DIR or ./xtts_server/speakers)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite speakers that are already registered",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be registered without writing any files",
    )
    args = parser.parse_args()

    pth_path = os.path.join(args.model_path, "speakers_xtts.pth")
    if not os.path.isfile(pth_path):
        print(f"ERROR: speakers_xtts.pth not found at {pth_path}", file=sys.stderr)
        print("       Make sure MODEL_PATH points to a full XTTS-v2 model directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {pth_path} …")
    data = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Flat dict: {display_name: {gpt_cond_latent: tensor, speaker_embedding: tensor}}
    speakers: dict = data if isinstance(data, dict) and not ("speakers" in data and len(data) == 1) else data["speakers"]

    print(f"Found {len(speakers)} studio speaker(s)\n")

    registered = 0
    skipped = 0
    errors = 0

    for display_name, tensors in speakers.items():
        slug = _slugify(display_name)
        if not slug:
            print(f"  SKIP  '{display_name}' — could not produce a valid slug")
            skipped += 1
            continue

        speaker_dir = os.path.join(args.speakers_dir, slug)
        latents_path = os.path.join(speaker_dir, "latents.npz")
        meta_path = os.path.join(speaker_dir, "meta.json")

        if os.path.isfile(latents_path) and not args.force:
            print(f"  skip  {slug!r:36s} (already exists — use --force to overwrite)")
            skipped += 1
            continue

        try:
            gpt = _to_numpy_latent(tensors["gpt_cond_latent"])
            emb = _to_numpy_embedding(tensors["speaker_embedding"])
        except Exception as exc:
            print(f"  ERROR {slug!r}: failed to convert tensors — {exc}")
            errors += 1
            continue

        label = "dry-run" if args.dry_run else "register"
        print(
            f"  {label:8s} {slug!r:36s}  "
            f"gpt={gpt.shape}  emb={emb.shape}  "
            f"('{display_name}')"
        )

        if args.dry_run:
            registered += 1
            continue

        try:
            os.makedirs(speaker_dir, exist_ok=True)
            np.savez(latents_path, gpt_cond_latent=gpt, speaker_embedding=emb)
            meta = {
                "name": slug,
                "original_name": display_name,
                "created_at": datetime.now(UTC).isoformat(),
                "source": "studio",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            registered += 1
        except Exception as exc:
            print(f"  ERROR {slug!r}: failed to write files — {exc}")
            errors += 1

    print()
    action = "would register" if args.dry_run else "registered"
    print(f"Done — {action} {registered}, skipped {skipped}, errors {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
