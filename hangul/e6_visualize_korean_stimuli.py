#!/usr/bin/env python3
"""
Quick visualization of Hangul target/test stimuli at different sizes.

Creates a 2x4 panel similar to the paper-style schematic:
- Row 1: target + same character at varied scales
- Row 2: target + different character at varied scales
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def list_imagefolder_samples(data_dir: str) -> List[Tuple[str, str]]:
    """
    Return flattened ImageFolder-like sample ordering:
    [(class_name, file_path), ...] with class and file names sorted.
    """
    class_dirs = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    samples: List[Tuple[str, str]] = []
    for cls in class_dirs:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
        for f in files:
            samples.append((cls, os.path.join(cls_dir, f)))
    return samples


def build_stimulus(img_path: str, canvas_size: int, char_size: int) -> np.ndarray:
    # Robustly extract glyph foreground, resize to desired char size, and center on black canvas.
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Estimate background polarity from borders.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    white_background = border.mean() > 127

    # Build foreground mask and bbox.
    if white_background:
        mask = arr < 245  # dark text on light background
        glyph = 255 - arr  # convert to white glyph on black
    else:
        mask = arr > 10   # light text on dark background
        glyph = arr

    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        glyph = glyph[y0:y1 + 1, x0:x1 + 1]

    img = Image.fromarray(glyph, mode="L").resize((char_size, char_size), Image.LANCZOS)

    canvas = Image.new("L", (canvas_size, canvas_size), color=0)
    x = (canvas_size - char_size) // 2
    y = (canvas_size - char_size) // 2
    canvas.paste(img, (x, y))
    return np.array(canvas)


def main():
    parser = argparse.ArgumentParser(description="Visualize Hangul target/test stimuli at experiment scales.")
    parser.add_argument("--data-dir", default="/gpfs/data/tserre/npant1/hangul_data", help="Hangul ImageFolder root.")
    parser.add_argument("--output", default="hangul/hangul_stimuli_preview.png", help="Output image path.")
    parser.add_argument("--canvas-size", type=int, default=200, help="Canvas size used for centering/padding.")
    parser.add_argument("--sizes", nargs=3, type=int, default=[13, 52, 130], help="Character sizes for 30', 2°, 5°.")
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Index of target/distractor pair in flattened dataset order (uses rows 2k and 2k+1).",
    )
    args = parser.parse_args()

    samples = list_imagefolder_samples(args.data_dir)
    if len(samples) < 2:
        raise RuntimeError(f"Need at least 2 images in {args.data_dir}")
    pair_start = 2 * args.pair_index
    if pair_start + 1 >= len(samples):
        raise IndexError(
            f"pair-index={args.pair_index} is out of range for {len(samples)} samples "
            f"(max pair-index={(len(samples) - 2) // 2})."
        )

    # Match e1_korean_imagenet pairing convention: target/distractor are adjacent indices.
    target_cls, target_path = samples[pair_start]
    diff_cls, diff_path = samples[pair_start + 1]
    s30, s2, s5 = args.sizes
    scale_label = {s30: "30'", s2: "2°", s5: "5°"}

    # Row 1: same character, varied scales.
    row1 = [
        build_stimulus(target_path, args.canvas_size, s30),
        build_stimulus(target_path, args.canvas_size, s30),
        build_stimulus(target_path, args.canvas_size, s2),
        build_stimulus(target_path, args.canvas_size, s5),
    ]
    # Row 2: different character, varied scales.
    row2 = [
        build_stimulus(target_path, args.canvas_size, s30),
        build_stimulus(diff_path, args.canvas_size, s30),
        build_stimulus(diff_path, args.canvas_size, s2),
        build_stimulus(diff_path, args.canvas_size, s5),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    size_order = [s30, s30, s2, s5]

    for c in range(4):
        axes[0, c].imshow(row1[c], cmap="gray", vmin=0, vmax=255)
        axes[1, c].imshow(row2[c], cmap="gray", vmin=0, vmax=255)
        # Put scale label inside each canvas (bottom-left), matching paper-style annotation.
        label = f"({scale_label[size_order[c]]})"
        for r in (0, 1):
            axes[r, c].text(
                0.05, 0.05, label,
                transform=axes[r, c].transAxes,
                color="white",
                fontsize=16,
                ha="left",
                va="bottom",
                fontweight="bold",
            )
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])

    text_fontsize = 20
    # Left column labels: Target on both rows.
    axes[0, 0].set_title("Target", fontsize=text_fontsize)
    axes[1, 0].set_title("Target", fontsize=text_fontsize)

    # Layout first.
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Then tighten spacing among the right 3 columns (cols 1,2,3) only.
    p1 = axes[0, 1].get_position()
    p2 = axes[0, 2].get_position()
    p3 = axes[0, 3].get_position()
    w = p1.width
    gap12 = p2.x0 - (p1.x0 + w)
    gap23 = p3.x0 - (p2.x0 + w)
    shrink = 0.45  # smaller -> tighter spacing between right 3 columns
    new_gap12 = gap12 * shrink
    new_gap23 = gap23 * shrink
    x1 = p1.x0
    x2 = x1 + w + new_gap12
    x3 = x2 + w + new_gap23

    for r in (0, 1):
        pr1 = axes[r, 1].get_position()
        pr2 = axes[r, 2].get_position()
        pr3 = axes[r, 3].get_position()
        axes[r, 1].set_position([x1, pr1.y0, pr1.width, pr1.height])
        axes[r, 2].set_position([x2, pr2.y0, pr2.width, pr2.height])
        axes[r, 3].set_position([x3, pr3.y0, pr3.width, pr3.height])

    row0_right_l = axes[0, 1].get_position().x0
    row0_right_r = axes[0, 3].get_position().x1
    row0_y = axes[0, 1].get_position().y1 + 0.01
    row1_right_l = axes[1, 1].get_position().x0
    row1_right_r = axes[1, 3].get_position().x1
    row1_y = axes[1, 1].get_position().y1 + 0.01

    fig.text(
        (row0_right_l + row0_right_r) / 2,
        row0_y,
        "Test = same character, varied scales",
        ha="center",
        va="bottom",
        fontsize=text_fontsize,
    )
    fig.text(
        (row1_right_l + row1_right_r) / 2,
        row1_y,
        "Test = different character, varied scales",
        ha="center",
        va="bottom",
        fontsize=text_fontsize,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
