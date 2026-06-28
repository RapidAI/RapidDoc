import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export SWHL table recognition test dataset.")
    parser.add_argument(
        "--dataset",
        default="SWHL/table_rec_test_dataset",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/.table_rec_test_dataset_export",
        help="Directory to save exported images, html files, and manifest.csv.",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    html_dir = output_dir / "html"
    image_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    data = dataset[args.split]

    image_ext = "jpg" if args.image_format == "jpeg" else args.image_format
    manifest_rows = []

    for idx, item in enumerate(data):
        image = item["image"]
        html = item["html"]

        image_path = image_dir / f"{idx:03d}.{image_ext}"
        html_path = html_dir / f"{idx:03d}.html"

        image = image.convert("RGB")
        if image_ext == "jpg":
            image.save(image_path, format="JPEG", quality=95)
        else:
            image.save(image_path, format="PNG")

        html_path.write_text(html, encoding="utf-8")

        manifest_rows.append(
            {
                "idx": idx,
                "image_path": str(image_path),
                "html_path": str(html_path),
                "image_width": image.width,
                "image_height": image.height,
                "html_len": len(html),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Exported {len(manifest_rows)} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
