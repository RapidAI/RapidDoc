"""Export shared Python candidates and evidence for Rust parity replay."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
from loguru import logger

from rapid_doc.model.table.fallback import grid_evidence, quality, structure_evidence
from rapid_doc.model.table.rapid_table import ModelType, RapidTableModel, _TableMarkup
from rapid_doc.model.table.table_evidence import TableEvidence, _raster_lines


def export(root, output):
    logger.remove()
    cases = []
    for folder in ("three-layer-v3-verified", "two-pdf-review"):
        for path in sorted((root / folder).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf8"))
            image_folder = root / folder
            if folder == "three-layer-v3-verified":
                image_folder = root / ("three-layer-holdout-v2" if path.stem.startswith("holdout") else "three-layer")
            for index, item in enumerate(data["tables"]):
                image_path = image_folder / f"{path.stem}-{index}.png"
                image = cv2.imread(str(image_path))
                rows = item["rows"]
                source = TableEvidence(image, rows, vector_lines=item["vector"])
                predictions, summaries = [], []
                for j, name in enumerate(("main", "tatr")):
                    grid = item["grids"][j]
                    evidence_grid = (item.get("evidence_grids") or [None, None])[j] if j == 0 else None
                    if evidence_grid is None:
                        evidence_grid = grid
                    tokens = item["tokens"][j]
                    tokens = {int(k): v for k, v in tokens.items()} if tokens is not None else None
                    result = _TableMarkup(item[name], SimpleNamespace(
                        cell_bboxes=[grid[0]], logic_points=[grid[1]],
                        token_cells=[tokens], evidence_grids=[evidence_grid],
                    )) if grid and item[name] else item[name]
                    predictions.append(result)
                    score = source.score(evidence_grid, tokens)
                    old = grid_evidence(grid, rows)
                    summaries.append(dict(
                        quality=vars(quality(item[name], rows)),
                        structure=vars(structure_evidence(item[name], rows)),
                        score=score.summary() if score else None,
                        missing=sorted(score.missing) if score else [],
                        missing_numbers=sorted(score.missing_numbers) if score else [],
                        conflicts=sorted(old.conflicts) if old else [],
                        supported=sorted(old.supported) if old else [],
                        evidence_grid=evidence_grid, tokens=tokens,
                    ))
                model = RapidTableModel.__new__(RapidTableModel)
                model.model_type = ModelType.UNET_SLANET_PLUS
                model.fallback_model_type = "table_transformer"
                model.fallback_to_text = False
                model.fallback_text_loss_threshold = .2
                model._structure = Mock(return_value=predictions[0])
                model._fallback_model = Mock()
                model._fallback_model._structure.return_value = predictions[1]
                selected = model.predict(image[:, :, ::-1], rows, vector_lines=item["vector"])
                cases.append(dict(
                    id=f"{path.stem}-{index}", image=str(image_path.resolve()), rows=rows,
                    vector=item["vector"], raster=list(_raster_lines(image, source.em)),
                    lines=source.lines, grids=item["grids"], main=item["main"], tatr=item["tatr"],
                    candidates=summaries, triggered=model._fallback_model._structure.called,
                    selected="main" if selected == item["main"] else "tatr",
                ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()), encoding="utf8")
    print(f"Exported {len(cases)} tables to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(os.environ["TEMP"]) / "rapiddoc-table-audit-20261113")
    parser.add_argument("--output", type=Path, default=Path("build/migration/table-evidence-sync.json"))
    args = parser.parse_args()
    export(args.source, args.output)
