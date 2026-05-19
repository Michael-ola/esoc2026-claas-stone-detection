from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class ExportedTree:
    """A compact exported decision tree."""

    children_left: list[int]
    children_right: list[int]
    feature: list[int]
    threshold: list[float]
    value: list[float]


@dataclass(frozen=True)
class RandomForestMCUExport:
    """Microcontroller-oriented Random Forest export payload."""

    model_name: str
    feature_names: list[str]
    trees: list[ExportedTree]
    n_classes: int
    class_labels: list[int]
    estimated_model_bytes: int


def export_random_forest_for_mcu(
    model: RandomForestClassifier,
    feature_names: list[str],
    model_name: str = "claas_stone_rf_mcu",
) -> RandomForestMCUExport:
    """Convert a fitted RandomForestClassifier into compact tree arrays."""
    if not hasattr(model, "estimators_"):
        raise ValueError("RandomForestClassifier must be fitted before export.")

    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    trees = [export_tree(estimator.tree_) for estimator in model.estimators_]

    class_labels = [int(label) for label in model.classes_.tolist()]
    n_classes = len(class_labels)

    return RandomForestMCUExport(
        model_name=model_name,
        feature_names=feature_names,
        trees=trees,
        n_classes=n_classes,
        class_labels=class_labels,
        estimated_model_bytes=estimate_export_size_bytes(trees=trees),
    )


def export_tree(tree: Any) -> ExportedTree:
    """Export one sklearn decision tree into portable arrays."""
    values = tree.value.squeeze(axis=1)

    if values.ndim == 1:
        positive_values = values.astype(float)
    else:
        positive_values = values[:, -1].astype(float)

    totals = np.maximum(values.sum(axis=-1), 1.0)
    positive_probability = positive_values / totals

    return ExportedTree(
        children_left=tree.children_left.astype(int).tolist(),
        children_right=tree.children_right.astype(int).tolist(),
        feature=tree.feature.astype(int).tolist(),
        threshold=tree.threshold.astype(float).tolist(),
        value=positive_probability.astype(float).tolist(),
    )


def estimate_export_size_bytes(trees: list[ExportedTree]) -> int:
    """Estimate model-array footprint for the exported forest.

    Assumption per node:
    - int16 left child index
    - int16 right child index
    - int16 feature index
    - float32 threshold
    - float32 positive-class probability

    This estimates only the exported model arrays, not the feature extraction
    buffers or application firmware.
    """
    bytes_per_node = 2 + 2 + 2 + 4 + 4
    node_count = sum(len(tree.feature) for tree in trees)
    return node_count * bytes_per_node


def write_export_json(
    export: RandomForestMCUExport,
    output_path: str | Path,
) -> None:
    """Write exported Random Forest artefact as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": export.model_name,
        "feature_names": export.feature_names,
        "n_classes": export.n_classes,
        "class_labels": export.class_labels,
        "estimated_model_bytes": export.estimated_model_bytes,
        "trees": [
            {
                "children_left": tree.children_left,
                "children_right": tree.children_right,
                "feature": tree.feature,
                "threshold": tree.threshold,
                "value": tree.value,
            }
            for tree in export.trees
        ],
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_c_header(
    export: RandomForestMCUExport,
    output_path: str | Path,
) -> None:
    """Write exported Random Forest as a C header."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    guard = f"{export.model_name.upper()}_H"
    lines: list[str] = []

    lines.extend(
        [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <stdint.h>",
            "",
            "typedef struct {",
            "    int16_t left;",
            "    int16_t right;",
            "    int16_t feature;",
            "    float threshold;",
            "    float value;",
            "} DecisionTreeNode;",
            "",
            f"#define CLAAS_RF_TREE_COUNT {len(export.trees)}",
            f"#define CLAAS_RF_FEATURE_COUNT {len(export.feature_names)}",
            f"#define CLAAS_RF_CLASS_COUNT {export.n_classes}",
            f"#define CLAAS_RF_ESTIMATED_MODEL_BYTES "
            f"{export.estimated_model_bytes}",
            "",
        ]
    )

    for tree_index, tree in enumerate(export.trees):
        lines.append(
            f"static const DecisionTreeNode claas_rf_tree_{tree_index}[] = {{"
        )

        for node_index in range(len(tree.feature)):
            lines.append(
                "    {"
                f"{tree.children_left[node_index]}, "
                f"{tree.children_right[node_index]}, "
                f"{tree.feature[node_index]}, "
                f"{tree.threshold[node_index]:.9g}f, "
                f"{tree.value[node_index]:.9g}f"
                "},"
            )

        lines.append("};")
        lines.append(
            f"static const uint16_t claas_rf_tree_{tree_index}_node_count = "
            f"{len(tree.feature)};"
        )
        lines.append("")

    tree_pointer_values = ", ".join(
        f"claas_rf_tree_{tree_index}"
        for tree_index in range(len(export.trees))
    )
    tree_node_count_values = ", ".join(
        f"claas_rf_tree_{tree_index}_node_count"
        for tree_index in range(len(export.trees))
    )

    lines.extend(
        [
            "static const DecisionTreeNode* claas_rf_trees[] = {",
            f"    {tree_pointer_values}",
            "};",
            "",
            "static const uint16_t claas_rf_tree_node_counts[] = {",
            f"    {tree_node_count_values}",
            "};",
            "",
            "static inline float claas_rf_eval_tree(",
            "    const DecisionTreeNode* nodes,",
            "    const float* features",
            ") {",
            "    int16_t node = 0;",
            "    while (nodes[node].left != -1 && nodes[node].right != -1) {",
            "        int16_t feature = nodes[node].feature;",
            "        if (features[feature] <= nodes[node].threshold) {",
            "            node = nodes[node].left;",
            "        } else {",
            "            node = nodes[node].right;",
            "        }",
            "    }",
            "    return nodes[node].value;",
            "}",
            "",
            "static inline float claas_rf_predict_proba(",
            "    const float* features",
            ") {",
            "    float score = 0.0f;",
            "    for (uint16_t i = 0; i < CLAAS_RF_TREE_COUNT; ++i) {",
            "        score += claas_rf_eval_tree(claas_rf_trees[i], features);",
            "    }",
            "    return score / (float)CLAAS_RF_TREE_COUNT;",
            "}",
            "",
            "static inline uint8_t claas_rf_predict(",
            "    const float* features,",
            "    float threshold",
            ") {",
            "    return claas_rf_predict_proba(features) >= threshold ? 1u : 0u;",
            "}",
            "",
            f"#endif  /* {guard} */",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_list(
    feature_names: list[str],
    output_path: str | Path,
) -> None:
    """Write feature order expected by the MCU artefact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")


def write_deployment_note(
    export: RandomForestMCUExport,
    output_path: str | Path,
    ram_limit_bytes: int = 2 * 1024 * 1024,
) -> None:
    """Write human-readable deployment note for Bonus 2."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    status = "PASS" if export.estimated_model_bytes < ram_limit_bytes else "FAIL"

    text = f"""# Bonus 2 MCU Random Forest Artefact

Chosen deployment model: constrained Random Forest classifier.

## Reason for selection

The constrained Random Forest was selected for the 2 MB inference-only
microcontroller target because it:

- reuses the Task 2 feature pipeline,
- avoids Python, PyTorch, TensorFlow, or dynamic allocation at inference time,
- compiles into plain C arrays and simple if/else tree traversal,
- is easier to audit than a rushed neural-network deployment,
- can be constrained with a small number of trees and limited tree depth.

The larger models developed or prototyped in the project can remain as
experimental models. For the automotive microcontroller scenario, the exported
Random Forest is the deployable choice.

## Generated artefacts

- `claas_stone_rf_mcu.h`: C header containing tree arrays and inference code.
- `claas_stone_rf_mcu.json`: JSON representation of the exported forest.
- `feature_order.txt`: exact feature vector order expected by the model.
- `README.md`: this deployment note.

## Memory estimate

Estimated exported model-array footprint:

```text
{export.estimated_model_bytes} bytes
```

RAM limit:

```text
{ram_limit_bytes} bytes
```

2 MB fit check:

```text
{status}
```

## Important note

This estimate covers the exported Random Forest model arrays only. The feature
extraction buffer and surrounding firmware must also be budgeted in a real ECU
integration. The selected model is intentionally tree-limited and depth-limited
to keep inference simple and memory-bounded.
"""
    path.write_text(text, encoding="utf-8")
