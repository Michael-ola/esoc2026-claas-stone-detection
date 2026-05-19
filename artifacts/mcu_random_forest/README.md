# Bonus 2 MCU Random Forest Artefact

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
11536 bytes
```

RAM limit:

```text
2097152 bytes
```

2 MB fit check:

```text
PASS
```

## Important note

This estimate covers the exported Random Forest model arrays only. The feature
extraction buffer and surrounding firmware must also be budgeted in a real ECU
integration. The selected model is intentionally tree-limited and depth-limited
to keep inference simple and memory-bounded.
