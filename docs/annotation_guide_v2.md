# Human review and annotation workflow v2

The local editor is available at **http://127.0.0.1:8010** while `serve_label_review.py` is running. Nothing is uploaded. Restart it from the repository with:

```powershell
.\.venv\Scripts\python.exe scripts/serve_label_review.py
```

## Gate 0 first

Select **Gate 0 audit**. Enter your reviewer name. For each of the 150 images:

1. Inspect the original RGB image and toggle the teacher overlay opacity.
2. Count real dynamic instances, including missed fish/divers/equipment. Teacher proposals are suggestions, not confirmed instances. Equipment qualifies only when it is moving or likely nonrigid; fixed archaeological objects do not.
3. Record whether the central classes can be distinguished by eye. Use **No / uncertain** and a note if they cannot.
4. Save the human audit. Repeat for all images.

Reviews are stored in `outputs/semantic_audit_v2/human_review.csv`. Three teacher classes (rock, statue and tile/artifacts) currently occupy less than 1% of sampled pixels. This may reflect teacher failure rather than true rarity. A reviewer must explicitly decide which to merge/ignore and which to retain with corrected labeling. Do not merge them just to improve a metric.

Finalize with `scripts/finalize_semantic_protocol.py gate --merge-map '<JSON>' --class-decisions '<JSON>'`. JSON keys are class IDs. `--merge-map` maps an ID to another ID or 255. `--class-decisions` records a written rationale for each rare class that is retained. The script refuses unresolved rare-class decisions or incomplete reviews. It saves a new final decision and freezes criteria only if class separability passes. A failed gate prevents training.

## Hand masks

After the gate, select **Hand labels**. There are 20 validation and 40 test tasks. The test tasks carry three descriptive sub-block IDs; these do not count as three independently retrained folds.

- Choose a class and brush size. Paint only pixels you can identify.
- Set uncertain boundaries, water column and unknown pixels to **255 Ignore**.
- **Load teacher suggestion** initializes an editable proposal. Inspect and correct the whole image before saving; clicking this button never makes a label reviewed.
- Include the mermaid statue and its fixed base in statue. Tile fragments belong to artifacts. Attached algae/sponges/seagrass belong to sessile life. Fish/divers belong to dynamic.
- Save with the same annotator name throughout. The manifest records UTC date, pass and mask SHA-256.

Masks are single-channel PNGs at 960x720 using IDs 0..5 and 255. The editor writes masks locally under `outputs/semantic_labels_v2/hand_masks`; suggestions remain separate under `suggestions`.

## Repeat consistency and freeze

The eight repeat tasks are validation images. Wait **at least 48 hours after pass 1**. Select **Blind repeat**, which starts from ignore and disables teacher suggestions. Do not consult the first-pass mask. Save with the same annotator name. The server enforces the delay.

After a passed gate, generate train-slab pseudo masks:

```powershell
.\.venv\Scripts\python.exe scripts/make_pseudo_labels.py
```

When all 60 hand masks, eight delayed repeat masks and train pseudo masks exist:

```powershell
.\.venv\Scripts\python.exe scripts/finalize_semantic_protocol.py labels
```

This computes intra-annotator IoU, verifies hashes and freezes the manifest. Later changes need a new annotation version. Final evaluation accepts reviewed hand masks only, and test masks never tune temperature, dynamic thresholds or model selection.

The separate 3-fold rotation needs fold-specific manifests and sufficient hand-label coverage in every rotated validation/test region. The original 60 tasks do not automatically provide that coverage. All students, match filters and point learners must be retrained for each fold.
