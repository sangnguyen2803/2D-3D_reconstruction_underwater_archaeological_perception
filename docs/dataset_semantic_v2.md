# Mermaid semantic dataset card v2

**State:** local proposals and annotation tasks prepared; no reviewed hand labels yet. Dataset-derived images and masks remain in ignored local output directories.

## Source and partition

The source is Avanthey and Beaudoin's [Mermaid dataset, DOI 10.17882/97987](https://doi.org/10.17882/97987). The original image inventory and reference-pose limitations remain in [dataset.md](dataset.md). v2 reuses the v1 camera-center partition: 226 train, 65 validation, 743 test and 210 excluded reference views, with a gap of 0.4472 reference units. Camera-center separation cannot establish disjoint visible surfaces.

| Set | Prepared | Reviewed | Purpose |
| --- | --- | --- | --- |
| Spatial audit | 150 images, 50 per active slab | 0 | Class separability and dynamic-object review |
| Pseudo train | 226 manifest placeholders | Not applicable | Generated only after a human gate pass |
| Hand validation | 20 images and teacher suggestions | 0 | Selection, calibration and pseudo-label quality audit |
| Hand test | 40 images and teacher suggestions | 0 | Evaluation only |
| Repeat annotation | 8 validation tasks | 0 | Same-annotator consistency after at least 48 hours |

The manifest is `outputs/semantic_labels_v2/manifest.json`, currently unfrozen. It records image IDs, role, spatial slab, source path, mask path, review status, annotator, UTC date, pass and hash. Frozen masks are verified by content hash each time a training/evaluation dataset is opened.

## Classes

| ID | Class | Definition |
| --- | --- | --- |
| 0 | Sand / sediment | Seabed sediment; avoid treating every unclear green region as sand |
| 1 | Rock | Fixed rocky bottom |
| 2 | Statue | Mermaid statue and its fixed base |
| 3 | Tile / artifacts | Recognizable archaeological tile fragments/artifacts |
| 4 | Sessile life | Attached algae, seagrass, sponge or other attached organism |
| 5 | Dynamic | Fish, divers and moving/nonrigid equipment |
| 255 | Ignore | Water column, ambiguous, uncertain boundaries and unlabeled pixels |

The six-class schema is proposed until Gate 0 is reviewed. Merge decisions are explicit in `configs/semantic/labels.yaml`. An ignored class is removed from losses/metrics. Zero teacher coverage is not proof a class is absent from the scene.

## Teacher procedure and noise

Pinned checkpoints: Grounding DINO tiny `a2bb814dd30d776dcf7e30523b00659f4f141c71` and SAM ViT-B `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`. Prompts are sand, rock, statue, tile, algae, fish, diver and equipment. A detected phrase must identify exactly one schema class. Fish/diver/equipment share dynamic.

Detection thresholds are box score 0.25 and text score 0.20. SAM masks require an IoU-head score of at least 0.70. Each accepted mask receives box score times clipped SAM score; conflicting classes within a margin of 0.03 and pixels below 0.15 combined score become ignore. At most 24 proposals/image are retained. These scores are not calibrated semantic probabilities.

The initial audit shows very low rock/statue/tile coverage, substantial ignore coverage and a large class-mixture difference between train and test. Text grounding and underwater color cast can produce false detections. Foundation outputs must be reviewed before detector eligibility or class merge decisions. Evaluation of teacher/student quality requires hand masks; no pseudo-label mIoU is reported as test accuracy.

See [annotation_guide_v2.md](annotation_guide_v2.md) for the same-annotator, SAM-assisted correction protocol. Intra-annotator IoU remains pending. A single scene and single annotator limit generalization and uncertainty estimates.

## License and distribution

The data license recorded by the project is CC BY-NC-ND 4.0. Its [legal code](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en) distinguishes producing adapted material for noncommercial use from sharing it. Whether a particular annotation mask constitutes adapted material needs case-specific assessment; this project keeps all image-derived labels local and does not publish them.

[SAM](https://huggingface.co/facebook/sam-vit-base) and [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) model cards declare Apache-2.0. [DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) uses Apache-2.0. [MIT-B0's model card](https://huggingface.co/nvidia/mit-b0) links to the [NVIDIA SegFormer source license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE), whose use limitation is research/evaluation only for non-NVIDIA users. Optional [Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) uses AGPL-3.0. Repository MIT code licensing does not replace these data/model terms.
