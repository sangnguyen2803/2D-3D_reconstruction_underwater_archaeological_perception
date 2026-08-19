# Project status v1

Reference gate, learned match filtering, full-dataset DINO retrieval and learned sparse confidence are implemented and evaluated.

- Reference audit: 0.2934px median, 94.14% below 1.5px.
- Learned context: AP 0.9459 / ROC AUC 0.9722 on the spatial development holdout.
- Combined SfM: 16 cameras / 1,912 sparse points / 0.4474px reprojection error.
- Retrieval: 1,244 images; DINO recall 9.04% vs sequential 10.08% on the default overlap proxy.
- Learned 3D AUSE: 0.4910 vs heuristic 0.4194; uncertainty calibration remains unsuccessful.
- Local tests: 35 passed.

Read [PROJECT_SUMMARY_v1.md](PROJECT_SUMMARY_v1.md) for methods, before/after tables, limitations and commands. Original summaries/results are preserved.
