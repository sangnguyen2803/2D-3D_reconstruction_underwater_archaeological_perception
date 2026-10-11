# Controlled calibration audit v1

This follow-up tests whether the reported calibration improvement survives a matched validation-data budget. Both temperature scaling and positive-slope Platt calibration fit the exact same saved logits from 64 validation images per spatial fold/seed. We reuse the frozen nine checkpoints and their full original held-out test populations. Test labels never fit either calibrator.

| Macro-average over three spatial folds and three seeds | Temperature on 64 validation images | Platt on the same 64 images |
|---|---:|---:|
| ECE, 10 bins | 10.60% | 3.50% |
| ECE, 15 bins | 10.71% | 3.57% |
| ECE, 20 bins | 10.71% | 3.60% |
| Brier | 0.181301 | 0.165956 |
| Log-loss | 0.545001 | 0.507385 |

Matched-budget mean ECE reduction: **67.00%**. The earlier 67.83% reduction compares Platt against the preserved full-validation temperature baseline; this audit gives a more controlled comparison.

ECE improves in 8/9 runs at 10 bins. Brier and log-loss improve in 6/9 runs: all three seeds on folds 1 and 2, while all three seeds on fold 0 become slightly worse on these proper scoring rules. The benefit therefore varies by spatial region. Score ordering and AP remain effectively unchanged; this is a probability calibration improvement, not a matching-ranking improvement.

Paired 95% bootstrap intervals for after minus before:

- ece: [-0.088309, -0.053362].
- brier: [-0.019286, -0.011559].
- log_loss: [-0.046924, -0.028242].

All three intervals lie below zero. These are 500 paired image-cluster bootstrap draws with identical draws across seeds within each fold, averaged over fixed folds. They are conditional evidence within one acquisition; they do not establish unseen-scene generalization. This is a post-hoc robustness audit of already examined held-out regions, not a new blind benchmark.

No additional network training, checkpoint changes, test-based calibration fitting, SfM/MVS reconstruction, commit or push occurred. Mesh trimming remains a separate observed-support cleanup result, not independently measured surface accuracy.

Evidence: [controlled metrics](outputs/calibration_subset64_v1/controlled_audit_v1.json). Earlier reports remain unchanged.

Conservative CV wording:

```latex
\item Improved \textbf{SegFormer-B0 confidence calibration}, reducing mean ECE from \textbf{10.60\% to 3.50\%} across \textbf{three spatial folds and three seeds} using validation-only Platt scaling.
```
