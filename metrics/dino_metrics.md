# ReID Baseline Metrics Report

Generated: 2026-03-25T10:37:55

## Run Summary

- DINOv2 variant: `vits14`
- Device: `cpu`
- References evaluated: 29
- Positive query cases: 182
- Negative query cases: 166
- Overall Top-1 Accuracy: 0.4529
- Overall mAP: 0.6918

## Best References by mAP

| Reference Image | Label | Positive Queries | Top-1 | mAP |
|---|---:|---:|---:|---:|
| Yellow_ball_6_9_IMG.png | yball06 | 7 | 0.7500 | 0.9048 |
| Ball_10_6_9_IMG.png | yball06 | 7 | 0.6875 | 0.8571 |
| Ball_2_IMG.png | yball02 | 6 | 0.6667 | 0.7917 |
| Ball_6_IMG.png | yball06 | 7 | 0.7083 | 0.7857 |
| Yellow_ball_6_9_2_IMG.png | yball06 | 7 | 0.5000 | 0.7857 |
| All_balls_IMG2.png | yball06 | 7 | 0.3125 | 0.7619 |
| Yellow_ball_6_10_IMG.png | yball06 | 7 | 0.4167 | 0.7262 |
| All_balls_IMG1.png | yball09 | 6 | 0.6111 | 0.7222 |
| Ball_10_6_9_IMG.png | yball09 | 6 | 0.4444 | 0.7222 |
| Yellow_ball_2_9_IMG.png | yball02 | 6 | 0.4167 | 0.7222 |

## Hardest References by mAP

| Reference Image | Label | Positive Queries | Top-1 | mAP |
|---|---:|---:|---:|---:|
| Ball_9_IMG.png | yball09 | 6 | 0.0833 | 0.4583 |
| Ball_10_IMG.png | yball10 | 6 | 0.0833 | 0.5278 |
| Yellow_ball_2_10_IMG.png | yball02 | 6 | 0.3333 | 0.5694 |
| Ball_10_6_9_IMG.png | yball10 | 6 | 0.2083 | 0.5833 |
| Yellow_ball_6_10_IMG.png | yball10 | 6 | 0.1667 | 0.6111 |
| All_balls_IMG1.png | yball10 | 6 | 0.2222 | 0.6389 |
| All_balls_IMG2.png | yball10 | 6 | 0.2222 | 0.6389 |
| Yellow_Ball_10_2_6.png | yball10 | 6 | 0.2778 | 0.6389 |
| Yellow_ball_2_10_IMG.png | yball10 | 6 | 0.2222 | 0.6389 |
| All_balls_IMG2.png | yball09 | 6 | 0.3889 | 0.6806 |

## Output Files

- `metrics_summary.json` — overall run summary
- `metrics_per_reference.csv` — one row per reference object
- `metrics_per_query.csv` — one row per evaluated query pair
- `metrics_report.md` — this report

## Notes

- Top-1 and mAP are computed with TorchMetrics.
- The per-query AP column is a simple reciprocal-rank style debug value for quick inspection.
- Negative query cases are logged but skipped for positive-match metrics.