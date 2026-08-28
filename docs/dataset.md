# Dataset provenance and conventions

Official record: https://www.seanoe.org/data/00868/97987/
DOI: https://doi.org/10.17882/97987
Authors: Loïca Avanthey and Laurent Beaudoin, 2023.

| Asset | Role |
| --- | --- |
| 107174.pdf | Image acquisition, sensor, auto-calibration, poses overview |
| 107175.pdf | Measured micro-geodesic reference network and GCP uncertainties |
| 107176.stl | Relative representation of the control network |
| 107177.xml | Camera transforms and covariances estimated by multi-view BA |
| 107179.zip | Original RGB JPEG imagery |

The EPITA overview reports 1,250 images, 3840×2800 and approximately 15 GB. The image
PDF reports 1,244 images at 3840×2880 and 3.69 GB. The archive itself may contain more
images than poses. Use the generated `outputs/dataset_inventory.json` as the local
inventory, and retain these discrepancies rather than silently rewriting provenance.

Calibration transcribed from page 2 of 107174.pdf:
f=2334.29 px; cx=-12.752 and cy=-16.6962 interpreted as offsets from the image center;
k1=-0.222446, k2=0.310621, k3=-0.0835057;
p1=-0.000995472, p2=-0.000078498. The source calls this auto-calibration.

Reference XML labels omit the image extension. The loader associates labels with
image filename stems and preserves missing reference poses. Transforms are interpreted
as camera-to-world. Rotation and homogeneous-transform validity are checked.

The GCP table and accompanying diagram contain apparent transcription discrepancies
for some coordinates. Automated absolute GCP evaluation is intentionally not enabled
without verified image observations and a resolved coordinate convention.
