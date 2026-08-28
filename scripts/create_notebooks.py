"""Generate English analysis notebooks that read existing local artifacts."""

from pathlib import Path

import nbformat as nbf


def main():
    root = Path("notebooks")
    root.mkdir(exist_ok=True)
    setup = "from pathlib import Path\nimport json\nimport numpy as np\nimport matplotlib.pyplot as plt\nPROJECT = Path.cwd()\nif PROJECT.name == 'notebooks': PROJECT = PROJECT.parent"
    notebooks = [
        (
            "01_dataset_exploration",
            "Dataset provenance and actual image inventory",
            "from underwater_vision.data.dataset_loader import MermaidDataset\nfrom underwater_vision.utils.io import read_rgb\ndataset = MermaidDataset(PROJECT/'data/raw/mermaid')\nprint('Images:', len(dataset.frames), 'BA reference poses:', len(dataset.poses))\nframe = dataset.frames[0]\nimage = read_rgb(frame.path)\nprint(frame.name, image.shape)\nplt.imshow(image); plt.axis('off')",
            "poses = np.array([f.reference_c2w[:3, 3] for f in dataset.frames if f.reference_c2w is not None])\nfig = plt.figure(); ax = fig.add_subplot(projection='3d'); ax.plot(*poses.T); ax.set_title('Published BA reference camera centers')",
        ),
        (
            "02_visibility_analysis",
            "Handcrafted evidence quality under blur",
            "import cv2\nfrom underwater_vision.visibility.deterministic import estimate_visibility\nfrom underwater_vision.utils.io import read_rgb\nfrom underwater_vision.data.dataset_loader import MermaidDataset\nfrom underwater_vision.preprocessing.images import resize_rgb\nimage, _ = resize_rgb(read_rgb(MermaidDataset(PROJECT/'data/raw/mermaid').frames[0].path))\nsharp, cues = estimate_visibility(image)\nblurred, _ = estimate_visibility(cv2.GaussianBlur(image, (25,25), 6))\nfig, ax = plt.subplots(1,3, figsize=(15,4))\nax[0].imshow(image); ax[1].imshow(sharp,vmin=0,vmax=1); ax[2].imshow(blurred,vmin=0,vmax=1)\nprint('Mean scores:', sharp.mean(), blurred.mean())",
            "plt.hist(sharp.ravel(), bins=50, alpha=.6, label='Original'); plt.hist(blurred.ravel(),bins=50,alpha=.6,label='Blurred'); plt.legend(); plt.title('Reliability sensitivity')",
        ),
        (
            "03_feature_matching",
            "Correspondences and geometric failure cases",
            "run = PROJECT/'outputs/visibility_eval'\npairs = json.loads((run/'pair_metrics.json').read_text())\nimport pandas as pd\ntable = pd.DataFrame(pairs)\ndisplay(table.describe())\ndisplay(table.sort_values('inlier_ratio').head(10))",
            "plt.scatter(table['matches'],table['inlier_ratio']); plt.xlabel('Candidate matches'); plt.ylabel('Self-verified inlier ratio'); plt.title('Filtering does not measure GT match precision')",
        ),
        (
            "04_reconstruction_analysis",
            "Sparse 3D confidence and reference trajectory agreement",
            "run = PROJECT/'outputs/visibility_eval'\ncloud = np.load(run/'sparse_confidence.npz')\nfig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(projection='3d')\nax.scatter(*cloud['xyz'].T,c=cloud['confidence'],cmap='viridis',s=1,vmin=0,vmax=1); ax.set_title('Heuristic point confidence')",
            "metrics = json.loads((run/'metrics.json').read_text()); display(metrics)\nplt.hist(cloud['confidence'],bins=40); plt.xlabel('Confidence evidence score'); plt.ylabel('Point count')",
        ),
        (
            "05_results",
            "Measured ablations and limitations",
            "import pandas as pd\nrows = []\nfor name in ['classical_eval','neural_eval','visibility_eval','consistency_eval','learned_eval','small_eval']:\n path=PROJECT/'outputs'/name/'metrics.json'\n if path.exists(): rows.append({'run':name,**json.loads(path.read_text())})\ntable=pd.DataFrame(rows); display(table[['run','registered_images','sparse_points','inlier_ratio','mean_reprojection_error_px']])",
            "table.plot.bar(x='run',y=['sparse_points'],legend=False); plt.title('Point support is a coverage proxy, not dense completeness')",
        ),
    ]
    for filename, title, cell1, cell2 in notebooks:
        cell1 = cell1.replace("outputs/visibility_eval", "outputs/visibility_gpu_eval")
        cell1 = cell1.replace(
            "'neural_eval','visibility_eval','consistency_eval'",
            "'neural_gpu_eval','visibility_gpu_eval','consistency_gpu_eval'",
        )
        book = nbf.v4.new_notebook()
        book.metadata["kernelspec"] = {
            "display_name": "Python 3 (underwater-vision)",
            "language": "python",
            "name": "underwater-vision",
        }
        book.cells = [
            nbf.v4.new_markdown_cell(
                "# "
                + title
                + "\n\nRead actual downloaded data and completed experiments. No results are fabricated."
            ),
            nbf.v4.new_code_cell(setup),
            nbf.v4.new_code_cell(cell1),
            nbf.v4.new_code_cell(cell2),
            nbf.v4.new_markdown_cell(
                "## Interpretation\n\nCamera poses are published bundle-adjustment references. Dense surface ground truth is unavailable. Assess coverage and reprojection together; confidence is not a calibrated probability."
            ),
        ]
        nbf.write(book, root / (filename + ".ipynb"))


if __name__ == "__main__":
    main()
