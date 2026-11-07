# Surface Water from Wildlife Camera Images

Workshop notebooks for segmenting river water in wildlife-camera photos with SAM3 and YOLOv8. The example photos and model checkpoints are not stored in this GitHub copy because of their size; download them from the workshop Google Drive folder.

## Get the Data and Models

Open the [workshop Drive folder](https://drive.google.com/drive/folders/1P1X86TbaQOwjSUs52jXv-_IvPvtsM_bE?usp=drive_link). You need permission from the folder owner.

For a **local run**, download the required photos and model files from Drive into the matching folders in this repository. For **Google Colab**, you do not need to download the large assets to your computer: add the shared workshop folder as a shortcut under My Drive, grant Colab access when prompted, and set `PROJECT_ROOT` to that Drive folder in the notebook. Use the complete folder name shown in Drive, or adjust the path if you renamed the shortcut. The shared inputs can remain Viewer-only; the notebooks save generated files to your personal Drive.

GitHub does not track empty directories. Small `.gitkeep` placeholder files preserve the expected asset-folder layout in this repository. Leave them in place or replace them with the downloaded files; do not commit the large photos or model checkpoints to GitHub.

Required layout:

```text
Part1_Module_2_Surface_Water_Dynamics_from_Wildlife_Camera_Imagery/
  water_surface_workshop.py
  SAM3_water_surface_workshop_v2.ipynb
  YOLOv8_water_surface_workshop_v2.ipynb
  sam3_model/
    config.json
    model.safetensors
    processor_config.json
    tokenizer and other model files
  YOLO/
    best.pt
    config_wlc_380pic.yaml
    yolov8n-seg.pt                 # optional; downloaded if missing
  example_photos/
    case1/                         # original camera photos
    case2/                         # camera photos already 640 pixels wide
    wlc_YOLODataset_380pic/
      images/train/
      images/val/
      labels/train/
      labels/val/
```

Keep the dataset next to the `YOLO` folder so the relative paths in `config_wlc_380pic.yaml` resolve. SAM3 can be placed in either `sam3_model/` or `models/sam3_model/` under the project root.

## Notebooks

### SAM3

[SAM3_water_surface_workshop_v2.ipynb](SAM3_water_surface_workshop_v2.ipynb) filters grayscale-like night-vision photos in case1, resizes retained photos to 640 pixels wide, and runs SAM3 with the text prompt `water`. It segments case2, selects the prediction with the largest water area, and lets the user draw a polygon reference. It then calculates equal-row water fractions and plots them over time.

The SAM3 notebook demonstrates inference, not model training. Its local checkpoint directory must contain the complete SAM3 model, processor, and tokenizer files.

### YOLOv8

[YOLOv8_water_surface_workshop_v2.ipynb](YOLOv8_water_surface_workshop_v2.ipynb) demonstrates a short 3-epoch training run on the labeled YOLO dataset. It then uses the supplied `YOLO/best.pt` for case1 and case2 inference, draws a polygon reference, and calculates the same water-fraction outputs. The demo-trained checkpoint is not used for inference.

## Shared Helper

[water_surface_workshop.py](water_surface_workshop.py) provides shared image loading, grayscale-like photo detection, 640-wide preprocessing, mask overlays, filename timestamp parsing, row-wise reference fractions, CSV exports, and time-series plotting. The notebooks pass a personal output directory to case1 preprocessing so shared originals remain unchanged.

## Setup Guides

- [INSTALL_DEPENDENCIES.md](INSTALL_DEPENDENCIES.md) describes the local Conda environments, CUDA-enabled PyTorch versions, and notebook dependencies. TensorFlow is not required.
- [COLAB_QUICK_START.md](COLAB_QUICK_START.md) walks through adding a Drive shortcut, opening a notebook in Colab, enabling a GPU, and finding results in personal Drive.

## Outputs

In Colab, generated files are saved in the participant's personal Drive:

```text
MyDrive/Workshop_Results/sam3/
MyDrive/Workshop_Results/yolov8/
```

These folders contain case1 preprocessed photos, masks, overlays, summaries, polygon references, CSV tables, training logs (YOLOv8), and plots. The notebooks do not modify the shared input photos or checkpoints.

SAM3 and YOLOv8 require an NVIDIA GPU for practical training/inference performance. Colab GPU availability and local GPU driver compatibility can vary. Check the setup guides before the workshop. The YOLOv8 training example is only three epochs and is not a converged model or an accuracy claim.