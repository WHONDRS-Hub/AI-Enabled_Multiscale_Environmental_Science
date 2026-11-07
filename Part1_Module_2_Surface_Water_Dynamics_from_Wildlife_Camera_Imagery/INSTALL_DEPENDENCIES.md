# Install Workshop Dependencies Locally
# For Google Colab, use [COLAB_QUICK_START.md](COLAB_QUICK_START.md) instead. Colab already supplies a GPU driver and PyTorch; do not run the local CUDA/PyTorch installation commands there.
This guide covers the YOLOv8 and SAM3 v2 workshop notebooks.

Use two Conda environments to keep their PyTorch versions separate. Do not install these packages into Conda's `base` environment.



## 1. Requirements

| Component | YOLOv8 workshop | SAM3 workshop |
| --- | --- | --- |
| Python for a new environment | 3.12 recommended | 3.12 |
| PyTorch | 2.8.0, CUDA 12.8 build | 2.10.0, CUDA 12.8 build |
| Matching torchvision | 0.23.0 | 0.25.0 |
| Model package | `ultralytics==8.3.233` | `transformers==5.9.0` |
| Matplotlib | 3.9 or newer | 3.9 or newer |
| Interactive drawing | `ipympl`, `ipywidgets` | `ipympl`, `ipywidgets` |

The verified local YOLOv8 environment uses Python 3.9.25, PyTorch 2.8.0+cu128, and torchvision 0.23.0+cu128 on Windows. The clean-install instructions below recommend Python 3.12 instead of creating another older Python environment.

The verified SAM3 environment uses Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Transformers 5.9.0, and NumPy 1.26.4 inside the supplied Ubuntu Docker image. Fresh environments created from this guide have not been independently tested on every operating system.

### Hardware

- An NVIDIA CUDA-capable GPU and a compatible NVIDIA driver are needed for GPU acceleration.
- The small YOLOv8 training demo uses batch 4; reduce it if memory is insufficient.
- Allow several tens of GB of free disk space for two environments, checkpoints, data, and outputs. A complete exported SAM3 folder can occupy approximately 6.4 GiB.

### CUDA, Drivers, and cuDNN

The commands below install PyTorch's **CUDA 12.8 wheels** from the `cu128` package index. These packages provide the CUDA runtime libraries and cuDNN needed by the supported prebuilt PyTorch operations. A separate CUDA Toolkit or manually installed cuDNN is normally unnecessary for these notebooks.

You still need a sufficiently recent NVIDIA driver compatible with the CUDA 12.8 build and your GPU. Update it using the official NVIDIA installer if needed. Driver installation is separate from Conda.

Check that the driver can see the GPU:

```bash
nvidia-smi
```

The CUDA version displayed by `nvidia-smi` describes the driver's CUDA support, not the CUDA runtime installed in your Python environment. Check `torch.version.cuda` after installing PyTorch. The actual GPU test is `torch.cuda.is_available()`.

Install the full CUDA Toolkit only if you later compile custom CUDA extensions or build packages from source. Those builds may require additional compiler/version matching beyond this guide.

## 2. Install Conda

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or an existing Anaconda distribution appropriate for your operating system.

On Windows, open **Anaconda Prompt** or **Anaconda PowerShell Prompt**. On Linux/WSL2, open a terminal with Conda initialized. Confirm that Conda works:

```bash
conda --version
```

Run the commands below one at a time. If an environment with the proposed name already exists, activate it rather than recreating or overwriting it.

## 3. Create the YOLOv8 Environment

### Create and activate

```bash
conda create --name yolov8_workshop python=3.12 pip -y
conda activate yolov8_workshop
python -m pip install --upgrade pip
```

### Install CUDA-enabled PyTorch

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Install PyTorch before the remaining packages so Ultralytics uses the intended CUDA build.

### Install notebook and model dependencies

```bash
python -m pip install ultralytics==8.3.233 numpy pandas pillow pyyaml "matplotlib>=3.9" jupyterlab ipykernel nbformat nbclient
python -m pip install --upgrade ipympl ipywidgets
python -m pip check
```

Ultralytics also installs its own dependencies, including OpenCV and SciPy. You do not need a separate package named `yolov8`, or the YOLO26 model family, for this notebook.

### Register the kernel

```bash
python -m ipykernel install --user --name yolov8_workshop --display-name "Python (yolov8_workshop)"
```

In [YOLOv8_water_surface_workshop_v2.ipynb](YOLOv8_water_surface_workshop_v2.ipynb), select **Python (yolov8_workshop)** as the kernel.

## 4. Create the SAM3 Environment

The tested SAM3 setup is Linux inside Docker. WSL2/Linux is another practical local option. The following Conda/pip commands describe a fresh Python environment; they do not install Docker, an NVIDIA driver, or GPU passthrough. Native-Windows SAM3 execution has not been validated for this workshop.

### Create and activate

```bash
conda deactivate
conda create --name sam3_workshop python=3.12 pip -y
conda activate sam3_workshop
python -m pip install --upgrade pip
```

### Install CUDA-enabled PyTorch

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

### Install notebook and model dependencies

```bash
python -m pip install transformers==5.9.0 accelerate huggingface_hub "numpy==1.26.4" pandas pillow "matplotlib>=3.9" jupyterlab ipykernel nbformat nbclient
python -m pip install --upgrade ipympl ipywidgets
python -m pip check
```

Transformers installs its supporting dependencies, such as tokenization and safetensors packages.

### Register the kernel

```bash
python -m ipykernel install --user --name sam3_workshop --display-name "Python (sam3_workshop)"
```

Select **Python (sam3_workshop)** for [SAM3_water_surface_workshop_v2.ipynb](SAM3_water_surface_workshop_v2.ipynb).

Jupyter and its kernel must run inside the Linux/container environment to use that environment's packages and GPU. A Windows Python kernel cannot directly use a Conda environment installed inside a container. See [README.md](README.md) for the existing Docker launch instructions.

## 5. Verify the Active Environment

Run these checks while the desired environment is activated:

```bash
python -c "import sys, torch, torchvision; print('Python:', sys.executable); print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

Expected: `GPU available: True`, a CUDA runtime of `12.8`, and your GPU name.

For YOLOv8:

```bash
python -c "import ultralytics; from ultralytics import YOLO; print('Ultralytics:', ultralytics.__version__)"
```

For SAM3:

```bash
python -c "import transformers; from transformers import Sam3Model, Sam3Processor; print('Transformers:', transformers.__version__)"
```

Check polygon-drawing support:

```bash
python -c "import matplotlib, ipympl, ipywidgets; print('Matplotlib:', matplotlib.__version__); print('ipympl:', ipympl.__version__); print('ipywidgets:', ipywidgets.__version__)"
```

If GPU availability is false, check the selected environment, NVIDIA driver, CUDA-enabled PyTorch build, and any WSL2/container GPU configuration before running training. Installing TensorFlow will not fix PyTorch GPU access.

## 6. Start Jupyter and Open the Notebook

Activate the desired environment and navigate to the workshop folder. For example, on Windows:

```powershell
conda activate yolov8_workshop
cd "AGU26_workshop"
jupyter lab
```

Open the notebook and select its corresponding kernel. Run the cells in order. If you install or upgrade packages in a running notebook, restart its kernel/session before continuing.

The polygon cell uses `%matplotlib widget`. If drawing fails, upgrade `ipympl` and `ipywidgets` in the same environment as the notebook kernel, restart the kernel, and rerun setup. Do not substitute `%matplotlib inline` for drawing: it produces a static image without polygon clicks.

## 7. Files Required Besides Python Packages

- Keep [water_surface_workshop.py](water_surface_workshop.py) with the two v2 notebooks. It provides preprocessing and water-fraction calculations.
- YOLOv8 inference requires `best.pt` in the `YOLO` folder. Demo training uses `yolov8n-seg.pt` and [YOLO/config_wlc_380pic.yaml](YOLO/config_wlc_380pic.yaml).
- SAM3 inference requires the complete local model folder, including its configuration, weights, and processor/tokenizer files. The v2 notebook supports either `sam3_model` or `models/sam3_model` under the project root.
- Keep `example_photos` and its case1, case2, and labeled YOLO dataset folders in their expected layout.
- When using the complete local SAM3 folder, no Hugging Face login is needed. Downloading gated weights from Hugging Face requires your own authorized account and acceptance of the model conditions.

Package installation does not download the workshop dataset or grant access to restricted model checkpoints.


## References

- [Official PyTorch version pairs and CUDA wheel commands](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)
- [Ultralytics installation guide](https://docs.ultralytics.com/quickstart/)
- [Transformers installation guide](https://huggingface.co/docs/transformers/installation)
- [ipympl installation and compatibility](https://matplotlib.org/ipympl/installing.html)