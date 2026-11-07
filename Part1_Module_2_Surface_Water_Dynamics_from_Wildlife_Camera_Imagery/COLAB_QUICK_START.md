# Run the Workshop in Google Colab

You need a Google account and access to the instructor's shared Google Drive folder. You do not need to download the images or models to your computer. Colab runs the code on Google's computers.

## 1. Add the Shared Folder to My Drive

1. Open the shared Google Drive link provided by the instructor.
2. Find **Part 1, Module 2: Surface Water Dynamics from Wildlife Camera Imagery**.
3. Right-click that folder and choose **Organize > Add shortcut**. Depending on the Drive interface, this may be called **Add shortcut to Drive**.
4. Select **My Drive**, then click **Add**.

Keep the shortcut directly under My Drive and do not rename it. A shortcut gives access to the shared files without copying the approximately 7 GB of data. Viewer access to the shared folder is sufficient; your results will be saved separately.

## 2. Open the Notebook

1. Open the shared folder and find either workshop notebook:
   - [YOLOv8_water_surface_workshop_v2.ipynb](YOLOv8_water_surface_workshop_v2.ipynb)
   - [SAM3_water_surface_workshop_v2.ipynb](SAM3_water_surface_workshop_v2.ipynb)
2. Right-click the notebook and choose **Open with > Google Colaboratory**.
3. If Colaboratory is missing, choose **Connect more apps**, search for **Colaboratory**, and connect it.
4. In Colab, choose **File > Save a copy in Drive**.
5. Continue working in that personal copy.

Only the small notebook is copied, not the images or models. The links above identify the files; open the instructor's shared Drive copies to start in Colab.

## 3. Enable a GPU

In Colab, choose **Runtime > Change runtime type**. Select **T4 GPU**, or another available GPU, and click **Save**.

GPU availability is subject to Colab's usage limits. Choose the GPU before running the notebook because changing runtimes can clear Python variables.

## 4. Run the Setup Cell

Click the **Run** button (the triangle icon) beside the first code cell. It installs the required packages.

If the notebook asks you to restart, choose **Runtime > Restart session**, then run the setup cell again. Some interfaces call this **Restart runtime**. Do not continue to training or inference until setup succeeds.

When asked to connect Google Drive, select your account and authorize access. This allows Colab to read the shared shortcut and save results in your personal Drive. Google Drive authorization is separate from the notebook's package installation.

## 5. Check the Folder Path

In **Paths and Configuration**, confirm that `PROJECT_ROOT` points to your shortcut:

```python
PROJECT_ROOT = Path(
    '/content/drive/MyDrive/'
    'Part 1, Module 2: Surface Water Dynamics from Wildlife Camera Imagery'
)
```

If you put the shortcut somewhere else or renamed it, update this path. You can browse **Files > drive > MyDrive** in Colab's left sidebar to find it. Copying the notebook to your own Drive does not change this shared-data path.

Leave `PERSONAL_OUTPUT_ROOT` pointing to your own `MyDrive/Workshop_Results`, not inside the shared folder.

## 6. Run the Remaining Cells

Run cells from top to bottom using the Run button, or select a code cell and press **Shift + Enter**. Wait for each cell to finish before continuing. Training and model loading can take longer than other steps.

At **Draw and Apply the Reference**:

1. Click consecutive points around the desired water-reference area on the photo.
2. Click the first point again to close the polygon.
3. Click **Apply and save** to calculate fractions and save results.

Use **Clear polygon** to start a new outline. Drawing a new reference does not require retraining or rerunning segmentation. The notebook cannot choose a polygon for you when using **Run all**.

## 7. Find Your Results

Results are saved automatically in your own Drive:

- **My Drive > Workshop_Results > yolov8**
- **My Drive > Workshop_Results > sam3**

These folders contain preprocessed images, masks, overlays, CSV tables, and plots. The YOLO folder also contains demo training outputs. The shared input images and model files remain unchanged.

The runtime's temporary storage and Python variables can disappear when Colab disconnects or restarts. Files already saved to your personal Drive remain available. Internet access is still required, but the large shared dataset and models are read by Colab rather than downloaded to each participant's computer.