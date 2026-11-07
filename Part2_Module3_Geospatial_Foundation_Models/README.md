# AI-Enabled Multiscale Environmental Science Workshop
## Part 3, Module 3: Geospatial Foundation Models for Landscape Prediction

Explore geospatial embeddings with **AlphaEarth, TESSERA, and OlmoEarth**, then train a model to classify wetland types and open water in the Florida Everglades.

**Notebook:** `geospatial_embeddings.ipynb`

## Before you begin

1. Get the notebook and data folder from the instructor.
2. Copy the data folder into **My Drive**, keeping the original filenames. Files only in “Shared with me” may not be found. Avoid duplicate copies of the workshop files.
3. Open the notebook in [Google Colab](https://colab.research.google.com/) and save your own copy.
4. Run the setup cells and authorize Google Drive access when prompted. A GPU is not required.

You will need a Google account and an internet connection. Some data are downloaded during the workshop; training inputs and a prediction map have been prepared in advance to save time.

## Workshop files

Keep these files together in your workshop data folder:

| Activity | Files |
| --- | --- |
| Explore embeddings | `stream_aoi.geojson` |
| Train a classifier | `everglades_nwi.geojson`, `olmo_X_valid.npy`, `olmo_y_valid.npy`, `class_map.json` |
| Compare with Sentinel-2 (bonus) | `compare_X_embed.npy`, `compare_X_s2.npy`, `compare_y.npy`, `compare_class_map.json`, `compare_s2_bands.json` |
| View the prepared prediction map (bonus) | `prediction_2024.tif` |

## Follow along

Run cells in order as the instructor guides you. **Avoid “Run all,”** which also starts the slow optional prediction step.

1. **Set up:** Install packages and import libraries.
2. **Explore embeddings:** Load the area of interest, retrieve three embedding datasets, and visualize their spatial patterns.
3. **Find similar locations:** Click within the loaded area on the AlphaEarth similarity map. Run the memory-cleanup cell when finished.
4. **Train a model:** Load the prepared Everglades samples, train a random forest, and examine accuracy and feature importance.

**Bonus 1:** Compare models trained on embeddings and Sentinel-2 bands.

**Bonus 2:** View predictions for 2024. Run the setup cells, including the one that sets `predictions_generated = False`. Skip the slow prediction cell marked optional, then run the final plotting cell to load the prepared TIFF from Drive.

## If something goes wrong

- **File not found:** Check that Drive is mounted and the file is in My Drive with its original name.
- **Missing variable:** Run the earlier cells in order. Restarting the session clears variables.
- **Package installation requests a restart:** Restart, then resume with the imports and mount Drive again.
- **Running out of RAM:** Run the cleanup cell before training. To revisit the similarity map afterward, rerun its data-loading and setup cells.
- **Downloads stall or an error persists:** Ask the instructor for help. The prepared training data allow you to continue with the modeling exercise.

## Data sources

- [AlphaEarth Foundations](https://source.coop/tge-labs/aef-mosaic)
- [TESSERA](https://source.coop/tessera/tessera)
- [OlmoEarth v1.3](https://source.coop/olmoearth/olmoearth-v1-3-embeddings)
- [U.S. Fish and Wildlife Service National Wetlands Inventory](https://www.fws.gov/program/national-wetlands-inventory)


Workshop training files and the prediction map are prepared derivatives supplied by the instructor. Refer to the linked providers for dataset citations and terms of use.
