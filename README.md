# AI-Enabled_Multiscale_Environmental_Science
> **From Smartphone Videos to Satellite-Derived Predictions to Geospatial Foundation Models**

Welcome to the **AI-Enabled Multiscale Environmental Science** workshop repository! This repository contains the training modules, workflows, and code designed to provide practical experience across three distinct scales of AI-enabled environmental science, using published research workflows and community-collected data from the **WHONDRS network** as working examples.

---

* **Prerequisites:** No advanced programming expertise is required—only basic familiarity with Python.
---

## Learning Outcomes

By the end of this workshop, participants will be able to:

1. **Describe Multiscale AI Approaches:** Explain how AI addresses scale mismatches between fine-scale process characterization and landscape-scale prediction, including the roles of computer vision for local data extraction and geospatial foundation models for spatial extrapolation.
2. **Quantify Surface Water Dynamics:** Apply image segmentation models to wildlife camera imagery to calculate wet fractions and approximate real-world stream width using camera positioning metadata, while assessing failure conditions (e.g., snow cover, night vision, extreme lighting).
3. **Map Fine-Scale Heterogeneity:** Use the [AI4GSD](https://drive.google.com/drive/folders/1P1X86TbaQOwjSUs52jXv-_IvPvtsM_bE) software pipeline to derive centimeter-resolution sediment grain size distributions from smartphone or drone videos with automated quality control.
4. **Utilize Geospatial Foundation Models:** Use geospatial foundation model embeddings for semantic search and landscape-scale prediction of environmental states, including downloading embeddings and developing predictive models from sparse field-derived labels.

---

## Repository & File Structure

This repository is structured into modules corresponding to the different scales and practical exercises presented during the workshop:

```text
.
├── Part1_Module_1_Sediment_Grain_Size_Mapping/   # Module 1: AI4GSD video processing & grain size estimation
├── Part1_Module_2_Surface_Water_Dynamics/        # Module 2: Wildlife camera image segmentation (YOLOv8, Mask2Former)
├── Part2_Module3_Geospatial_Foundation_Models/    # Module 3: Pre-extracted embeddings & Everglades inundation prediction
└── README.md                                      # Repository orientation & overview
```
---
## Getting Started

You can run these interactive notebooks directly in your browser or locally:

1. **Google Colab (Recommended):**
Access the Workshop Google Drive [folder](https://drive.google.com/drive/folders/1P1X86TbaQOwjSUs52jXv-_IvPvtsM_bE?usp=sharing). Click on the individual Jupyter Notebook files within the module folders and use the "Open in Colab" badge to launch directly in a GPU-accelerated cloud environment.
3. **Local Environment:** Clone this repository and run locally:
   ```bash
   git clone [https://github.com/WHONDRS-Hub/AI-Enabled_Multiscale_Environmental_Science.git](https://github.com/WHONDRS-Hub/AI-Enabled_Multiscale_Environmental_Science.git)
   cd AI-Enabled_Multiscale_Environmental_Science

---

## Facilitators & Instructors

* **Co-Coordinators:** Viviana Freire Zapata, Vanessa Garayburu-Caruso
* **Presenters & Instructors:** James Stegen, Yunxiang Chen, Jie Bao, Jenna Abrahamson
* **Data Sources & Network:** WHONDRS Network
