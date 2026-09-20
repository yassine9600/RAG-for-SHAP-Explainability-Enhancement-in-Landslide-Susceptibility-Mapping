================================================================================
  LANDSLIDE SUSCEPTIBILITY MAPPING AND EXPLANATION FRAMEWORK
  Powered by LightGBM · TreeSHAP · Retrieval-Augmented Generation (RAG)
  Al Haouz Province, Morocco — 2023 Mw 6.8 Earthquake Context
================================================================================

OVERVIEW
--------
This package delivers a complete research pipeline in two parts:

  PART 1 — LANDSLIDE SUSCEPTIBILITY PREDICTION (optional, Google Colab)
            Train and compare machine learning models on the slope unit dataset.
            Produces the LightGBM model bundle used in Part 2.

  PART 2 — INTERACTIVE EXPLANATION APPLICATION (main, runs locally)
            Loads the trained model, displays the susceptibility map, computes
            and corrects SHAP attributions, and generates RAG-LLM explanations
            for any selected slope unit — accessible to non-specialists.

--------------------------------------------------------------------------------
DELIVERED FILES — COMPLETE PACKAGE CONTENTS
--------------------------------------------------------------------------------

  README.txt                              This file

  Landslide_Susceptibility_Prediction/    PART 1 — Training scripts (Colab)
  │
  ├── landslide_Based_LightGBM.py         LightGBM training, SHAP, and bundle
  │                                       export for the best-performing model
  ├── LSM_Comparaison_Models.py           Comparison of five ML models:
  │                                       RF, XGBoost, LightGBM, AdaBoost, KNN
  │                                       with ROC curves and performance table
  └── SU_2_5_c4.xlsx                      Training dataset — one row per slope
                                          unit, one column per conditioning
                                          factor, plus a binary Landslide label

  app.py                                  PART 2 — Main Streamlit application
  rag_engine.py                           RAG pipeline and SHAP diagnostic engine
  lgbm_landslide_bundle.pkl               Pre-trained LightGBM model bundle
  covariates.xlsx                         Slope unit conditioning factor values
  SU_2_5_C4_WITH_PREDICTIONS.zip          Shapefile of slope unit polygons
  requirements.txt                        Python package list for installation
  pdfs/                                   Scientific PDF papers for RAG system

--------------------------------------------------------------------------------
PART 1 — LANDSLIDE SUSCEPTIBILITY PREDICTION (Google Colab)
--------------------------------------------------------------------------------

This part is optional. A ready-to-use model bundle (lgbm_landslide_bundle.pkl)
is already provided so you can go directly to Part 2. Use Part 1 only if you
want to retrain the model, reproduce the model comparison results, or apply
the pipeline to a different dataset.

WHY GOOGLE COLAB?
  The training scripts use GPU acceleration to speed up model fitting,
  cross-validation, and SHAP computation. Google Colab provides free GPU
  access without any local installation. Running on a standard laptop CPU
  is possible but significantly slower (30–60 minutes vs. 3–5 minutes on GPU).

FILES IN THIS SECTION
  SU_2_5_c4.xlsx              Training dataset. Contains one row per slope
                              unit polygon with columns for each conditioning
                              factor and a binary target column "Landslide"
                              (1 = landslide, 0 = non-landslide).

  landslide_Based_LightGBM.py Trains the LightGBM classifier using randomized
                              cross-validation, computes global and local SHAP
                              values, and exports the trained model as a .pkl
                              bundle file ready for use in Part 2.

  LSM_Comparaison_Models.py   Trains and evaluates five machine learning models
  (optional)                  (Random Forest, XGBoost, LightGBM, AdaBoost, KNN)
                              on the same dataset, produces ROC curves, AUC
                              scores, accuracy, precision, recall, and F1 table,
                              and identifies the best-performing model.
                              Run this script first if you want to reproduce the
                              model selection results from the study.

HOW TO RUN IN GOOGLE COLAB — STEP BY STEP

  STEP 1 — Open Google Colab
    Go to: https://colab.research.google.com
    Sign in with any Google account (free).

  STEP 2 — Enable GPU runtime
    In the top menu: Runtime → Change runtime type
    Set "Hardware accelerator" to GPU (T4 GPU is sufficient).
    Click Save.

  STEP 3 — Upload the dataset
    In the left panel, click the folder icon to open the Files panel.
    Click the upload icon (upward arrow) and select:
      SU_2_5_c4.xlsx
    Wait for the upload to complete. The file will appear in the panel
    under /content/SU_2_5_c4.xlsx

  STEP 4 — Upload the Python script
    Upload either:
      LSM_Comparaison_Models.py   (for model comparison — run this first)
      landslide_Based_LightGBM.py (for LightGBM training and bundle export)
    Click the upload icon again and select the chosen .py file.

  STEP 5 — Open the script as a notebook
    In the Files panel, right-click the uploaded .py file and select
    "Open with → Text Editor" to inspect it, or run it directly (Step 6).

  STEP 6 — Install required packages
    Click the + Code button at the top to add a new code cell.
    Paste the following and press Shift + Enter to run:

      !pip install lightgbm shap scikit-learn pandas openpyxl matplotlib
               xgboost imbalanced-learn joblib scipy seaborn

  STEP 7 — Run the script
    Add another code cell and paste:

      exec(open("landslide_Based_LightGBM.py").read())

    or for model comparison:

      exec(open("LSM_Comparaison_Models.py").read())

    Press Shift + Enter. The script will execute in full.
    Outputs (charts, metrics, model bundle) will appear below the cell.

  STEP 8 — Download the model bundle (LightGBM script only)
    After landslide_Based_LightGBM.py finishes, a file named
    lgbm_landslide_bundle.pkl will appear in the Files panel.
    Right-click it and select "Download" to save it to your computer.
    This file is the input for Part 2 of the application.

EXPECTED OUTPUTS

  From LSM_Comparaison_Models.py:
    • Performance table (AUC, Accuracy, Precision, Recall, F1) for all 5 models
    • ROC curve comparison plot
    • Confusion matrices for each model
    • Identification of the best-performing model

  From landslide_Based_LightGBM.py:
    • Global SHAP beeswarm and bar plots
    • Feature importance ranking
    • Cross-validation AUC score
    • lgbm_landslide_bundle.pkl  — model bundle file for the application

NOTE ON THE DATASET
  SU_2_5_c4.xlsx contains the conditioning factors extracted for all slope
  unit polygons in Al Haouz Province. The "Landslide" column is the binary
  target: 1 for polygons with confirmed landslide inventory, 0 for stable
  polygons identified by the Frequency Ratio method. The Variance Inflation
  Factor (VIF) screening step is included in the scripts — factors exceeding
  VIF > 10 (specifically Slope) are excluded automatically before training.

--------------------------------------------------------------------------------
PART 2 — INTERACTIVE EXPLANATION APPLICATION (Local Machine)
--------------------------------------------------------------------------------

This is the main application. It loads the pre-trained LightGBM model, maps
susceptibility across Al Haouz Province, computes and corrects SHAP artefacts,
and generates RAG-LLM narrative explanations for selected slope units.

PREREQUISITES

  1. Python 3.10 or higher
     Download from: https://www.python.org/downloads/
     Verify with:   python --version

  2. All delivered files placed in one folder (do not rename them)

  3. OpenRouter API key for LLM explanations (free)
     Register at:   https://openrouter.ai
     Maps and SHAP charts work without a key. Only the narrative
     explanation feature requires one.

INSTALLATION

  STEP 1 — Place all files in a single folder
    Example folder name: landslide_app
    Keep the pdfs/ subfolder inside it.

  STEP 2 — Open a terminal inside that folder
    Windows : click the address bar of the folder, type cmd, press Enter
    Mac/Linux: right-click the folder, select "Open Terminal here"

  STEP 3 — Install all required packages
    Type the following and press Enter:

      pip install -r requirements.txt

    This installs all dependencies at once. It may take 5 to 10 minutes
    on the first run. Do not close the terminal until it finishes.

    If you get a "pip not found" error, try:
      python -m pip install -r requirements.txt

RUNNING THE APPLICATION

  STEP 4 — Launch the application
    In the terminal, type:

      streamlit run app.py

    Your browser will open automatically at http://localhost:8501
    If not, copy that address and paste it into any browser.

  STEP 5 — Upload the three input files
    In the left sidebar, upload each file using the corresponding button:

      Model Bundle (.pkl)   →   lgbm_landslide_bundle.pkl
      Covariates (.xlsx)    →   covariates.xlsx
      Shapefile (.zip)      →   SU_2_5_C4_WITH_PREDICTIONS.zip

    Wait for the green confirmation after each upload.
    The application will not proceed until all three files are loaded.

  STEP 6 — Configure LLM and RAG settings
    In the sidebar under "LLM · RAG Settings":
      a) Enter your OpenRouter API key
      b) Select a language model (default is recommended)
      c) Set temperature to 0.15
      d) Set max output tokens to 1024
      e) Set RAG top-k chunks to 5
      f) Verify the PDF path shows ./pdfs
      g) Click "Build / Rebuild Vector DB" — required only once.
         The app reads all PDFs, computes embeddings, and stores them
         locally. This takes 5–15 minutes for a full paper collection
         and is skipped automatically on subsequent runs.

NAVIGATING THE FIVE TABS

  Susceptibility Map     Interactive map with 5 basemap options (Relief,
                         Topo, Hillshade, Light, Satellite). Adjustable
                         opacity reveals terrain beneath the susceptibility
                         layer. Download a 300 dpi static map.

  Statistics             Susceptibility class distribution, zone statistics,
                         and radar charts per class.

  SHAP Explainability    Global SHAP importance: beeswarm, bar, and violin
                         plots. Download at 1200 dpi.

  Polygon Explorer       Click any polygon to view its SHAP waterfall plot,
                         feature attribution table, and RAG-LLM explanation.
                         Click "Generate Corrected Explanation" to produce
                         the L1/L2-corrected narrative with spatial context.

  Validation             Three sub-tabs: Detection (L1/L2 diagnosis),
                         Correction (uncorrected vs. corrected comparison),
                         Validation (EQI and RAGAS quality scores).

UNDERSTANDING THE RESULTS

  Susceptibility Classes
    Very Low    Stable terrain under normal conditions
    Low         Minor risk under extreme events
    Moderate    Monitor during heavy rainfall or seismic activity
    High        Field inspection strongly recommended
    Very High   Restrict access — near-certain instability

  SHAP Values
    Positive SHAP  This factor increases landslide probability here
    Negative SHAP  This factor decreases landslide probability here
    L1 flag        Collinearity artefact — interpret as joint driver
    L2 flag        Sign reversal — excluded from causal ranking, disclosed

  EQI Score (0–100)
    Grade A (75 or above)  Publication-ready explanation
    Grade B (50 to 74)     Revision recommended
    Grade C (below 50)     Major issues — re-run recommended

  RAGAS Scores (0–1)
    Context Precision    Relevance of retrieved literature chunks
    Context Recall       Completeness of retrieved chunks
    Answer Relevance     Alignment of answer with the geotechnical question
    Faithfulness         Proportion of claims grounded in retrieved evidence

--------------------------------------------------------------------------------
FOLDER STRUCTURE — COMPLETE PACKAGE
--------------------------------------------------------------------------------

  package/
  │
  ├── README.txt
  ├── requirements.txt
  │
  ├── Landslide_Susceptibility_Prediction/        PART 1 — Colab scripts
  │   ├── landslide_Based_LightGBM.py
  │   ├── LSM_Comparaison_Models.py
  │   └── SU_2_5_c4.xlsx
  │
  ├── app.py                                      PART 2 — Main application
  ├── rag_engine.py
  ├── lgbm_landslide_bundle.pkl
  ├── covariates.xlsx
  ├── SU_2_5_C4_WITH_PREDICTIONS.zip
  │
  ├── pdfs/                                       RAG knowledge base
  │   ├── paper1.pdf
  │   └── ...
  │
  └── chroma_db/                                  Auto-created on first run
      └── ...

--------------------------------------------------------------------------------
TECHNICAL NOTES
--------------------------------------------------------------------------------

  • Streamlit caches loaded files. If you replace any input file, go to
    the top-right menu in the browser (⋮) and select "Clear cache"
    before re-uploading.

  • The L1 collinearity threshold is |ρ| ≥ 0.60 (Spearman) with a
    dominance ratio of 3.0×. These are adjustable in rag_engine.py
    via _L1_CORR_THRESHOLD and _L1_DOMINANCE_THRESHOLD.

  • The embedding model (all-MiniLM-L6-v2, approximately 90 MB) is
    downloaded automatically from Hugging Face on the first "Build
    Vector DB" run. Internet is required for this step only.

  • The chroma_db/ folder can be safely deleted to rebuild the vector
    database from scratch when new PDFs are added.

  • To stop the application, return to the terminal and press Ctrl + C.

--------------------------------------------------------------------------------
CITATION
--------------------------------------------------------------------------------

  "Retrieval-Augmented Generation for SHAP Explainability Enhancement
   and Artefact Correction in Machine Learning-Based Landslide
   Susceptibility Mapping — Application to Al Haouz Province, Morocco"

  Study area : Al Haouz Province, Morocco
  Context    : 2023 Mw 6.8 Al Haouz earthquake, 8 September 2023

================================================================================