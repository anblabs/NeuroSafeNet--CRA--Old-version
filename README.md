# NeuroSafeNet

**Conserved biological response states organize nanoparticle biosafety across neurological systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd.svg)](https://hydra.cc/)

---

## 📖 Overview

Predicting the biological safety and therapeutic recovery outcomes of engineered nanomaterials remains a major challenge in nanomedicine and regulatory toxicology due to physicochemical diversity, experimental heterogeneity, and context-dependent nano–bio interactions.

**NeuroSafeNet** implements an integrative, hierarchical machine-learning and data curation framework designed to decode and predict nanoparticle biosafety and recovery across heterogeneous biological systems (spanning *in vitro* and *in vivo* neurological models, and cross-organ liver and lung systems).

Through the synthesis of **599 original reports** encompassing **893 experimental contexts**, this framework introduces the **Conserved Response Axis (CRA)**—a unified, mechanistically interpretable biological space organized by three foundational programmes:
1. **Pro-inflammatory signaling** (e.g., TNF-α, IL-1β, IL-6, NF-κB, p38 MAPK)
2. **Anti-inflammatory & resolution pathways** (e.g., IL-10, TGF-β, SOCS family)
3. **Apoptosis-associated responses** (e.g., Caspase-3/8/9, Cyt-c, BAX/BCL-2)

```
Hierarchical Feature Integration Cascade (AOP Framework)
┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│        Step 1          │      │        Step 1+2         │      │       Step 1+2+3        │
│   Material Properties  │ ───▶ │  + Exposure & Context   │ ───▶ │  + Biological Response  │
│        (MIE-P)         │      │         (MIE-E)         │      │          (KE)           │
│ [MP: Size, Shape, Zeta,│      │ [MP + EC + B: Dose,     │      │ [MP + EC + B + CR:      │
│  Polymers, Synthesis]  │      │  Route, Cell, Organism] │      │  Pro/Anti-inf, Apopt.]  │
└────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                              │
                                                                              ▼
                                                                Conserved Response Axis (CRA)
                                                                • Biosafety (~90% Bal. Acc.)
                                                                • Recovery  (~78% Bal. Acc.)
```

---

## 🔬 Manuscript Summary & Scientific Highlights

### Title
> **Conserved biological response states organize nanoparticle biosafety across neurological systems**

### Authors
Sri Renukadevi Balusamy$^{1\#}$, Davoodbasha Mubarak Ali$^{2,3\#}$, Irfan Ullah$^{4}$, Girish Birappa$^{5}$, Samad Abdus$^{6}$, Priyanka Singh$^{7}$, Shadi Rahimi$^{8}$, Sumit Kumar$^{9}$, Mamata Karmacharya$^{9}$, Yeon Ju Kim$^{6}$, Suresh Ramakrishna$^{5,10}$, Ivan Mijakovic$^{8,11}$, Seungae Lee$^{12}$, Haribalan Perumalsamy$^{13,14\#*}$

$^\#$*These authors contributed equally*  
$^*$**Corresponding authors:**
* Haribalan Perumalsamy ([harijai2004@hanyang.ac.kr](mailto:harijai2004@hanyang.ac.kr))
* Sumit Kumar ([sumitwithchem@gmail.com](mailto:sumitwithchem@gmail.com))
* Seungah Lee ([moon11311@naver.com](mailto:moon11311@naver.com))

### Affiliations
1. Department of Food Science and Biotechnology, Sejong University, Seoul, Republic of Korea
2. School of Life Sciences, B.S. Abdur Rahman Crescent Institute of Science and Technology, Chennai, India
3. Centre Surface Technology and Applications, Korea Aerospace University, Goyang-si, Republic of Korea
4. Department of Computer Science and Engineering, Kyung Hee University (Global Campus), Yongin, South Korea
5. Graduate School of Biomedical Science and Engineering, Hanyang University, Seoul, Republic of Korea
6. Graduate School of Biotechnology, College of Life Science, Kyung Hee University, Yongin-Si, Republic of Korea
7. Department of Health Technology, Technical University of Denmark, Kongens Lyngby, Denmark
8. Division of Systems and Synthetic Biology, Chalmers University of Technology, Gothenburg, Sweden
9. Department of Biomedical Engineering, Ulsan National Institute of Science and Technology (UNIST), Ulsan, South Korea
10. College of Medicine, Hanyang University, Seoul, South Korea
11. The Novo Nordisk Foundation Biotechnology Research Institute for the Green Transition, Technical University of Denmark, Kongens Lyngby, Denmark
12. Department of Applied Chemistry and Institute of Natural Sciences, Kyung Hee University, Yongin-si, Republic of Korea
13. Center for Creative Convergence Education, Hanyang University, Seoul, Republic of Korea
14. Research Institute for Convergence of Basic Science, Hanyang University, Seoul, Republic of Korea

### Key Discoveries & Validation
* **Predictive Generalization via Hierarchical Integration:** Models incorporating biological response programmes (Step 1+2+3) significantly outperform purely physicochemical models (Step 1; Balanced Accuracy rose from 66% to 90% for biosafety, and 64% to 78% for recovery).
* **SHAP Feature Attribution:** Tree-based SHAP and ensemble interpretability identify pro-inflammatory and apoptosis-associated programmes as the dominant predictive drivers of biosafety and recovery classification.
* **Orthogonal Experimental Validation:** Evaluated using solution plasma-synthesized cellulose–cobalt nanocomposites (CxCoy) in IFN-$\gamma$-induced BV2 microglial neuroinflammation, measuring cell viability (MTT), nitric oxide (Griess), NF-$\kappa$B p65 / NOTCH2 translocation, ROS generation, and Annexin V/PI flow cytometry.
* **Functional Uptake Regulation:** Endocytic inhibition (Chlorpromazine & Methyl-$\beta$-cyclodextrin) coupled with quantitative ICP–MS confirmed that intracellular nanoparticle internalization directly drives state transitions along the CRA.
* **Cross-Scale & Cross-Organ Generalization:** Conserved response patterns validated across 3D biomimetic microglial spheroids, neurological disease transcriptomics (Alzheimer's disease [GSE142066], Multiple Sclerosis [GSE244966], Traumatic Brain Injury [GSE306006]), and cross-organ systems (Liver: GSE43515, GSE55286; Lung: GSE259384, GSE280933) across diverse nanomaterial classes (Carbon black, MWCNTs, AgNPs, Lipid Nanoparticles).

---

## 🏷️ Feature Architecture & Nomenclature (AOP Framework)

The dataset and modeling pipeline adhere to Adverse Outcome Pathway (AOP) terminology:

| Pipeline Code | Paper Term | Composition | Description & Included Variables |
| :--- | :--- | :--- | :--- |
| **`MIE-P`** | **`MP`** | Material Properties | Physicochemical attributes: Core Size, Hydrodynamic Diameter, Shape, Surface Charge, PDI, Polymer Type, Synthesis Method. |
| **`MIE-E`** | **`MP + EC + B`** | Material + Exposure + Biological System | Combines Material Properties (`MP`) with Exposure Conditions (`EC`: dose, concentration, exposure duration, stimulant) and Biological Model (`B`: cell line/species/organism, sex, age, administration route, injury model). |
| **`KE`** | **`CR`** | Cellular Response | Key Events measuring intermediate biological effects (cell toxicity %, inflammatory marker release, cellular uptake). |
| **`AO`** | **Adverse Outcome** | Target Endpoints | Evaluated target classes: **`AO_Biosafety`** and **`AO_Recovery`** (*Good* vs. *Bad*). |

> 📌 **Note on Dataset:** The dataset file `Data/Final_dataset.xlsx` contains two worksheets:
> 1. **`Final_dataset`**: The harmonized nanoparticle data matrix.
> 2. **`Important Note`**: The dedicated reference sheet defining all feature tiers (`MIE-P`, `MIE-E`, `KE`, `AO`) and their biological mappings.

---

## 📁 Repository Structure

```text
Neuro_Inflammation/
├── configs/                              # Modular Hydra / OmegaConf configurations
│   ├── config.yaml                       # Master training & cross-validation configuration
│   ├── paths.py                          # Dynamic path resolver
│   ├── settings.py                       # Global ML hyperparameters (n_splits=10, random_state=42)
│   ├── data/
│   │   ├── data.yaml                     # Active dataset and feature column selection
│   │   └── aop_biosafety_example.yaml    # Preconfigured AOP biosafety experiment spec
│   ├── io/
│   │   └── io.yaml                       # Output folder naming & export flags
│   ├── preprocessing/
│   │   └── preprocessing.yaml            # Imputation and categorical encoding rules
│   └── split/
│       └── split.yaml                    # Stratified train/internal/external split ratios
├── Data/
│   └── Final_dataset.xlsx                # Harmonized dataset with 'Important Note' sheet
├── outputs/                              # Generated artifacts, models, and predictions
│   ├── metrics_summary.csv               # Complete benchmark performance across all models
│   ├── preprocessed_Model_3/             # Processed NumPy feature matrices & encoders
│   └── training_Model_3/                 # Model binaries, CV summaries, and predictions
├── src/
│   ├── constants.py
│   ├── ninflam/                          # Core NeuroSafeNet Python package
│   │   ├── cli.py                        # CLI entrypoint (ninflam-prep)
│   │   ├── pre_pipeline.py               # Preprocessing orchestration pipeline
│   │   ├── config/                       # Dataclass schemas
│   │   ├── core/                         # Custom exceptions & logging handlers
│   │   ├── data/                         # Excel loaders, cleaners, and normalizers
│   │   ├── io/                           # Model serializers and artifacts managers
│   │   ├── pipeline/                     # Training & 10-fold cross-validation pipeline
│   │   ├── pre_train/                    # Imputers, encoders, and scalers
│   │   └── splitting/                    # Stratified splitting algorithms
│   ├── evaluation/                       # Evaluation & safe probability prediction routines
│   ├── feat/                             # Feature extraction utilities
│   ├── models/                           # Registry for model instances
│   └── trainer/                          # Training wrappers
├── main_preprocess.py                    # Preprocessing execution script
├── main_train.py                         # Model training & benchmark summary execution script
├── pyproject.toml                        # Package dependencies & metadata
├── requirements.txt                      # PIP requirements
└── README.md                             # Project documentation
```

---

## 🚀 Installation & Environment Setup

### Prerequisites
* Python $\ge$ 3.9
* Virtual environment (venv or conda)

### 1. Clone the Repository
```bash
git clone https://github.com/anblabs/NeuroSafeNet.git
cd NeuroSafeNet
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Package & Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

*Optional high-performance gradient boosting libraries:*
```bash
pip install xgboost catboost
```

---

## ⚙️ Configuration

The pipeline uses modular YAML files located in `configs/`:

* **`configs/config.yaml`**: Controls training mode, 10-fold cross-validation, and active model algorithms:
  ```yaml
  log_level: INFO
  training:
    use_kfold: true
    n_splits: 10
    models:
      - RF
      - DecisionTree
      - NaiveBayes
      - LogisticRegression
      - SVM
      - KNN
      - LDA
      - MLP
      - XGBoost
      - CatBoost
    min_class_count: 2
  ```
* **`configs/data/data.yaml`**: Specifies target endpoint (`AO_Biosafety` or `AO_Recovery`) and active input feature columns (`MIE-P`, `MIE-E`, `KE`).
* **`configs/split/split.yaml`**: Configures stratified external (20%) and internal (20%) test splits.
* **`configs/io/io.yaml`**: Controls output directory tagging and artifact export options.

---

## 🛠️ Usage & Workflow

### Step 1: Data Preprocessing
Cleans raw literature-curated matrices, handles missing data via stratified imputers, encodes categorical descriptors, performs stratified partitioning, and exports serializable feature artifacts:

```bash
python main_preprocess.py
```
*Or via CLI:*
```bash
ninflam-prep --mode preprocess
```

**Generated Preprocessing Artifacts (`outputs/preprocessed_Model_3/...`):**
* `splits.npz`: Feature matrices (`X_tr`, `X_int`, `X_ext`, `y_tr`, `y_int`, `y_ext`)
* `feature_names.npy`: Array of transformed feature identifiers
* `labels_encoded.npz`: Encoded class mappings (`Good` / `Bad`)
* `preprocessor.joblib`: Serialized imputation and transformation pipeline

---

### Step 2: Model Training & Evaluation
Trains benchmarked classifiers, executes stratified 10-fold cross-validation, saves fitted models, evaluates external holdout test data, and formats a console comparison table:

```bash
python main_train.py
```
*Or via CLI:*
```bash
ninflam-prep --mode train
```

**Live Console Output:**
Upon completion, `main_train.py` prints the consolidated performance table directly to the console:
```text
====================================================================================================================================
                                     ALL MODEL EVALUATION & BENCHMARK METRICS
====================================================================================================================================
Model Name           | 10-Fold CV Acc     | CV Bal Acc   | CV Prec    | CV Rec     | CV F1      | Test Acc   | Test F1   
------------------------------------------------------------------------------------------------------------------------------------
RF (Best CV Acc)     | 85.71% ± 3.40%     |     82.50%   |    84.12%  |    82.50%  |    83.15%  |    85.71%  |    78.79%
XGBoost (Best CV F1) | 89.80% ± 2.90%     |     86.40%  |    88.20%  |    86.40%  |    85.80%  |    89.80%  |    85.80%
...
====================================================================================================================================
👉 BEST OVERALL MODEL (by 10-Fold CV Accuracy): XGBoost (89.80% ± 2.90%)
👉 BEST MODEL (by 10-Fold CV Macro F1):        XGBoost (85.80%)
👉 BEST MODEL (by Test Split Accuracy):         XGBoost (89.80%)
```

**Generated Training Artifacts (`outputs/training_Model_3/...`):**
* `metrics_summary.csv`: Summary table comparing all models (Accuracy, F1-macro, ROC-AUC, PR-AUC).
* `models/*.joblib`: Serialized classifier binaries (e.g., `RF.joblib`, `SVM.joblib`).
* `metrics/*.json`: Detailed internal and external test-set classification reports.
* `predictions/*.csv`: Sample-by-sample ground truth vs prediction probabilities.
* `cross_validation/*_cv_summary.csv`: Mean and standard deviation across all 10 folds.
* `run_metadata.json`: Full environment, feature list, and experiment tracking logs.

---

## 📊 Supported Machine Learning Models

| Model Identifier | Algorithm Description | Implementation Backend |
| :--- | :--- | :--- |
| **RF** | Random Forest Classifier (`n_estimators=300`) | `scikit-learn` |
| **XGBoost** | Extreme Gradient Boosting Classifier | `xgboost` |
| **CatBoost** | Categorical Boosting Classifier | `catboost` |
| **SVM** | Support Vector Classifier (RBF Kernel, probability calibrated) | `scikit-learn` |
| **MLP** | Multi-Layer Perceptron Neural Network (128 $\times$ 64 hidden layers) | `scikit-learn` |
| **LogisticRegression** | L2-regularized Logistic Regression (`lbfgs` solver, `max_iter=2000`) | `scikit-learn` |
| **DecisionTree** | CART Decision Tree Classifier (`max_depth=10`) | `scikit-learn` |
| **KNN** | K-Nearest Neighbors Classifier ($k=5$) | `scikit-learn` |
| **LDA** | Linear Discriminant Analysis | `scikit-learn` |
| **NaiveBayes** | Gaussian Naive Bayes Classifier | `scikit-learn` |

---

## 📜 Citation

If you use **NeuroSafeNet**, the **Conserved Response Axis (CRA)** framework, or the curated dataset in your research, please cite:

```bibtex
@article{balusamy2026conserved,
  title={Conserved biological response states organize nanoparticle biosafety across neurological systems},
  author={Balusamy, Sri Renukadevi and Ali, Davoodbasha Mubarak and Ullah, Irfan and Birappa, Girish and Abdus, Samad and Singh, Priyanka and Rahimi, Shadi and Kumar, Sumit and Karmacharya, Mamata and Kim, Yeon Ju and Ramakrishna, Suresh and Mijakovic, Ivan and Lee, Seungae and Perumalsamy, Haribalan},
  journal={Preprint / Under Review},
  year={2026}
}
```

---

## 📬 Contact & Support

For technical questions regarding the code, data pipelines, or models, please open an issue or contact:
* **Haribalan Perumalsamy:** [harijai2004@hanyang.ac.kr](mailto:harijai2004@hanyang.ac.kr)
* **Sumit Kumar:** [sumitwithchem@gmail.com](mailto:sumitwithchem@gmail.com)
* **Seungah Lee:** [moon11311@naver.com](mailto:moon11311@naver.com)

