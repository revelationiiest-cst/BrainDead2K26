## About BrainDead
<img width="2880" height="1472" alt="Gemini_Generated_Image_3aaoid3aaoid3aao" src="https://github.com/user-attachments/assets/025f8eab-34c1-4a01-9f41-640751e75f13" />


Get ready for an exciting challenge in **Data Analysis** and **Machine Learning** at Revelation 2K26 with **BrainDead**! 🚀

We present **two compelling problem statements** for this flagship competition.

✅ **Both problems are compulsory** — whether you solve one, both, or even a part, we encourage you to submit and showcase your work.

On behalf of Team Revelation, we wish you all the best! 💻🔍  
**#BrainDead #Revelation26 #BrainDead26 #DataAnalysis #MachineLearning**


#**Revelation 2K25 | Department of Computer Science and Technology, IIEST Shibpur**
<img width="1920" height="520" alt="NB Banner" src="https://github.com/user-attachments/assets/81016c7b-92b2-4a4f-b769-89a567973996" />

🚀 **Exciting News!** Revelation 2K26 is back, proudly presented by the Department of Computer Science and Technology, IIEST Shibpur.  
This year, we are thrilled to collaborate with multiple leading product-based companies eager to recruit top talent from our major technical events.

🎉 Don’t miss this opportunity to showcase your skills, stand out, and secure your future in the tech world!  
**See you at Revelation 2K26!**  

**#Revelation26 #TechFest #IIESTS #CareerOpportunities**

### 📞 Contact Us

📧 **Email**: [revelationiiest@gmail.com](mailto:revelationiiest@gmail.com)  
📲 Join WhatsApp Group: **[Click here](https://chat.whatsapp.com/FiZG1P3RnmiEhphbiK8JYa)** 

--
## Submission Requirements
For **Problem Statement 1 & 2**, prepare a concise report in **.pptx** or **.pdf** format.

### Report Submission Rules
- Append your **Problem Statement 2** report to your **Problem Statement 1** report.
- Upload your **iPython notebook (.ipynb)** to GitHub.
- Submit your **public GitHub repository link** via **UnStop**.

### Important Notes
- Both problem statements are compulsory.
- You can submit whichever problem(s) you solve. Partial submissions will be evaluated based on your report and results.
- For any unattempted problem, your score for that question will be **0**.
- If you submit the report but **do not submit the notebook/code**, your score for that problem will be **0** *(Aryabhatta to be blamed! 🙃)*.


---
# **Problem Statement:1 🎬 ReelSense: Explainable Movie Recommender System with Diversity Optimization**
#### <div align="right">🎯 Marks: 40</div>


## 📌 Problem Statement

**ReelSense** is a movie recommendation system challenge that goes beyond just predicting user ratings. The project involves:

1. Building personalized, explainable **Top-K movie recommendations** using hybrid approaches  
2. Ensuring **diversity and coverage** to avoid popularity bias  
3. Generating **natural language explanations** for each recommendation  
4. Reporting metrics for **ranking, diversity, and novelty**



## 📁 Dataset

Dataset: [MovieLens Latest Small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)  
Source: GroupLens Research  
License: [MovieLens Terms of Use](https://grouplens.org/datasets/movielens/)  
Size: 100,836 ratings by 610 users on 9,742 movies

### Files Used:

- `ratings.csv`: User ratings of movies (0.5 to 5.0)
- `movies.csv`: Movie metadata (title, genres)
- `tags.csv`: User-assigned free-form tags
- `links.csv`: External IDs (IMDB, TMDb)



## 🧹 Preprocessing

- Time-based train-test split (Leave-last-N per user)
- Cleaning tags and genres
- Constructed user-item interaction matrix
- Parsed timestamp to datetime
- Standardized movie genre/tag features



## 🔍 Exploratory Data Analysis (EDA)

Key Visualizations:
- Distribution of ratings
- Genre popularity vs. rating trends
- User activity histogram
- Long-tail analysis
- Trends in rating behavior over time



## 🧠 Recommendation Models

| Type             | Methodologies Used                     |
|------------------|----------------------------------------|
| Popularity-based | Top-N most-rated / highest-rated       |
| Collaborative    | User-User CF, Item-Item CF             |
| Matrix Factorization | SVD using Surprise library         |
| Hybrid Model     | Genre+Tag content features + CF blend  |



## ✨ Explainability Layer

For each recommendation, generated explanations like:

> "Because you liked *Inception* and *The Matrix*, which share the tags 'sci-fi' and 'mind-bending'"

Explanation sources:
- Tag similarity
- Genre overlap
- Collaborative user neighborhood



## 🎯 Evaluation Metrics

### A. Rating Prediction (for MF models)
- RMSE / MAE

### B. Top-K Ranking
- Precision@K, Recall@K, NDCG@K (K=10)
- MAP@K

### C. Diversity & Novelty
- Catalog Coverage
- Intra-List Diversity
- Popularity-Normalized Hits


## 📦 Deliverables

Participants must submit the following:

- **Concise report** (figures + insights + references)  
- **Cleaned dataset pipeline** + notebooks/scripts  
- **EDA visuals** (clearly labeled)  
- **Model training + evaluation code**




---

# **Problem Statement:2 Cognitive Radiology Report Generation**
#### <div align="right">🎯 Marks: 60</div>

#### 📌 The Challenge:
In the high-pressure environment of a radiology reading room, "reader fatigue" leads to a 3-5% discrepancy rate in human-generated reports. Your task is to develop a **Deep Learning Framework** that acts as a "Second Reader"—an AI capable of automating the drafting of radiology reports.

Participants must build a model that:
1.  **Ingests:** Chest X-Ray (CXR) images (PA/Lateral views) and Clinical Indication text (e.g., *"55M with fever"*).
2.  **Processes:** Aligns visual features with medical ontology (RadLex) and mimics clinical reasoning.
3.  **Outputs:** A structured, clinically accurate text report comprising **Findings** and **Impression** sections.

**The Core Innovation:**
To win, your solution must move beyond standard black-box transformers. You must demonstrate **"Cognitive Simulation"** inspired by the **Hi-CliTr** framework, implementing:
* **Hierarchical Visual Perception (PRO-FA)**
* **Knowledge-Enhanced Classification (MIX-MLP)**
* **Triangular Cognitive Attention (RCTA)**

#### 📂 Dataset Specifications:
You will utilize two primary datasets. **MIMIC-CXR** is the gold standard for training, while **IU-Xray** is used for benchmarking domain generalization.

1.  **MIMIC-CXR (Primary Training Data)** ([PhysioNet Access Required](https://physionet.org/content/mimic-cxr/2.0.0/))
2.  **IU-Xray (Benchmarking Data)** ([Kaggle Mirror](https://www.kaggle.com/raddar/chest-xrays-indiana-university) or OpenI)

#### 📊 Data Field Description:

**A. MIMIC-CXR Structure**
The dataset contains 377,110 images corresponding to 227,835 radiographic studies.
```json
{
  "subject_id": "Patient identifier (Unique per patient)",
  "study_id": "Examination identifier (Target reports are linked to this)",
  "dicom_id": "Image identifier (Specific to a single X-ray view)",
  "text": "The full free-text radiology report (Target)",
  "ViewPosition": "PA (Posterior-Anterior) or LATERAL"
}

```

**B. IU-Xray Structure (`indiana_projections.csv` & `indiana_reports.csv`)**
A smaller, open-access dataset for testing robustness.

```json
{
  "uid": "Unique Report ID (Foreign Key for merging)",
  "filename": "Name of the image file (e.g., 1_IM-0001-4001.dcm.png)",
  "projection": "View position (Frontal/Lateral)",
  "findings": "Target Output 1: Detailed observations of the scan",
  "impression": "Target Output 2: Summary diagnosis",
  "indication": "Input Feature: Patient symptoms/history (Context)",
  "MeSH": "Medical Subject Headings (useful for label classification)"
}

```

---

### 🛠 Technical Requirements (Mandatory Architecture)

To solve issues of "hallucination" and lack of interpretability, your submission **must** implement the following three architectural concepts.

#### 1. Module 1: Hierarchical Visual Alignment (PRO-FA)

* **Logic:** A radiologist analyzes images at different scales: Whole Organ (Heart), Region (Lobe), and Pixel (Lesion).
* **Requirement:** Your visual encoder (e.g., ViT) must extract features at three distinct granularities: **Pixel-level, Region-level, and Organ-level**.
* **Constraint:** Use **RadLex embeddings** to align these visual features with medical text (e.g., ensuring the model knows what a "Lung" looks like).

#### 2. Module 2: Knowledge-Enhanced Classification (MIX-MLP)

* **Logic:** Before writing, a doctor forms a "mental diagnosis" (e.g., "Positive for Pneumonia").
* **Requirement:** Implement a multi-label classification branch that predicts disease tags (using **CheXpert**) before generating text.
* **Constraint:** Use a **Multi-path MLP** (Residual Path + Expansion Path) to model disease co-occurrence and handle noisy labels.

#### 3. Module 3: Triangular Cognitive Attention (RCTA)

* **Logic:** Doctors use a verification loop: Look at Image  Check History  Form Hypothesis  Verify with Image.
* **Requirement:** Implement a triangular attention mechanism:
1. **Image queries Clinical Text**  Creates Context.
2. **Context queries Predicted Labels**  Creates Hypothesis.
3. **Hypothesis queries Image (again)**  Verification (Closed Loop).


### 🔔 **Important Notes:**
- **Dates:** February 6th - 8th, 2026.
- **Submission Portal:** [Unstop](https://unstop.com/) (All registrations and final submissions must be made here).
- **Evaluation:** Solutions are judged on **Clinical Efficiency (CE)** and **Semantic Structure**, not just text similarity.
- **Strict Constraint:** Simple Encoder-Decoder models (image captioning) will be penalized. You must implement the **Cognitive Modules** described below.
---

### 📊 Performance Metrics

Your solution will be evaluated on a hidden test set based on **Natural Language Generation (NLG)** quality and **Clinical Efficiency (CE)**.

| Metric Category | Metrics Used | Weight | Description | Target Benchmark |
| --- | --- | --- | --- | --- |
| **Clinical Accuracy** | **CheXpert F1** | **40%** | Does the report diagnose the correct diseases? (Precision/Recall on 14 pathologies). | **F1 > 0.500** |
| **Structural Logic** | **RadGraph F1** | **30%** | Does the model correctly link entities? (e.g., *"Pneumonia"* is `located_at` *"Left Lower Lobe"*). | **Relations > 0.500** |
| **NLG Fluency** | **CIDEr, BLEU-4** | **30%** | Is the text readable, grammatically correct, and capturing rare words? | **CIDEr > 0.400** |

---

### 💻 Environment & Tools

* **Platform:** Google Colab Pro / Local GPU Cluster (Mixed Precision Training Recommended).
* **Frameworks:** PyTorch, TensorFlow.
* **Required Libraries:** `transformers` (Hugging Face), `chexpert-labeler`, `radgraph`.
* **Docker:** All submissions must be containerized for reproducibility.

---

### 📝 Submission Guidelines

**1. Repository Structure**
Your GitHub repository must follow this structure:

```bash
BrainDead-Solution/
├── data/               # Scripts to download/preprocess data
├── models/
│   ├── encoder.py      # PRO-FA implementation
│   ├── classifier.py   # MIX-MLP implementation
│   └── decoder.py      # RCTA attention & generator
├── training/           # Training loops and loss functions
├── evaluation/         # CheXpert and RadGraph evaluators
├── notebooks/          # Demo notebooks (inference examples)
├── requirements.txt    # Dependencies
└── README.md           # Documentation (Architecture diagram, setup steps)

```

**2. Submission Format**

1. **Codebase:** Push to a public GitHub repository.
2. **Model Checkpoint:** Host your best model weights on Drive/Dropbox/Mega and include the link in `submission.txt`.
3. **Report:** A brief PDF (max 2 pages, IEEE format) explaining how you implemented the three mandatory modules.
4. **Demo Video:** A 2-minute screencast showing your model generating a report from a raw X-ray.

**3. Deadline:** February 8th, 2026, 11:59 PM IST.

---

### 🏆 Marking Criteria

| Component | Points | Details |
| --- | --- | --- |
| **Implementation of Modules** | **40** | Did you successfully implement PRO-FA, MIX-MLP, and RCTA? |
| **Clinical F1 Score** | **30** | Performance on the hidden test set (Clinical Accuracy). |
| **Code Quality** | **15** | Modularity, reproducibility, and cleanliness of code. |
| **Innovation** | **15** | Creative handling of data imbalance or novel loss functions. |

---

### 🚀 Heads Up!

We will be checking all submissions for plagiarism (code and text). "Brain Dead" requires active brains! Any rigorous copying without attribution will lead to immediate disqualification.

**Good luck! May your loss functions converge and your F1 scores soar!** 🩺🤖

```
