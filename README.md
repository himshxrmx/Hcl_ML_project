# AI Resume Screening & Shortlisting

A machine learning model that predicts whether a job candidate should be shortlisted, and ranks candidates by their probability of being shortlisted so HR can pick the top N.

## Dataset

`ai_resume_screening.csv`: 30,000 candidates. The target is `shortlisted` (70% Yes, 30% No).

| Feature | Description |
|---|---|
| `years_experience` | Years of work experience |
| `skills_match_score` | Match between candidate skills and job requirements (0–100) |
| `education_level` | High School / Bachelors / Masters / PhD |
| `project_count` | Number of projects |
| `resume_length` | Length of the resume |
| `github_activity` | GitHub activity level |

## Approach

1. **Cleaning:** checked for missing values, duplicates and invalid values (none found); capped outliers with the IQR rule instead of deleting rows.
2. **EDA:** correlation matrix, feature vs. target plots, shortlist rate by education.
3. **Encoding:** education mapped in order (High School 0 → PhD 3); target Yes/No → 1/0.
4. **Split and scale:** stratified 80/20 train/test split; `StandardScaler` fitted on the training data only.
5. **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and XGBoost, compared on the test set.
6. **Shortlisting:** candidates are ranked by predicted probability, and the top N (or everyone above a threshold) is shortlisted.

## Results (test set, 6,000 candidates)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.9065** | 0.9275 | **0.9397** | **0.9335** | **0.9669** |
| Logistic Regression | 0.9043 | 0.9281 | 0.9356 | 0.9318 | 0.9650 |
| XGBoost | 0.9022 | 0.9228 | 0.9385 | 0.9306 | 0.9634 |
| Random Forest | 0.8995 | 0.9214 | 0.9361 | 0.9287 | 0.9618 |
| Decision Tree | 0.8665 | 0.9025 | 0.9070 | 0.9047 | 0.8398 |

Baseline ("always Yes"): accuracy 0.70, F1 0.82, ROC-AUC 0.50.

**Best model: Gradient Boosting.** It has the highest F1 and ROC-AUC, and a train/test accuracy gap of only 0.009. Decision Tree and Random Forest overfit: they scored 1.00 on the training set.

## Project structure

```
resume_screening.ipynb    Full notebook: cleaning → EDA → training → evaluation → shortlisting
ai_resume_screening.csv   Dataset
app.py                    Streamlit web app for screening a single candidate
models/                   Saved model, scaler and metadata (education mapping, feature order)
requirements.txt          Python packages
```

## How to run

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Web app:

```bash
venv\Scripts\python.exe -m streamlit run app.py
```

Notebook: open `resume_screening.ipynb` in Jupyter and run all cells. This also retrains the model and saves it again to `models/`.

## Limitations

- The dataset is synthetic; real hiring data is messier.
- Only 6 numeric features are used; the model does not read the resume text itself.
- Features like education can carry bias, so a human should review final decisions.
