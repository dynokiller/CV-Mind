# AI Resume Analyzer — Dataset Documentation

## Dataset Overview

- **Dataset name**: `synthetic_resumes.csv` / `v3_synthetic_resumes.csv`
- **Purpose**: Supervised training of the resume domain classification model.
- **Primary task**: Multi-class classification of resumes into technical domains.
- **Source**: Programmatically generated synthetic resumes combined with templated
  experience/skills snippets for each domain.

## Domains

The current datasets focus on the following domains:

- **Cybersecurity**
- **Data Science**
- **Web Development**
- **Cloud**
- **DevOps**
- **AI/ML**

Additional non-technical or adjacent domains may be present in extended datasets
used during experimentation, but the production classifier is expected to at
least cover the six core domains above.

## Fields

Each row in `synthetic_resumes.csv` / `v3_synthetic_resumes.csv` contains:

- **`resume_text`**: Full plain‑text representation of a resume.
- **`skills`** (implicit inside `resume_text`): Technologies, tools, and soft skills.
- **`experience`** (implicit): Descriptions of roles, responsibilities, and achievements.
- **`education`** (implicit): Degree names, universities, graduation years.
- **`projects`** (implicit): Project titles and short descriptions.
- **`certifications`** (implicit): Vendor or platform certifications when present.
- **`domain` / `domain_label`**: Target label indicating the primary domain of the resume.

Downstream services (MODEL 1) further derive structured features from
`resume_text` such as extracted skills, estimated years of experience, and
education indicators.

## Size and Splits

Approximate characteristics (may vary between synthetic generator versions):

- **Total samples**: 5,000–15,000 synthetic resumes per dataset version.
- **Train/validation/test split**:
  - Train: ~80%
  - Validation: ~10%
  - Test: ~10%
- **Token length**:
  - Typical resume length: 150–800 tokens after cleaning.

Exact counts and split statistics are logged during training runs in
`training/train.py` and `resume_ai_model/training/train_model.py`.

## Class Distribution

To mitigate class imbalance, the training pipelines:

- Use **stratified train/test splits** (`train_test_split(..., stratify=label)`).
- Compute **class weights** for loss functions (e.g. `compute_class_weight` in
  `training/train.py`), especially for the RoBERTa and XGBoost‑based models.

In a typical configuration, each of the six core domains has at least several
hundred synthetic samples, with distributions adjusted to approximate realistic
hiring volume (e.g., more Web/Data/Cloud than niche domains).

## Usage Notes

- The datasets are primarily **synthetic**, so additional real‑world resumes are
  recommended for:
  - Calibration of decision thresholds.
  - Robustness testing on noisy formatting and language.
  - Measuring true generalisation beyond templated text.
- The active‑learning feedback endpoint (`/feedback` in `ai_resume_analyzer/api`)
  is designed to capture corrected domain labels from users; these can be merged
  into future dataset versions to improve model performance over time.

