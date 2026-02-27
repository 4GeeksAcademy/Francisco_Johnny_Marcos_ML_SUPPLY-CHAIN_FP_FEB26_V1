# DataCo Supply Chain Risk Intelligence
**Strategic Logistics Analysis & Predictive Modeling**

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/4GeeksAcademy/Francisco_Johnny_Marcos_ML_SUPPLY-CHAIN_FP_FEB26_V1)

[![Project Status](https://img.shields.io/badge/Status-In--Progress-orange)](#)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This project focuses on identifying logistical bottlenecks and predicting delivery risks within the **DataCo Global Supply Chain** dataset. Our team is leveraging advanced Exploratory Data Analysis (EDA) and Machine Learning to transform raw shipping data into actionable business intelligence.

### The Team
Lead Data Visualization & Cleaning, Supervised Learning & Risk Prediction and Unsupervised Clustering & Segmentation.
* **Francisco (F)**
* **Johnny (J)**
* **Marcos (M)**

### Team Collaboration & Strategic Goals

Our workflow is built on a circular feedback loop where EDA insights directly inform model architecture:

| Feature Lead | Primary Strategic Goal | Technical Ownership |
| :--- | :--- | :--- |
| **F, J & M** | **Data Integrity & Visualization** | Optimization of high-cardinality features (Zipcodes) and multi-variate correlation analysis to reduce noise. |
| **F, J & M** | **Predictive Risk Assessment** | Developing a Supervised Random Forest model to minimize 'Late Delivery' overhead for the business. |
| **F, J & M** | **Strategic Segmentation** | Utilizing Unsupervised K-Means clustering to identify high-value/low-risk regional hubs. |

### Integrated Workflow
1. **Pre-processing (The Team):** Engineering the `data/processed` layer to ensure consistent scaling and normalization for all models.
2. **Modeling (The Team):** Implementing both Supervised and Unsupervised approaches to provide a 360-degree view of Supply Chain health.
3. **Synthesis (The Team):** Combining individual metrics into a unified project dashboard.

---

## Key Discoveries (EDA Phase)

### Geographical Concentration
Through coordinate mapping, we discovered a significant data cluster centered in **Puerto Rico (Caguas)**. This localized concentration is a primary factor in the supply chain's performance metrics.

### Addressing High Cardinality
We successfully engineered a solution for high-cardinality features such as `Customer_Zipcode`. By implementing a custom factorization and JSON mapping system, we preserved data integrity while optimizing the dataset for model training.

### Multicollinearity Management
Our correlation analysis identified extreme redundancy in financial metrics (Sales vs. Price vs. Profit). We streamlined the feature set by dropping variables with a **1.00 correlation**, reducing model noise and improving computational efficiency.

## Tech Stack & Directory Structure
* **Environment**: GitHub Codespaces
* **Libraries**: Pandas, Seaborn, Matplotlib, Scikit-Learn, Pillow (PIL)

```text
.
├── README.es.md
├── README.md
├── data
│   ├── interim                          # Category mappings and temporary dictionaries
│   │   ├── categories_preview.png
│   │   ├── category_mappings.json
│   │   ├── heatmap_preview.png
│   │   └── preview.jpg                  # Project visual overview
│   ├── processed                        # Cleaned, factorized, and normalized data
│   └── raw
│       └── DataCoSupplyChainDataset.csv # Original DataCo dataset
├── learn.json
├── models                               # Saved .pkl models (Supervised & Unsupervised)
├── requirements.txt
└── src
    ├── EDA.ipynb                        # Primary EDA & Visualization
    ├── app.py
    ├── supply_chain_logistics.db
    └── utils.py                         # Helper functions
   
