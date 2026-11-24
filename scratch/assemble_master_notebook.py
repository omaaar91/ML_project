import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

print("Assembling Master Capstone Notebook...")

master_nb = new_notebook()

# Master Title Cell
master_nb.cells.append(new_markdown_cell("""# Nile Connect - Telecom Customer Intelligence
## Machine Learning Capstone Project (Full Integrated Master Submission)
**Diploma:** AI & Data Science Diploma  
**Company:** Nile Connect (Egyptian Telecom Operator)  
**Deliverable:** 1 Complete Notebook Covering Parts 1 through 8 (100 Points)  
**Tracking:** Integrated MLflow Tracking (`sqlite:///E:/tik tok/ML_project/mlflow.db`)

---
### Capstone Project Structure:
- **Part 1:** Know your data and clean it (Data Audit & EDA across 3 datasets)
- **Part 2:** Predict the monthly bill (Simple & Multiple Linear Regression)
- **Part 3:** Who will leave? (7 Churn Classification Models & Business Cost Analysis)
- **Part 4:** Fair comparison and tuning (5-Fold Stratified CV, GridSearch, MLflow)
- **Part 5:** Customer Segments & Dimensionality Reduction (KMeans, HAC, DBSCAN, PCA)
- **Part 6:** Which services go together? (Market Basket Analysis & Apriori Rules)
- **Part 7:** Put the two worlds together (Supervised + Unsupervised Combined)
- **Part 8:** Final Executive Story & Actionable Recommendations for the CEO
"""))

# Load notebooks
nb1_path = 'Notebooks/Cleancustomer.ipynb'
nb2_path = 'Notebooks/Cleanusage_behavior.ipynb'
nb3_path = 'Notebooks/Clean_service_transactions.ipynb'
nb4_path = 'Notebooks/Part7_combine_supervised_unsupervised.ipynb'

with open(nb1_path, 'r', encoding='utf-8') as f:
    nb1 = nbformat.read(f, as_version=4)

with open(nb2_path, 'r', encoding='utf-8') as f:
    nb2 = nbformat.read(f, as_version=4)

with open(nb3_path, 'r', encoding='utf-8') as f:
    nb3 = nbformat.read(f, as_version=4)

with open(nb4_path, 'r', encoding='utf-8') as f:
    nb4 = nbformat.read(f, as_version=4)

print(f"Notebook 1 cells: {len(nb1.cells)}")
print(f"Notebook 2 cells: {len(nb2.cells)}")
print(f"Notebook 3 cells: {len(nb3.cells)}")
print(f"Notebook 4 cells: {len(nb4.cells)}")

# 1. Append Notebook 1 (Parts 1-4)
master_nb.cells.extend(nb1.cells)

# 2. Append Notebook 2 (Part 5) with a separator markdown
master_nb.cells.append(new_markdown_cell("""---
# PART 5: Customer Segmentation & Dimensionality Reduction (Clustering + PCA)
**File:** `usage_behavior.csv` (2,000 customers, 13 features)  
**Unsupervised Exploration:** KMeans, HAC (Ward vs Complete), DBSCAN Anomaly Detection, and PCA
---"""))
master_nb.cells.extend(nb2.cells)

# 3. Append Notebook 3 (Part 6) with a separator markdown
master_nb.cells.append(new_markdown_cell("""---
# PART 6: Which Services Go Together? (Association Rules & Apriori)
**File:** `service_transactions.csv` (12,993 purchase events, 4,500 baskets, 13 services)  
**Market Basket Analysis:** Apriori, Association Rules (Support, Confidence, Lift), and Bundle Recommendations
---"""))
master_nb.cells.extend(nb3.cells)

# 4. Append Notebook 4 (Part 7) with a separator markdown
master_nb.cells.append(new_markdown_cell("""---
# PART 7: Put the Two Worlds Together (Supervised + Unsupervised Combined)
**Files:** `customers.csv` + `usage_behavior.csv` (2,000 shared customers)  
**Integrated Evaluation:** Segment-Level Churn Analysis and Supervised Model Retraining
---"""))
master_nb.cells.extend(nb4.cells)

# 5. Append Part 8 (Executive Report Markdown)
master_nb.cells.append(new_markdown_cell("""---
# PART 8: Tell the Story (Final Executive Report for the CEO)
*Written for the CEO of Nile Connect, answering the 4 core business questions with rigorous data evidence.*

### 1. Churn Intelligence: "About 1 in 4 customers has left. Who is next?"
- **The Finding:** 25.35% overall churn rate. Churn is concentrated in **Month-to-Month contracts (85% of churners)**, customers with **3+ support calls**, and early tenure (< 12 months).
- **The Number:** Our Tuned Logistic Regression model achieves **83.97% Recall** on test data, capturing 84 out of 100 churners and delivering the lowest financial business loss (**79,900 EGP** vs. 234,000 EGP baseline loss).
- **Recommended Action:** Automate real-time alerts when a customer logs 3 support calls; offer a proactive 15% discount on converting to an annual contract.

### 2. Billing Transparency: "Why does one customer pay 200 EGP and another 600 EGP?"
- **The Finding:** Billing is governed by 4 transparent drivers with an **R² of 0.841** and an MAE of **28.4 EGP**.
- **The Number:** Base Plan (+174 EGP for Premium), Data Usage (+5.1 EGP/GB), Home Internet (+82 EGP), and Extra Services (+22 EGP/service).
- **Recommended Action:** Launch an interactive "Bill Explainer" in the Nile Connect mobile app itemizing these exact four levers to eliminate bill shock.

### 3. Customer Personas: "Who are our customers, really?"
- **The Finding:** Consensus clustering (KMeans k=5, HAC Ward ARI=0.9985, PCA 5 components = 92.4% variance) identifies 5 distinct personas.
- **The Personas & Tailored Offers:**
  1. *Night Gamers & Streamers (26.3%):* "Nile Night Rush" uncapped data 12 AM - 8 AM (80 EGP/mo).
  2. *Budget / Light Users (22.7%):* "Nile Smart Saver" micro-bundle 2 GB + 200 Flex min (45 EGP/mo).
  3. *Social Media Addicts (28.3%):* "Nile Infinite Social Pass" unlimited social media (110 EGP/mo).
  4. *Business Jetsetters (20.9%):* "Nile Global Passport" 1500 local min + 60 intl min + roaming (350 EGP/mo).
  5. *Extreme Power Users (1.9% - 36.8% Churn Risk!):* "VIP Dedicated 5G Line" with corporate SLA (550 EGP/mo).

### 4. Bundling Intelligence: "Which services do people buy together?"
- **The Finding:** 112 association rules mined from 4,500 baskets. Discarded high-confidence rules with Lift ~ 1 (e.g. Roaming -> Mobile Data Pack) as independent baseline noise.
- **Top 3 High-Affinity Commercial Bundles:**
  1. *Global Traveler Pass:* `Intl_Calling` &harr; `Roaming_Pack` (Conf: **92.2%**, Lift: **8.96x**).
  2. *Smart Home Ultra Connect:* `Router_Upgrade` &harr; `Home_Internet` (Conf: **93.1%**, Lift: **2.41x**).
  3. *Total Device Shield & Care:* `Device_Installment` &harr; `Antivirus` (Conf: **83.7%**, Lift: **5.97x**).

### 5. Limits and Next Steps (Strategic Road-Map):
- **Data Gaps:** Absence of longitudinal time-series timestamps, network telemetry (ping jitter, packet loss), and CSAT/NPS feedback.
- **Future Actions:** Implement real-time automated churn scoring, collect monthly time-series cohorts for Survival Analysis, and deploy the Cosine Similarity Recommender Engine in the mobile self-service app.
"""))

master_path = 'Notebooks/Nile_Connect_ML_Capstone.ipynb'
with open(master_path, 'w', encoding='utf-8') as f:
    nbformat.write(master_nb, f)

print(f"Master Notebook successfully assembled at: {master_path}")
print(f"Total cells in master notebook: {len(master_nb.cells)}")
