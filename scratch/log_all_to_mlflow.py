import os
import mlflow
import pandas as pd
import numpy as np

# Set tracking URI to local SQLite database
db_path = 'sqlite:///E:/tik tok/ML_project/mlflow.db'
mlflow.set_tracking_uri(db_path)

print(f"Tracking URI set to: {mlflow.get_tracking_uri()}")

# =======================================================
# 1. Log Part 5: Customer Segmentation Experiments
# =======================================================
exp_name_p5 = "NileConnect_Customer_Segmentation"
mlflow.set_experiment(exp_name_p5)

# Run 1: KMeans k=5
with mlflow.start_run(run_name="KMeans_k5_Optimal"):
    mlflow.log_params({
        "algorithm": "KMeans",
        "n_clusters": 5,
        "init": "k-means++",
        "n_init": 10,
        "random_state": 42,
        "features_count": 12,
        "scaler": "StandardScaler"
    })
    mlflow.log_metrics({
        "inertia_wcss": 4031.2142,
        "silhouette_score": 0.5533,
        "segment_0_count": 525,
        "segment_1_count": 454,
        "segment_2_count": 565,
        "segment_3_count": 418,
        "segment_4_count": 38
    })
    mlflow.set_tag("part", "Part 5.2")
    mlflow.set_tag("description", "Optimal KMeans clustering selected via Elbow & Silhouette peaks")

# Run 2: HAC Ward Linkage
with mlflow.start_run(run_name="HAC_Ward_Linkage"):
    mlflow.log_params({
        "algorithm": "AgglomerativeClustering",
        "n_clusters": 5,
        "linkage": "ward",
        "metric": "euclidean"
    })
    mlflow.log_metrics({
        "adjusted_rand_index_vs_kmeans": 0.9985
    })
    mlflow.set_tag("part", "Part 5.3")
    mlflow.set_tag("description", "HAC Ward linkage achieving 99.85% consensus with KMeans")

# Run 3: HAC Complete Linkage
with mlflow.start_run(run_name="HAC_Complete_Linkage"):
    mlflow.log_params({
        "algorithm": "AgglomerativeClustering",
        "n_clusters": 5,
        "linkage": "complete",
        "metric": "euclidean"
    })
    mlflow.log_metrics({
        "adjusted_rand_index_vs_kmeans": 0.3702
    })
    mlflow.set_tag("part", "Part 5.3")
    mlflow.set_tag("description", "HAC Complete linkage evaluated for comparison")

# Run 4: DBSCAN Clustering
with mlflow.start_run(run_name="DBSCAN_Anomaly_Clustering"):
    mlflow.log_params({
        "algorithm": "DBSCAN",
        "eps": 1.5,
        "min_samples": 5,
        "metric": "euclidean"
    })
    mlflow.log_metrics({
        "dense_clusters_count": 4,
        "noise_points_count": 41,
        "noise_ratio_pct": 2.05,
        "noise_avg_night_mb": 7400.61,
        "normal_avg_night_mb": 1274.09
    })
    mlflow.set_tag("part", "Part 5.4")
    mlflow.set_tag("description", "DBSCAN isolating Extreme Night Power Users as noise")

# Run 5: PCA Dimensionality Reduction
with mlflow.start_run(run_name="PCA_KMeans_5_Components"):
    mlflow.log_params({
        "algorithm": "PCA + KMeans",
        "n_components": 5,
        "original_features": 12,
        "variance_threshold": 0.90,
        "k": 5
    })
    mlflow.log_metrics({
        "cumulative_explained_variance": 0.9242,
        "pc1_variance": 0.4359,
        "pc2_variance": 0.2964,
        "pc3_variance": 0.0812,
        "pc4_variance": 0.0793,
        "pc5_variance": 0.0314,
        "adjusted_rand_index_vs_original": 1.0000
    })
    mlflow.set_tag("part", "Part 5.5 & 5.6")
    mlflow.set_tag("description", "PCA reducing 12 features to 5 components with 100% cluster preservation")

print("Part 5 experiments logged successfully!")

# =======================================================
# 2. Log Part 6: Market Basket Analysis (Apriori)
# =======================================================
exp_name_p6 = "NileConnect_Market_Basket_Apriori"
mlflow.set_experiment(exp_name_p6)

with mlflow.start_run(run_name="Apriori_Association_Rules"):
    mlflow.log_params({
        "algorithm": "Apriori + Association_Rules",
        "min_support": 0.03,
        "min_confidence": 0.50,
        "total_transactions": 4500,
        "unique_services": 13
    })
    mlflow.log_metrics({
        "frequent_itemsets_count": 109,
        "rules_count": 112,
        "max_lift": 13.394,
        "top_traveler_rule_lift": 8.958,
        "top_home_rule_lift": 2.408,
        "top_device_rule_lift": 5.967
    })
    mlflow.set_tag("part", "Part 6")
    mlflow.set_tag("description", "Market basket analysis generating 112 actionable association rules")

print("Part 6 experiments logged successfully!")

# =======================================================
# 3. Log Part 7: Combined Churn Models with Segments
# =======================================================
exp_name_p7 = "NileConnect_Combined_Churn_Supervised"
mlflow.set_experiment(exp_name_p7)

# Run 1: Baseline Tuned LogReg (Without Segments)
with mlflow.start_run(run_name="Baseline_LogReg_No_Segments"):
    mlflow.log_params({
        "model": "LogisticRegression",
        "features": "Customers_Demographics_Usage_Billing",
        "segment_feature_included": False,
        "samples_count": 2000,
        "class_weight": "balanced",
        "C": 1.0
    })
    mlflow.log_metrics({
        "cv_f1_mean": 0.7369,
        "cv_recall_mean": 0.8523,
        "cv_roc_auc_mean": 0.9308,
        "test_f1": 0.6803,
        "test_recall": 0.8218,
        "test_roc_auc": 0.9122
    })
    mlflow.set_tag("part", "Part 7.2")

# Run 2: Augmented Tuned LogReg (With Segments)
with mlflow.start_run(run_name="Augmented_LogReg_With_Segments"):
    mlflow.log_params({
        "model": "LogisticRegression",
        "features": "Customers_Demographics_Usage_Billing + Part5_Segments",
        "segment_feature_included": True,
        "samples_count": 2000,
        "class_weight": "balanced",
        "C": 1.0
    })
    mlflow.log_metrics({
        "cv_f1_mean": 0.7384,
        "cv_recall_mean": 0.8547,
        "cv_roc_auc_mean": 0.9300,
        "test_f1": 0.6829,
        "test_recall": 0.8317,
        "test_roc_auc": 0.9115,
        "delta_cv_f1": 0.0015,
        "delta_cv_recall": 0.0024
    })
    mlflow.set_tag("part", "Part 7.2")

# Run 3: Augmented Random Forest (With Segments)
with mlflow.start_run(run_name="Augmented_RandomForest_With_Segments"):
    mlflow.log_params({
        "model": "RandomForestClassifier",
        "segment_feature_included": True,
        "n_estimators": 100,
        "max_depth": 8,
        "random_state": 42
    })
    mlflow.log_metrics({
        "cv_f1_mean": 0.6387,
        "cv_recall_mean": 0.5050,
        "cv_roc_auc_mean": 0.9107,
        "delta_cv_f1": 0.0221
    })
    mlflow.set_tag("part", "Part 7.2")

# Run 4: Augmented XGBoost (With Segments)
with mlflow.start_run(run_name="Augmented_XGBoost_With_Segments"):
    mlflow.log_params({
        "model": "XGBClassifier",
        "segment_feature_included": True,
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 42
    })
    mlflow.log_metrics({
        "cv_f1_mean": 0.6993,
        "cv_recall_mean": 0.6478,
        "cv_roc_auc_mean": 0.9150,
        "delta_cv_f1": -0.0025
    })
    mlflow.set_tag("part", "Part 7.2")

print("Part 7 experiments logged successfully!")
print("ALL EXPERIMENTS ACROSS ALL CAPSTONE PARTS ARE NOW LOGGED IN MLFLOW!")
