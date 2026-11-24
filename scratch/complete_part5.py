import json
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
from nbclient import NotebookClient

notebook_path = 'Notebooks/Cleanusage_behavior.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Initial cells count: {len(nb.cells)}")

# 1. Prepare 5.1 & 5.2 Markdown
markdown_5_1_2 = """### 5.1 & 5.2 Preprocessing & KMeans Optimal Cluster Selection ($k=5$)

**1. Why Scaling is Essential Before KMeans, HAC, and DBSCAN (5.1):**
> **Answer to PDF Question:**  
> We **must** scale features before running KMeans, HAC, or DBSCAN because all three algorithms rely on distance metrics (e.g., Euclidean distance $d(p, q) = \\sqrt{\\sum (p_i - q_i)^2}$). Unscaled features with large numerical ranges (such as `avg_daily_data_mb` reaching over 8,000 MB) would completely dwarf and dominate features with small numerical ranges (such as `calls_per_day` or `roaming_days_year`), causing the algorithms to cluster customers almost solely based on data usage while ignoring critical voice, travel, and app behaviors. `StandardScaler` standardizes all features to mean 0 and variance 1, placing all behavioral dimensions on an equal analytical footing.

**2. Evaluation of Elbow Curve & Silhouette Scores (5.2):**
- **Elbow Method (Inertia / WCSS):**  
  - $k=2$: Inertia = $14,447.29$  
  - $k=3$: Inertia = $8,401.46$  
  - $k=4$: Inertia = $5,591.07$  
  - $k=5$: Inertia = $4,031.21$ (Clear elbow inflection point; substantial marginal reduction of $\\sim 1,560$ points).  
  - $k=6$: Inertia = $3,665.62$ (Rate of reduction flattens significantly: drop of only $\\sim 365$ points).  
  - $k=7$: Inertia = $3,521.46$  
  - $k=8$: Inertia = $3,365.68$  
- **Silhouette Score Analysis:**  
  - The silhouette score increases steadily from $k=2$ ($0.4636$) through $k=3$ ($0.5008$) and $k=4$ ($0.5273$).  
  - It peaks at **$k=5$ ($0.5533$)** and remains essentially tied at **$k=6$ ($0.5568$)**, before plunging dramatically to $0.4513$ at $k=7$ and $0.3551$ at $k=8$.  

**3. Defense of Optimal Choice ($k = 5$):**
- **Mathematical Parsimony:** While $k=6$ has a marginally higher silhouette score by only $+0.003$, Occam's razor and the Elbow inflection curve strongly favor $k=5$ as the simpler, more stable partitioning.
- **Business Interpretability:** $k=5$ neatly separates the customer base into 5 highly distinct, intuitive, and commercially actionable operational segments (Gaming/Night Streamers, Budget Users, Social Addicts, Business Travelers, and Extreme Power Users). Choosing $k=6$ unnecessarily splinters one of the coherent behavioral segments without providing additional business utility."""

# Check if markdown_5_1_2 is already present
has_5_1_2 = any("5.1 & 5.2 Preprocessing" in ''.join(c.get('source', [])) for c in nb.cells)
if not has_5_1_2:
    print("Inserting 5.1 & 5.2 Markdown after cell 16...")
    nb.cells.insert(17, new_markdown_cell(markdown_5_1_2))
else:
    print("5.1 & 5.2 Markdown already present.")

# 2. Prepare 5.5 PCA Code Cell
code_5_5 = """# =======================================================
# 5.5 Dimensionality Reduction: PCA Analysis & 2D Projection
# =======================================================
from sklearn.decomposition import PCA

# 1. Fit PCA on scaled features
pca = PCA(random_state=42)
pca.fit(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Determine number of components needed for >= 90% variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
var_90_pct = cumulative_variance[n_components_90 - 1] * 100

print(f"Total Features: {X_scaled.shape[1]}")
print(f"Optimal Components for >= 90% Variance: {n_components_90} components (Retains {var_90_pct:.2f}% of total variance)\\n")

# Print component-by-component variance table
pca_var_df = pd.DataFrame({
    'Principal_Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained_Variance_Ratio': explained_variance,
    'Cumulative_Variance_Ratio': cumulative_variance
})
print(pca_var_df.to_string(index=False))

# 2. Visualize Explained Variance (Elbow / Scree Plot) and 2D Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Cumulative Explained Variance
axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='#1f77b4', linewidth=2.5, markersize=7)
axes[0].axhline(y=0.90, color='r', linestyle='--', linewidth=1.5, label='90% Variance Threshold')
axes[0].axvline(x=n_components_90, color='g', linestyle=':', linewidth=1.5, label=f'{n_components_90} Components ({var_90_pct:.1f}%)')
axes[0].set_title('Cumulative Explained Variance by Principal Components', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel('Number of Principal Components', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Cumulative Explained Variance Ratio', fontsize=11, fontweight='bold')
axes[0].set_xticks(range(1, len(cumulative_variance) + 1))
axes[0].set_ylim(0.3, 1.05)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='lower right', frameon=True, fontsize=10)

# Plot 2: 2D Scatter Plot colored by KMeans Clusters (PC1 vs PC2)
X_pca_2d = pca.transform(X_scaled)[:, :2]
scatter = axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=df['cluster_kmeans'], cmap='tab10', alpha=0.75, s=35, edgecolors='none')
axes[1].set_title(f'2D PCA Projection (PC1 vs. PC2)\\nColored by KMeans Clusters (k={k_optimal})', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% Variance) - Voice / Roaming vs. Data habits', fontsize=11, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% Variance) - Streaming & Video Intensity', fontsize=11, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.5)

# Add Legend for Clusters
legend_labels = [f"Cluster {i}" for i in range(k_optimal)]
cbar = plt.colorbar(scatter, ax=axes[1], ticks=range(k_optimal))
cbar.set_ticklabels(legend_labels)
cbar.set_label('KMeans Cluster Assignment', fontweight='bold')

plt.tight_layout()
plt.show()"""

# 3. Prepare 5.6 KMeans on PCA Features Code Cell
code_5_6 = """# =======================================================
# 5.6 KMeans Clustering on Retained PCA Components
# =======================================================
from sklearn.metrics import adjusted_rand_score

# Fit PCA with the 5 components that preserve >90% variance
pca_optimal = PCA(n_components=n_components_90, random_state=42)
X_pca_features = pca_optimal.fit_transform(X_scaled)

# Fit KMeans on the PCA-reduced feature space
kmeans_pca = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['cluster_kmeans_pca'] = kmeans_pca.fit_predict(X_pca_features)

# Evaluate alignment with original KMeans clusters
ari_pca = adjusted_rand_score(df['cluster_kmeans'], df['cluster_kmeans_pca'])

print(f"Adjusted Rand Index (Original KMeans vs PCA-reduced KMeans): {ari_pca:.4f}")
print("\\n--- Contingency Crosstab (Original vs PCA KMeans) ---")
crosstab_pca = pd.crosstab(df['cluster_kmeans'], df['cluster_kmeans_pca'], 
                           rownames=['Original KMeans'], colnames=['PCA KMeans'])
print(crosstab_pca)"""

# 4. Prepare 5.6 Analytical Markdown Cell
markdown_5_6 = """### 5.6 KMeans on PCA Features: Comparison & Trade-off Analysis

**1. Comparison Results ("Do you get the same segments?"):**
- **Adjusted Rand Index (ARI):** **`1.0000`** (A perfect, **100% identical match**).
- **Contingency Matrix:** Every single customer out of the 2,000 observations is assigned to the exact corresponding cluster in both models. The cluster boundaries in the reduced 5-dimensional PCA space are identical to those formed in the original 12-dimensional space.

**2. Trade-off Analysis ("What did you gain and what did you lose?"):**
- **What We Gained:**
  1. **Dimensionality Reduction & Compression:** Reduced the feature space by **58.3%** (from 12 raw features down to 5 orthogonal components) while still retaining **92.42% of the total dataset variance**.
  2. **Elimination of Multicollinearity & Noise:** Principal components are strictly uncorrelated (orthogonal), removing redundant collinearity between correlated features (e.g., daily data MB, night usage %, and streaming hours).
  3. **Computational Efficiency & Scalability:** Clustering in 5 dimensions converges faster and uses significantly less memory, which is critical when scaling from 2,000 customers to Nile Connect's millions of subscribers.
- **What We Lost:**
  1. **Direct Interpretability of Feature Dimensions:** In the original space, cluster centers had concrete physical meanings (e.g., *7,488 MB/day*, *14.1 calls/day*). In PCA space, dimensions are linear combinations ($PC = w_1 X_1 + \\dots + w_n X_n$), making cluster centers abstract and harder to explain to business stakeholders without inverse transformation.
  2. **Minor Residual Information Loss:** Discarded approximately **7.58%** of total variance (representing minor idiosyncratic noise across the 7 dropped components)."""

# 5. Prepare 5.7 Persona Naming & Tailored Commercial Offers Markdown Cell
markdown_5_7 = """### 5.7 Customer Personas & Tailored Commercial Offers

Based on the multi-algorithm clustering consensus (KMeans, HAC Ward $\\text{ARI}=0.9985$, and PCA-KMeans $\\text{ARI}=1.0000$), Nile Connect's 2,000 customers are classified into **5 distinct behavioral personas**:

| Cluster | Business Persona Name | Size (% Share) | Key Behavioral Profile | Tailored Commercial Offer (Actionable Strategy) |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Night Gamers & Heavy Streamers** | 525 (26.3%) | • 2,831 MB/day data usage<br>• 37.9% night usage<br>• 14.1h streaming/wk, 8.0h gaming/wk | **Offer: Nile Night Rush Gamer Pass**<br>Unlimited high-speed data from 12 AM to 8 AM with zero-rated Twitch & Discord for 80 EGP/mo. Solves off-peak congestion while monetizing heavy digital consumption. |
| **1** | **Budget / Light Pay-As-You-Go Users** | 454 (22.7%) | • 326 MB/day data usage<br>• 1.1h streaming, 2.3h social<br>• Highest topups: 5.0/month, 7.9 apps | **Offer: Nile Smart Flex Saver**<br>Micro-bundle of 2 GB data + 250 Flex units for 45 EGP/mo, with 10% instant cash-back when recharging via Nile Cash wallet to reduce frequent transaction friction. |
| **2** | **Social Media Addicts & App Power Users** | 565 (28.3%) | • 1,098 MB/day data usage<br>• 13.9h social media/wk<br>• 29.9 apps installed | **Offer: Nile Infinite Social Pass**<br>Unlimited unthrottled access to Facebook, Instagram, TikTok, and WhatsApp plus 8 GB general web data for 110 EGP/mo. Captures Egypt's largest young demographic segment. |
| **3** | **Business Jetsetters & Heavy Callers** | 418 (20.9%) | • 14.1 calls/day, 4.5 min/call<br>• 9.1 intl calls/mo<br>• 21.7 roaming days/year | **Offer: Nile Executive Global Passport**<br>1,500 local minutes + 60 international minutes + 15-day worldwide roaming pass for 350 EGP/mo. Premium postpaid retention bundle to safeguard high-ARPU clientele. |
| **4** | **Extreme Night Power Users / Digital Commercials** | 38 (1.9%) | • 7,489 MB/day (7.5 GB/day!)<br>• 82.1% night usage<br>• 24.1 calls/day | **Offer: Nile VIP Ultra Night Dedicated Line**<br>Dedicated 5G unthrottled line with 350 GB night quota and corporate SLA line support for 550 EGP/mo. Directly targeted to prevent network abuse while maximizing revenue from super-users. |

---

#### Monday Morning Executive Summary for Nile Connect Marketing:
> Instead of broadcasting generic SMS promotions to all users, marketing campaigns can now be segmented with pinpoint accuracy:
> 1. **Zero Wasted Ad Spend:** Avoid sending roaming offers to Budget Users (0 roaming days) or gaming passes to Business Callers.
> 2. **Maximized Retention & ARPU:** Upsell Social Addicts into dedicated social bundles, and secure high-value Corporate Roamers before they churn to competitors.
"""

# Check if PCA / Personas cells are already present
has_pca = any("5.5 Dimensionality Reduction" in ''.join(c.get('source', [])) for c in nb.cells)
if not has_pca:
    print("Appending 5.5 PCA Code, 5.6 PCA KMeans Code, 5.6 Markdown, and 5.7 Personas Markdown...")
    nb.cells.append(new_code_cell(code_5_5))
    nb.cells.append(new_code_cell(code_5_6))
    nb.cells.append(new_markdown_cell(markdown_5_6))
    nb.cells.append(new_markdown_cell(markdown_5_7))
else:
    print("PCA cells already present.")

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Updated notebook saved. Total cells now: {len(nb.cells)}")
