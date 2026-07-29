# 🛍️ Mall Customer Segmentation Analysis

## 🎯 Project Overview

- **Objective:** Group mall customers into distinct segments based on income and spending behaviour, so marketing and retention efforts can be aimed at specific customer types instead of the customer base as a whole.
- **Dataset:** Mall_Customers.csv, 200 customers with CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).
- **Methodology:** Data loading and quality checks, descriptive statistics and exploratory plots, an age-group breakdown, k selection via the elbow method and silhouette score, k-means clustering on income and spending score, descriptive labelling of each cluster (including a grand-centroid check so the middle cluster doesn't collide with the four income/spending quadrant labels), and a gender breakdown within each cluster. Implemented in mall_analysis.py and mirrored in mall_analysis.ipynb.

## 🔍 Key Findings

- **Distributions:** age, income, and spending score are each spread fairly broadly across the customer base rather than concentrated around one value, and plotting income against spending score by gender doesn't show either gender clearly dominating a particular corner of that space.

![Exploratory analysis](viz/01_eda.png)

- **Choosing k:** the elbow method and silhouette score both point to k = 5 as a reasonable number of clusters, balancing simplicity against how well separated the groups are.

![Elbow method and silhouette score](viz/02_elbow_silhouette.png)

- **Five customer segments:** clustering on income and spending score produces five groups: a high-income, high-spending segment ("target"), a high-income, low-spending segment ("re-engage"), a low-income, high-spending segment, a low-income, low-spending segment, and a middle-income, average-spending segment sitting near the centre of the other four. The shaded regions below show which area of the income/spending space each cluster owns, not just the raw points.

![Cluster segments](viz/03_clusters.png)

- **Gender within clusters:** the gender mix looks broadly similar across most segments, without one gender clearly dominating any single income/spending group.

![Gender split within each cluster](viz/04_gender_cluster.png)

- **Spending by age group:** average spending score is broken down across five age bands (18-25, 26-35, 36-45, 46-60, 60+), with younger bands generally averaging higher spending scores than older ones.

![Average spending score by age group](viz/05_age_groups.png)

## 🚀 Suggested Uses

- **Prioritise the "target" segment:** high-income, high-spending customers are the most valuable group and are natural candidates for loyalty programmes and premium offers.
- **Re-engage high-income, low-spending customers:** this group has the means to spend more, so targeted promotions or understanding why they aren't converting could unlock real value.
- **Treat the middle segment as a baseline:** the middle-income, average-spending group is the closest thing to a "typical" customer here, useful as a comparison point when judging whether a campaign is shifting behaviour elsewhere.
- **Watch younger age bands for spending momentum:** if younger customers are already spending more on average, they may be worth investing in earlier for long-term loyalty.

## ⚠️ Notes

- All five charts in \`viz/\` are generated directly by \`mall_analysis.py\` (also runnable through \`mall_analysis.ipynb\`), so they update automatically whenever the clustering logic changes.
- Findings above are described qualitatively rather than with exact percentages, since they're based on visual inspection of the charts rather than a separately executed analysis.
