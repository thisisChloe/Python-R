# Quantium Data Analytics — Chip Category Analysis

A end-to-end retail analytics project completed as part of the [Quantium Data Analytics Virtual Experience](https://www.theforage.com/simulations/quantium/data-analytics-rqkb) on Forage. The project covers customer segmentation, purchasing behaviour analysis, and store trial evaluation for a chip category, culminating in a strategic recommendation for the Category Manager.

## Skills Demonstrated

![Data Cleaning](https://img.shields.io/badge/Data%20Cleaning-BFDBFE?style=for-the-badge&labelColor=BFDBFE&color=BFDBFE) ![Data Validation](https://img.shields.io/badge/Data%20Validation-DDD6FE?style=for-the-badge) ![Exploratory Data Analysis](https://img.shields.io/badge/Exploratory%20Data%20Analysis-BBF7D0?style=for-the-badge) ![Feature Engineering](https://img.shields.io/badge/Feature%20Engineering-FED7AA?style=for-the-badge) ![Statistical Analysis](https://img.shields.io/badge/Statistical%20Analysis-FECACA?style=for-the-badge) ![Data Visualisation](https://img.shields.io/badge/Data%20Visualisation-99F6E4?style=for-the-badge) ![Commercial Thinking](https://img.shields.io/badge/Commercial%20Thinking-E5E7EB?style=for-the-badge) ![Presentation Skills](https://img.shields.io/badge/Presentation%20Skills-FEF08A?style=for-the-badge)

---

## Project Structure

```
├── task1-customer-analysis/
│   ├── QVI_chip_analysis.Rmd       # Customer & product analysis (R)
│   └── QVI_chip_analysis.ipynb     # Supplementary Python version
├── task2-store-trial/
│   └── QVI_task2_store_trial.Rmd   # Store trial evaluation (R)
├── task3-report/
│   ├── QVI_task3_report.Rmd        # Client-ready strategic report (R)
│   └── report_header.tex           # LaTeX styling
├── .gitattributes
├── .gitignore
└── README.md
```

> **Data files not included** — download the dataset from the [Forage platform](https://www.theforage.com/simulations/quantium/data-analytics-rqkb) or this [Google Drive folder](https://drive.google.com/drive/folders/1ggkySu0TDnP_VAZTdz6dTnQB0gvShfpz?usp=sharing).

---

## Task 1 — Customer Segment & Product Analysis

**Objective:** Understand who buys chips and what drives their spend.

**Approach:**
- Cleaned 264K+ transaction rows: removed salsa products, excluded a commercial buyer (200+ units/transaction), converted Excel serial dates
- Engineered `PACK_SIZE` (gram weight) and `BRAND` (first word, 8 variants standardised) from product names
- Merged with customer loyalty segment data (7 lifestages × 3 premium tiers = 21 segments)
- Defined metrics: total sales, customers, avg spend per customer, avg price per unit, avg transactions per customer

**Key Findings:**
- **Budget Older Families** generate the highest absolute revenue — bulk household buying
- **Mainstream Young Singles/Couples** are the standout growth segment: large customer base, high purchase frequency, and above-average willingness to pay despite not being classified as Premium
- **175g** is the dominant pack size; families index toward 330g–380g
- **Kettle** and **Smiths** lead on revenue across most segments

---

## Task 2 — Store Trial Evaluation (Stores 77, 86, 88)

**Objective:** Determine whether the Feb–Apr 2019 trial produced a statistically significant sales uplift and identify the driver.

**Approach:**
- Built monthly store-level metrics (total sales, unique customers, avg transactions per customer)
- Selected control stores using a combined score of **Pearson correlation** and **magnitude distance** on 7 months of pre-trial data
- Scaled control stores to each trial store's baseline level
- Tested significance using **paired t-tests** and 95% confidence intervals derived from pre-trial residuals

**Key Findings:**

| Trial Store | Control Store | Sales Uplift | Significant? | Driver |
|:-----------:|:-------------:|:------------:|:------------:|:-------|
| 77 | Selected by scoring | Positive | Yes | More customers |
| 86 | Selected by scoring | Minimal | No | — |
| 88 | Selected by scoring | Positive | Yes | More customers |

- Stores 77 and 88 both exceeded the confidence interval during the trial — uplift was **customer-driven** (more unique shoppers), not frequency-driven
- Store 86 showed no statistically significant change

---

## Task 3 — Strategic Report

**Objective:** Present findings and recommendations to the Category Manager using the **Pyramid Principles** framework (answer first → situation → findings → recommendation).

**Report covers:**
- Category KPIs and monthly sales trend
- Segment prioritisation matrix (spend per customer vs price per unit, sized by customer count)
- Brand and pack size analysis
- Trial results with confidence interval charts for all 3 stores
- Segmented recommendations for ranging, promotions, and trial rollout

---

## How to Run

**Requirements:** R 4.x, pandoc, xelatex (via TinyTeX)

```r
# Install packages
install.packages(c("tidyverse", "readxl", "lubridate", "scales",
                   "gridExtra", "patchwork", "rmarkdown", "tinytex"))
tinytex::install_tinytex()

# Render all three reports
Sys.setenv(RSTUDIO_PANDOC = "~/bin")   # path to pandoc if not on system PATH
rmarkdown::render("QVI_chip_analysis.Rmd",    output_format = "pdf_document")
rmarkdown::render("QVI_task2_store_trial.Rmd", output_format = "pdf_document")
rmarkdown::render("QVI_task3_report.Rmd",      output_format = "pdf_document")
```

---

## Tools & Libraries

- **R** — tidyverse, ggplot2, lubridate, scales, gridExtra, patchwork, rmarkdown
- **Python** — pandas, numpy, matplotlib, seaborn (supplementary notebook)
- **LaTeX** — xelatex + TinyTeX for PDF rendering


---

## 🥇 Certificate
[Quantium Certificate.pdf](https://github.com/user-attachments/files/29821682/Quantium.Certificate.pdf)
