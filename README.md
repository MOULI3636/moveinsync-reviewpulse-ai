# ReviewPulse AI — MoveInSync Review Intelligence

**ReviewPulse AI** is a review-analysis and rating-improvement project built to explore customer feedback from MoveInSync's Google Play Store reviews. It organizes review data, analyzes ratings and sentiment, identifies recurring complaint themes, prioritizes issues, and presents findings through an interactive dashboard.

> **Project goal:** Turn customer reviews into actionable product and operational insights by highlighting where users are dissatisfied and what teams should investigate first.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dashboard Sections](#dashboard-sections)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Run the Streamlit Dashboard](#run-the-streamlit-dashboard)
- [Fetch Reviews](#fetch-reviews)
- [How the Analysis Works](#how-the-analysis-works)
- [Outputs](#outputs)
- [Important Notes](#important-notes)
- [Future Improvements](#future-improvements)

---

## Overview

ReviewPulse AI combines review-data preparation and analysis with an interactive dashboard. It is designed to help investigate:

- What proportion of reviews are positive, neutral, or negative?
- Which complaint categories appear most often?
- Which issues are associated with the lowest star ratings?
- How do ratings and complaint patterns change over time or across app versions?
- Which issues should be investigated first based on frequency, severity, trend, and confidence?
- How can teams measure whether a fix improves the customer experience?

The project supports evidence-led investigation. Review themes and model outputs are signals for further validation—not proof of root cause on their own.

## Key Features

- **Review data collection** using a Google Play review-fetching script.
- **Dataset organization** with current/raw and organized CSV files.
- **Rating analysis** including rating distribution, mean rating, and low-rating share.
- **Sentiment analysis** to classify review text as positive, neutral, or negative.
- **Complaint-theme analysis** to group feedback into issue categories.
- **Root-cause investigation support** that separates observed review evidence from hypotheses to test.
- **Issue prioritization** using a weighted scoring approach.
- **Trend and forecast views** for exploring historical ratings and status-quo projections.
- **Rating-improvement simulator** for assumption-based scenarios.
- **Interactive dashboard** with filters and review exploration.
- **Exportable analysis/report outputs**, depending on the features implemented in the application.

## Dashboard Sections

The dashboard may include the following analysis views:

1. **Overview** — key performance indicators and an executive summary.
2. **Dataset Explorer** — searchable review records and derived fields.
3. **Rating Analysis** — rating distribution, statistics, and trends.
4. **Sentiment Analysis** — sentiment proportions and alignment with star ratings.
5. **Complaint Themes** — categorized issues and exploratory clusters.
6. **Root-Cause Analysis** — observed evidence, hypotheses, and telemetry checks.
7. **Trends & Forecasts** — historical patterns and short-horizon projections.
8. **Priority Matrix** — ranked issues and impact-versus-effort view.
9. **Recommendations** — suggested engineering and operational actions.
10. **Rating Simulator** — assumption-driven rating scenarios.
11. **Export Reports** — report/data export options, where available.
12. **Settings & Methodology** — analysis methods, assumptions, and limitations.

---

## Project Structure

```text
Movinsync Project/
├── app.py                         # Streamlit application entry point
├── reviewpulse_ai.py              # ReviewPulse AI analysis/dashboard application
├── fetch_moveinsync_reviews.py    # Script to fetch Google Play Store reviews
├── moveinsync_current_reviews.csv # Current/raw review dataset
├── moveinsync_organized_reviews.csv # Cleaned and organized review dataset
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

### File descriptions

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point for running the web dashboard. |
| `reviewpulse_ai.py` | Main ReviewPulse AI analysis application. Confirm which UI/features are in this file before deployment. |
| `fetch_moveinsync_reviews.py` | Collects public Google Play Store reviews, subject to the script's configured package ID, country, language, and fetch limit. |
| `moveinsync_current_reviews.csv` | Current/raw review data collected for analysis. |
| `moveinsync_organized_reviews.csv` | Organized review dataset used by the analysis/dashboard. |
| `requirements.txt` | Lists Python packages required by the project. |
| `README.md` | Project overview, setup instructions, and documentation. |

> **Note:** The structure above follows the files visible in the project screenshot. If your actual filenames or application responsibilities differ, update this section to match the code before publishing.

## Technology Stack

The current `requirements.txt` lists these dependencies:

- **Python** — application language
- **Streamlit** — web dashboard framework
- **Pandas** — tabular data processing
- **NumPy** — numerical operations
- **Scikit-learn** — machine-learning utilities
- **NLTK** — natural-language processing and sentiment-related utilities
- **Plotly** — interactive visualizations
- **OpenPyXL** — Excel file support
- **ReportLab** — PDF/report generation

Exact features depend on the code in the repository and the installed package versions.

## Dataset

The project includes two CSV files:

- `moveinsync_current_reviews.csv` — current/raw reviews.
- `moveinsync_organized_reviews.csv` — organized data intended for analysis.

The organized dataset used during analysis contained **2,000 review records**. The dashboard reported an average rating of approximately **3.86★**, with **26.55%** of the sample rated 1–2 stars. These are sample-specific results and can change when the dataset is refreshed.

Typical review fields may include review text, star rating, review date/time, app version, helpful votes, and review identifiers. Check the actual CSV headers before relying on any field names.

### Data and privacy

- Use public review data responsibly and comply with the source platform's terms.
- Avoid publishing unnecessary reviewer-identifying information.
- Do not commit credentials, access tokens, or private configuration files.
- If repository size is a concern, consider using Git LFS or providing a download/source script instead of committing large datasets.

## Getting Started

### 1. Clone the repository

Replace the placeholder URL with your actual GitHub repository URL.

```bash
git clone https://github.com/<your-username>/moveinsync-reviewpulse-ai.git
cd moveinsync-reviewpulse-ai
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal—usually `http://localhost:8501`. Open that URL in your browser.

If the dashboard expects a particular CSV filename or path, keep the expected dataset in the project directory or update the path in the code.

## Fetch Reviews

The repository contains `fetch_moveinsync_reviews.py` for collecting reviews. Run it with:

```bash
python fetch_moveinsync_reviews.py
```

Before running, inspect the script to confirm the configured app package ID, country, language, sorting order, review limit, and output filename. Review-fetching libraries and public endpoints can change; handle errors and verify the fetched data before using it in analysis.

## How the Analysis Works

The dashboard screenshots supplied for this project describe the following approach:

1. **Data ingestion and validation** — match expected columns, parse star ratings, and handle review timestamps.
2. **Sentiment analysis** — use VADER/lexicon-based sentiment with a fallback approach described by the application.
3. **Complaint categorization** — use rule-/keyword-led categories and exploratory text clustering.
4. **Prioritization** — rank issue categories using frequency, severity, recent growth, and confidence.
5. **Forecasting and simulation** — display short-horizon rating projections and assumption-driven rating-improvement scenarios.
6. **Action planning** — map high-priority themes to engineering or operational investigations.

These methods should be confirmed against the implementation in the repository. Text classification can misinterpret sarcasm, short comments, and multilingual reviews. Root-cause hypotheses should be validated using internal logs, telemetry, controlled reproduction, and before/after measurements.

## Outputs

Depending on the implemented code, the project can be used to produce:

- Interactive review-analysis dashboard
- Rating distribution and sentiment charts
- Complaint-theme summaries
- Prioritized issue list
- Trend/forecast and scenario views
- Exported tables or reports, if supported by the application

## Important Notes

- Forecasts and rating-simulator outputs are **assumption-driven estimates**, not guarantees.
- A review category indicates a reported experience; it does not prove the underlying technical cause.
- The included CSVs may become stale. Re-run the fetch script and review data quality before updating results.
- Confirm the dashboard works from a fresh environment before publishing or deploying.
- Do not commit `__pycache__/`, `.venv/`, `.env`, or other generated/private files.

## Future Improvements

- Add automated data-quality checks and duplicate-review handling.
- Add unit tests for rating calculations, category assignment, and filtering.
- Evaluate sentiment and low-rating models with precision, recall, F1-score, and calibration.
- Add release/version annotations to trend charts.
- Connect review themes to issue-tracking workflows.
- Add deployment instructions and a hosted demo link once available.

---

## Author

**Mouli Banerjee**  
Registration No.: `12300455`  
Email: `banerjeemouli26@gmail.com`  
Phone: `+91 7602927810`

---

*Built as a review-intelligence project to turn public customer feedback into structured analysis and actionable investigation priorities.*
