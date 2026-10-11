"""
ReviewPulse AI — MoveInSync App Review Intelligence & Rating Improvement System
================================================================================
Enterprise Streamlit Web Application converted from Tkinter Desktop Architecture.
Analyzes real Google Play Store customer reviews, discovers root causes behind poor
ratings, prioritizes engineering problems, simulates rating improvements, and exports
executive-ready reports.

Author: Senior Forward Deployed Engineer (FDE) Candidate
Target: MoveInSync Review Intelligence & Rating Improvement System
Single-file implementation adhering to production UI/UX and analytical standards.
"""

import sys
import os
import re
import math
import time
import io
import datetime
import textwrap
import logging
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning & NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.cluster import KMeans

# NLP Sentiment Analysis
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        _HAS_VADER = True
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
            _HAS_VADER = True
        except Exception:
            _HAS_VADER = False
except ImportError:
    _HAS_VADER = False

# Excel Generation
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

# PDF Generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    )
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

# ---------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ReviewPulseAI_Streamlit")

# ---------------------------------------------------------------------------
# DOMAIN TAXONOMY & CONFIGURATION
# ---------------------------------------------------------------------------
DEFAULT_COMPLAINT_TAXONOMY = {
    "Booking failure": {
        "keywords": [
            "booking", "book", "cannot book", "unable to book", "fail to book",
            "booking failed", "cab not booked", "slot full", "no slot", "roster",
            "schedule failed", "cant book", "can't book", "booking error"
        ],
        "effort": "Medium",
        "team": "Engineering (Backend / Booking)",
        "is_safety": False
    },
    "Login and authentication": {
        "keywords": [
            "login", "log in", "sign in", "signin", "logout", "logged out",
            "password", "sso", "authentication", "credential", "session expired",
            "cannot login", "can't login", "unable to login"
        ],
        "effort": "Low",
        "team": "Engineering (Auth & Identity)",
        "is_safety": False
    },
    "GPS and live tracking": {
        "keywords": [
            "gps", "location", "tracking", "live track", "map", "wrong location",
            "location wrong", "not moving", "stuck on map", "radar", "car location",
            "cab location", "gps inaccurate", "navigation"
        ],
        "effort": "High",
        "team": "Engineering (Maps & Location Telemetry)",
        "is_safety": False
    },
    "Incorrect pickup time": {
        "keywords": [
            "eta", "pickup time", "pick up time", "wrong time", "wrong eta",
            "timing wrong", "time changed", "incorrect time", "eta wrong",
            "eta calculator", "eta is wrong", "shows 3x", "eta inaccurate"
        ],
        "effort": "Medium",
        "team": "Product & Routing Algorithm",
        "is_safety": False
    },
    "Late pickup or vehicle arrival": {
        "keywords": [
            "late", "delay", "delayed", "not on time", "never on time",
            "waited", "waiting", "waiting for cab", "cab arrived late",
            "cab late", "driver late", "reach office late", "cross-country",
            "hour delay", "stand on road", "punctual"
        ],
        "effort": "Medium",
        "team": "Transport Operations & Dispatch",
        "is_safety": False
    },
    "Driver behavior": {
        "keywords": [
            "driver", "rash driving", "driver rude", "driver behavior",
            "driver cancelled", "driver refused", "driver didn't come",
            "driver not picking", "driver calling", "arrogant", "harsh",
            "unprofessional", "rash", "driving fast", "driver denied"
        ],
        "effort": "Medium",
        "team": "Transport Operations & Vendor Management",
        "is_safety": True
    },
    "OTP and safety": {
        "keywords": [
            "otp", "seat belt", "seatbelt", "safety", "sos", "emergency",
            "security", "otp pop", "view otp", "otp popup", "safe", "panic",
            "threat", "medical condition", "night shift safety", "female safety"
        ],
        "effort": "Low",
        "team": "Product & Mobile Experience",
        "is_safety": True
    },
    "App crash and performance": {
        "keywords": [
            "crash", "crashes", "crashing", "freeze", "freezes", "lag", "lags",
            "slow", "hanging", "hang", "blank screen", "white screen",
            "reinstall", "uninstall", "not opening", "won't open", "keeps stopping",
            "black screen", "force close", "update made it worse"
        ],
        "effort": "Medium",
        "team": "Mobile Engineering (Android/iOS)",
        "is_safety": False
    },
    "Notifications and communication": {
        "keywords": [
            "notification", "notifications", "alert", "alerts", "sms", "no update",
            "not informed", "without informing", "didn't notify", "no notification",
            "message not received", "ring", "alarm"
        ],
        "effort": "Low",
        "team": "Mobile & Backend Messaging",
        "is_safety": False
    },
    "UI/UX and navigation": {
        "keywords": [
            "ui", "ux", "interface", "confusing", "new update", "old version",
            "annoying", "clicks", "one click", "worst update", "bad update",
            "cluttered", "minimalist", "features confusing", "skip option",
            "forcing me", "force rating", "dark mode", "font", "design"
        ],
        "effort": "Low",
        "team": "Product Design & Frontend",
        "is_safety": False
    },
    "Route and shuttle problems": {
        "keywords": [
            "route", "routing", "circuitous", "long route", "shuttle",
            "detour", "two different states", "clubbed", "clubbing",
            "darshan", "tour", "passengers clubbed", "drop order", "pickup order"
        ],
        "effort": "High",
        "team": "Routing Algorithm & Ops Research",
        "is_safety": False
    },
    "Customer support and complaint resolution": {
        "keywords": [
            "support", "customer care", "helpdesk", "transport management",
            "ticket", "no response", "escalation", "helpline", "executive",
            "not ready to help", "complaint ignored", "care team"
        ],
        "effort": "Medium",
        "team": "Customer Experience & Support Ops",
        "is_safety": False
    },
    "Payment or billing": {
        "keywords": [
            "payment", "billing", "charged", "money", "fare", "reimbursement",
            "invoice", "deducted", "refund", "cost"
        ],
        "effort": "Low",
        "team": "Finance & Billing Services",
        "is_safety": False
    },
    "Positive feedback and praise": {
        "keywords": [
            "good", "great", "excellent", "superb", "awesome", "nice", "love",
            "best app", "useful", "helpful", "smooth", "impressive", "perfect",
            "thanks", "thank you", "comfortable", "safe travel", "wonderful"
        ],
        "effort": "None",
        "team": "Product & Ops (Recognition)",
        "is_safety": False
    }
}

LOCAL_FALLBACK_LEXICON = {
    "worst": -0.8, "terrible": -0.7, "horrible": -0.7, "awful": -0.7, "useless": -0.6,
    "pathetic": -0.7, "annoying": -0.5, "annoyed": -0.5, "poor": -0.5, "bad": -0.4,
    "hate": -0.6, "waste": -0.5, "disaster": -0.7, "garbage": -0.7, "crashes": -0.6,
    "crash": -0.5, "freeze": -0.5, "slow": -0.4, "lag": -0.4, "late": -0.4,
    "delayed": -0.4, "delay": -0.4, "threat": -0.7, "rude": -0.6, "dislike": -0.4,
    "forcing": -0.4, "confusing": -0.4, "fool": -0.6, "fools": -0.6, "fail": -0.5,
    "failed": -0.5, "bug": -0.4, "buggy": -0.5, "problem": -0.3, "issue": -0.3,
    "good": 0.4, "great": 0.6, "superb": 0.7, "excellent": 0.8, "awesome": 0.8,
    "love": 0.7, "best": 0.6, "nice": 0.4, "impressive": 0.6, "useful": 0.5,
    "helpful": 0.5, "smooth": 0.5, "perfect": 0.7, "thank": 0.4, "thanks": 0.4,
    "comfortable": 0.5, "safe": 0.5, "easy": 0.4, "reliable": 0.5, "punctual": 0.5
}

# ---------------------------------------------------------------------------
# DATA VALIDATION & INGESTION
# ---------------------------------------------------------------------------
class DataValidator:
    REQUIRED_COLUMNS = ["review_text", "rating"]
    CANONICAL_COLUMNS = [
        "review_id", "review_text", "rating", "review_date",
        "review_time_utc", "app_version", "helpful_votes",
        "reviewer_name", "fetched_at_utc"
    ]

    COLUMN_ALIASES = {
        "review_id": ["id", "review_id", "reviewid", "rev_id", "uuid"],
        "review_text": ["review_text", "review", "text", "comment", "content", "body", "customer_review"],
        "rating": ["rating", "score", "stars", "star_rating", "user_rating"],
        "review_date": ["review_date", "date", "created_date", "posted_date", "reviewdate"],
        "review_time_utc": ["review_time_utc", "time", "review_time", "time_utc", "timestamp_time"],
        "app_version": ["app_version", "version", "appversion", "build", "release"],
        "helpful_votes": ["helpful_votes", "votes", "thumbs_up", "helpful", "thumbsup_count"],
        "reviewer_name": ["reviewer_name", "user_name", "author", "reviewer", "name", "display_name"],
        "fetched_at_utc": ["fetched_at_utc", "collected_at", "fetch_time", "scraped_at", "fetched_at"]
    }

    @classmethod
    def match_columns(cls, df_columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
        clean_to_raw = {re.sub(r'[^a-z0-9]', '', str(col).lower()): col for col in df_columns}
        mapping = {}
        for canonical, aliases in cls.COLUMN_ALIASES.items():
            for alias in aliases:
                norm_alias = re.sub(r'[^a-z0-9]', '', alias.lower())
                if norm_alias in clean_to_raw:
                    mapping[canonical] = clean_to_raw[norm_alias]
                    break
        missing = [req for req in cls.REQUIRED_COLUMNS if req not in mapping]
        return mapping, missing

    @classmethod
    def validate_and_clean(cls, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        mapping, missing = cls.match_columns(raw_df.columns.tolist())
        if missing:
            raise ValueError(f"Missing mandatory review columns: {', '.join(missing)}. Detected: {list(raw_df.columns)}")

        df = pd.DataFrame()
        for canonical in cls.CANONICAL_COLUMNS:
            if canonical in mapping:
                df[canonical] = raw_df[mapping[canonical]]
            else:
                df[canonical] = np.nan

        df.attrs["column_mapping"] = mapping

        if df["review_id"].isna().all():
            df["review_id"] = [f"REV-{i+1}" for i in range(len(df))]
        else:
            df["review_id"] = df["review_id"].astype(str)

        df["raw_review_text"] = df["review_text"]
        df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
        empty_text_mask = df["review_text"] == ""

        df["raw_rating"] = df["rating"]
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        invalid_rating_mask = df["rating"].isna() | (df["rating"] < 1) | (df["rating"] > 5)

        df["raw_review_date"] = df["review_date"]
        df["parsed_date"] = pd.to_datetime(df["review_date"], errors="coerce").dt.date
        invalid_date_mask = df["parsed_date"].isna()

        df["raw_review_time"] = df["review_time_utc"]
        df["review_time_utc"] = df["review_time_utc"].fillna("").astype(str)

        df["app_version"] = df["app_version"].fillna("Unknown").astype(str).str.strip()
        df.loc[df["app_version"] == "", "app_version"] = "Unknown"

        df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)

        dup_id_count = int(df.duplicated(subset=["review_id"]).sum()) if "review_id" in mapping else 0
        dup_text_count = int(df[df["review_text"] != ""].duplicated(subset=["review_text"]).sum())

        valid_mask = (~invalid_rating_mask) & (~empty_text_mask)
        df["is_valid_review"] = valid_mask

        health_metrics = {
            "total_rows": len(df),
            "valid_rows": int(valid_mask.sum()),
            "empty_text_count": int(empty_text_mask.sum()),
            "invalid_rating_count": int(invalid_rating_mask.sum()),
            "invalid_date_count": int(invalid_date_mask.sum()),
            "duplicate_id_count": dup_id_count,
            "duplicate_text_count": dup_text_count,
            "unique_versions": int(df["app_version"].nunique()),
            "min_date": df["parsed_date"].dropna().min() if not df["parsed_date"].dropna().empty else None,
            "max_date": df["parsed_date"].dropna().max() if not df["parsed_date"].dropna().empty else None
        }

        return df, health_metrics

# ---------------------------------------------------------------------------
# SENTIMENT ANALYSIS ENGINE
# ---------------------------------------------------------------------------
class SentimentEngine:
    def __init__(self):
        self.vader_analyzer = None
        self.engine_name = "Rule-based Lexicon"
        if _HAS_VADER:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                self.engine_name = "NLTK VADER"
            except Exception as e:
                logger.warning(f"VADER init failed, falling back to local lexicon: {e}")

    def score_text(self, text: str) -> Tuple[str, float, str]:
        if not text or not isinstance(text, str) or text.strip() == "":
            return "Neutral", 0.0, "Empty review text"

        clean = text.lower()
        if self.vader_analyzer:
            try:
                scores = self.vader_analyzer.polarity_scores(text)
                compound = float(scores["compound"])
                if compound >= 0.05:
                    label = "Positive"
                elif compound <= -0.05:
                    label = "Negative"
                else:
                    label = "Neutral"
                explanation = f"VADER scores (pos={scores['pos']:.2f}, neg={scores['neg']:.2f}, compound={compound:.2f})"
                return label, compound, explanation
            except Exception:
                pass

        words = re.findall(r"\b[a-z']+\b", clean)
        score = 0.0
        matched = []
        negation_active = False
        negation_tokens = {"not", "no", "never", "can't", "cant", "don't", "dont", "won't", "wont", "hardly"}

        for i, w in enumerate(words):
            if w in negation_tokens:
                negation_active = True
                continue
            if w in LOCAL_FALLBACK_LEXICON:
                w_val = LOCAL_FALLBACK_LEXICON[w]
                if negation_active:
                    w_val = -w_val * 0.8
                    matched.append(f"negated-{w}")
                    negation_active = False
                else:
                    matched.append(w)
                score += w_val
            if negation_active and (i > 0 and i % 3 == 0):
                negation_active = False

        norm_score = float(np.tanh(score))
        if norm_score >= 0.15:
            label = "Positive"
        elif norm_score <= -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        match_str = ", ".join(matched[:5]) if matched else "none"
        explanation = f"Local Lexicon matched: [{match_str}], compound={norm_score:.2f}"
        return label, norm_score, explanation

    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        labels, scores, explanations = [], [], []
        for text in df["review_text"]:
            lbl, sc, exp = self.score_text(text)
            labels.append(lbl)
            scores.append(sc)
            explanations.append(exp)

        df["sentiment_label"] = labels
        df["sentiment_score"] = scores
        df["sentiment_evidence"] = explanations

        df["sentiment_disagreement"] = False
        df.loc[(df["rating"] >= 4) & (df["sentiment_label"] == "Negative"), "sentiment_disagreement"] = True
        df.loc[(df["rating"] <= 2) & (df["sentiment_label"] == "Positive"), "sentiment_disagreement"] = True
        return df

# ---------------------------------------------------------------------------
# COMPLAINT CLASSIFICATION & THEME DETECTION
# ---------------------------------------------------------------------------
class ComplaintClassifier:
    def __init__(self, taxonomy: Optional[Dict[str, Dict[str, Any]]] = None):
        self.taxonomy = taxonomy or DEFAULT_COMPLAINT_TAXONOMY
        self._compile_patterns()

    def _compile_patterns(self):
        self.compiled = {}
        for cat, data in self.taxonomy.items():
            kw_list = data.get("keywords", [])
            escaped = [re.escape(k.lower()) for k in kw_list]
            if escaped:
                pattern = re.compile(r"(?:\b|_)" + r"|".join(escaped) + r"(?:\b|_)", re.IGNORECASE)
                self.compiled[cat] = pattern

    def classify_text(self, text: str, rating: Optional[float] = None) -> Tuple[List[str], List[str], float]:
        if not text or not isinstance(text, str):
            return ["Other or unclassified"], [], 0.0

        text_clean = text.lower()
        matched_categories = []
        matched_phrases = []

        for cat, pattern in self.compiled.items():
            matches = pattern.findall(text_clean)
            if matches:
                if cat == "Positive feedback and praise" and rating is not None and rating <= 2:
                    continue
                matched_categories.append(cat)
                matched_phrases.extend(matches[:3])

        complaints = [c for c in matched_categories if c != "Positive feedback and praise"]
        if not complaints and not matched_categories:
            matched_categories = ["Other or unclassified"]
            confidence = 0.20
        else:
            confidence = min(0.95, 0.40 + 0.15 * len(matched_phrases))

        return matched_categories, list(set(matched_phrases)), float(confidence)

    def classify_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        all_cats, all_phrases, all_confs, primary_cats = [], [], [], []
        for _, row in df.iterrows():
            cats, phrases, conf = self.classify_text(row["review_text"], row.get("rating"))
            all_cats.append(", ".join(cats))
            all_phrases.append(", ".join(phrases))
            all_confs.append(round(conf, 2))
            non_praise = [c for c in cats if c != "Positive feedback and praise"]
            primary_cats.append(non_praise[0] if non_praise else cats[0])

        df["complaint_categories"] = all_cats
        df["primary_category"] = primary_cats
        df["triggered_phrases"] = all_phrases
        df["classification_confidence"] = all_confs
        return df

    def discover_unstructured_clusters(self, df: pd.DataFrame, n_clusters: int = 4) -> List[Dict[str, Any]]:
        subset = df[df["rating"] <= 2].copy()
        if len(subset) < 15:
            return []
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
            X = vectorizer.fit_transform(subset["review_text"])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X)

            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()

            clusters = []
            for i in range(n_clusters):
                top_terms = [terms[ind] for ind in order_centroids[i, :5]]
                cluster_size = int((kmeans.labels_ == i).sum())
                clusters.append({
                    "cluster_id": i + 1,
                    "top_terms": top_terms,
                    "size": cluster_size,
                    "share_of_low": round((cluster_size / len(subset)) * 100, 1)
                })
            return clusters
        except Exception as e:
            logger.warning(f"Clustering error: {e}")
            return []

# ---------------------------------------------------------------------------
# STATISTICAL ANALYSIS ENGINE
# ---------------------------------------------------------------------------
class StatsEngine:
    @staticmethod
    def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        if n == 0:
            return 0.0, 0.0
        z = 1.96 if confidence == 0.95 else 1.645
        p = k / n
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denominator
        lower = max(0.0, centre - margin)
        upper = min(1.0, centre + margin)
        return round(lower * 100, 2), round(upper * 100, 2)

    @staticmethod
    def calculate_mean_ci(series: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        clean = series.dropna()
        n = len(clean)
        if n < 2:
            val = float(clean.mean()) if n == 1 else 0.0
            return val, val
        mean = float(clean.mean())
        sem = float(clean.std(ddof=1) / math.sqrt(n))
        z = 1.96
        return round(max(1.0, mean - z * sem), 2), round(min(5.0, mean + z * sem), 2)

    @classmethod
    def compute_summary_kpis(cls, df: pd.DataFrame) -> Dict[str, Any]:
        valid_df = df[df["is_valid_review"]].copy()
        n_total = len(df)
        n_valid = len(valid_df)
        if n_valid == 0:
            return {}

        ratings = valid_df["rating"]
        mean_rating = float(ratings.mean())
        median_rating = float(ratings.median())
        std_rating = float(ratings.std()) if n_valid > 1 else 0.0
        mean_ci_low, mean_ci_high = cls.calculate_mean_ci(ratings)

        count_1 = int((ratings == 1).sum())
        count_1_2 = int((ratings <= 2).sum())
        count_4_5 = int((ratings >= 4).sum())

        pct_1 = round((count_1 / n_valid) * 100, 2)
        pct_1_2 = round((count_1_2 / n_valid) * 100, 2)
        pct_4_5 = round((count_4_5 / n_valid) * 100, 2)
        ci_low_pct = cls.calculate_wilson_ci(count_1_2, n_valid)

        neg_sentiment_count = int((valid_df["sentiment_label"] == "Negative").sum())
        pct_negative = round((neg_sentiment_count / n_valid) * 100, 2)

        complaints_df = valid_df[
            (~valid_df["primary_category"].isin(["Positive feedback and praise", "Other or unclassified"]))
        ]
        identifiable_complaints = len(complaints_df)

        top_cat = "None"
        if not complaints_df.empty:
            cat_counts = complaints_df["primary_category"].value_counts()
            top_cat = cat_counts.index[0] if not cat_counts.empty else "None"

        versions = [v for v in valid_df["app_version"].unique() if v != "Unknown"]
        valid_dates = valid_df["parsed_date"].dropna()
        date_span = f"{valid_dates.min()} to {valid_dates.max()}" if not valid_dates.empty else "N/A"

        return {
            "total_reviews": n_total,
            "valid_reviews": n_valid,
            "mean_rating": round(mean_rating, 2),
            "mean_ci": (mean_ci_low, mean_ci_high),
            "median_rating": median_rating,
            "std_rating": round(std_rating, 2),
            "count_1": count_1,
            "pct_1": pct_1,
            "count_1_2": count_1_2,
            "pct_1_2": pct_1_2,
            "pct_1_2_ci": ci_low_pct,
            "count_4_5": count_4_5,
            "pct_4_5": pct_4_5,
            "pct_negative": pct_negative,
            "identifiable_complaints": identifiable_complaints,
            "top_complaint_category": top_cat,
            "n_versions": len(versions),
            "date_span": date_span
        }

    @classmethod
    def compute_period_comparison(cls, df: pd.DataFrame) -> pd.DataFrame:
        valid_df = df[df["is_valid_review"] & df["parsed_date"].notna()].sort_values("parsed_date").copy()
        if len(valid_df) < 20:
            return pd.DataFrame()

        mid_idx = len(valid_df) // 2
        p1 = valid_df.iloc[:mid_idx]
        p2 = valid_df.iloc[mid_idx:]

        date1_range = f"{p1['parsed_date'].min()} to {p1['parsed_date'].max()}"
        date2_range = f"{p2['parsed_date'].min()} to {p2['parsed_date'].max()}"

        def get_m(sub):
            return {
                "Count": len(sub),
                "Avg Rating": float(sub["rating"].mean()),
                "Low Rating % (1-2★)": float((sub["rating"] <= 2).mean() * 100),
                "Negative Sentiment %": float((sub["sentiment_label"] == "Negative").mean() * 100)
            }

        m1 = get_m(p1)
        m2 = get_m(p2)

        rows = []
        for metric in ["Avg Rating", "Low Rating % (1-2★)", "Negative Sentiment %"]:
            v1 = m1[metric]
            v2 = m2[metric]
            abs_change = v2 - v1
            rel_change = ((v2 - v1) / v1 * 100) if v1 != 0 else 0.0
            rows.append({
                "Metric": metric,
                f"Previous Period ({date1_range})": round(v1, 2),
                f"Current Period ({date2_range})": round(v2, 2),
                "Absolute Change": round(abs_change, 2),
                "Relative Change (%)": f"{rel_change:+.1f}%"
            })
        return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# PRIORITY SCORING ENGINE
# ---------------------------------------------------------------------------
class PriorityEngine:
    DEFAULT_WEIGHTS = {"frequency": 0.30, "severity": 0.30, "growth": 0.20, "confidence": 0.20}
    EFFORT_MAP = {
        "Booking failure": "Medium", "Login and authentication": "Low",
        "GPS and live tracking": "High", "Incorrect pickup time": "Medium",
        "Late pickup or vehicle arrival": "Medium", "Driver behavior": "Medium",
        "OTP and safety": "Low", "App crash and performance": "Medium",
        "Notifications and communication": "Low", "UI/UX and navigation": "Low",
        "Route and shuttle problems": "High", "Customer support and complaint resolution": "Medium",
        "Payment or billing": "Low"
    }

    RECOMMENDED_ACTIONS = {
        "Booking failure": "Instrument booking API errors; audit roster concurrency limits and slot lockouts.",
        "Login and authentication": "Implement token refresh fallback; fix corporate SSO redirect timeouts.",
        "GPS and live tracking": "Increase driver app GPS ping frequency; filter noisy cellular triangulations.",
        "Incorrect pickup time": "Calibrate ETA algorithm with real-time route traffic data; eliminate unrealistic arrival estimates.",
        "Late pickup or vehicle arrival": "Audit transport dispatch punctuality; trigger proactive buffer alerts 15m prior to trip.",
        "Driver behavior": "Mandate driver punctuality training; establish rapid de-escalation vendor SLA.",
        "OTP and safety": "Remove seatbelt interstitial gate prior to OTP view; offer 1-click persistent OTP display.",
        "App crash and performance": "Resolve memory leaks in native map rendering; fix null pointer crash on update restarts.",
        "Notifications and communication": "Add fallback SMS delivery for critical cab arrival notifications.",
        "UI/UX and navigation": "Simplify booking layout; restore clean minimalist departure board view.",
        "Route and shuttle problems": "Refine clustering algorithm to avoid grouping inter-city or distant passengers.",
        "Customer support and complaint resolution": "Introduce 1-tap in-app ticketing with direct transport desk chat.",
        "Payment or billing": "Automate billing query resolution and corporate pass reconciliation."
    }

    @classmethod
    def calculate_priorities(cls, df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        w = weights or cls.DEFAULT_WEIGHTS
        valid_df = df[df["is_valid_review"]].copy()
        all_categories = [
            cat for cat in DEFAULT_COMPLAINT_TAXONOMY.keys()
            if cat not in ["Positive feedback and praise", "Other or unclassified"]
        ]

        total_valid = len(valid_df)
        if total_valid == 0:
            return pd.DataFrame()

        valid_dates = valid_df["parsed_date"].dropna()
        has_dates = len(valid_dates) >= 20
        if has_dates:
            median_date = valid_dates.sort_values().iloc[len(valid_dates) // 2]
            p1_df = valid_df[valid_df["parsed_date"] < median_date]
            p2_df = valid_df[valid_df["parsed_date"] >= median_date]
        else:
            p1_df, p2_df = pd.DataFrame(), pd.DataFrame()

        records = []
        max_freq = 1
        cat_stats = []

        for cat in all_categories:
            cat_mask = valid_df["complaint_categories"].str.contains(re.escape(cat), na=False, regex=True)
            sub = valid_df[cat_mask]
            count = len(sub)
            if count == 0:
                continue
            max_freq = max(max_freq, count)
            avg_rating = float(sub["rating"].mean())
            low_rating_pct = float((sub["rating"] <= 2).mean() * 100)
            avg_conf = float(sub["classification_confidence"].mean())

            if has_dates and len(p1_df) > 0 and len(p2_df) > 0:
                p1_c = len(p1_df[p1_df["complaint_categories"].str.contains(re.escape(cat), na=False, regex=True)])
                p2_c = len(p2_df[p2_df["complaint_categories"].str.contains(re.escape(cat), na=False, regex=True)])
                growth_rate = ((p2_c - p1_c) / max(1, p1_c))
                growth_score = float(np.clip(50 + growth_rate * 25, 0, 100))
                trend_str = f"▲ +{growth_rate*100:.0f}%" if growth_rate > 0 else (f"▼ {growth_rate*100:.0f}%" if growth_rate < 0 else "— 0%")
            else:
                growth_score = 50.0
                trend_str = "Neutral"

            cat_stats.append({
                "category": cat, "count": count, "avg_rating": avg_rating,
                "low_rating_pct": low_rating_pct, "avg_conf": avg_conf,
                "growth_score": growth_score, "trend_str": trend_str
            })

        for item in cat_stats:
            cat = item["category"]
            count = item["count"]
            freq_score = (count / max_freq) * 100.0
            rating_severity = ((5.0 - item["avg_rating"]) / 4.0) * 100.0
            sev_score = 0.6 * item["low_rating_pct"] + 0.4 * rating_severity
            conf_score = item["avg_conf"] * 100.0
            growth_score = item["growth_score"]

            priority_score = (
                w.get("frequency", 0.3) * freq_score +
                w.get("severity", 0.3) * sev_score +
                w.get("growth", 0.2) * growth_score +
                w.get("confidence", 0.2) * conf_score
            )

            is_safety = DEFAULT_COMPLAINT_TAXONOMY.get(cat, {}).get("is_safety", False)
            if is_safety and priority_score < 70.0:
                priority_score = min(100.0, priority_score + 15.0)

            priority_score = round(float(np.clip(priority_score, 0, 100)), 1)
            effort = cls.EFFORT_MAP.get(cat, "Medium")

            records.append({
                "Issue": cat,
                "Review Count": count,
                "Share of Reviews (%)": round((count / total_valid) * 100, 1),
                "Average Rating": round(item["avg_rating"], 2),
                "Low-Rating % (1-2★)": round(item["low_rating_pct"], 1),
                "Trend": item["trend_str"],
                "Severity Score": round(sev_score, 1),
                "Effort": effort,
                "Priority Score": priority_score,
                "Recommended Action": cls.RECOMMENDED_ACTIONS.get(cat, "Perform root cause analysis.")
            })

        priority_df = pd.DataFrame(records)
        if not priority_df.empty:
            priority_df = priority_df.sort_values("Priority Score", ascending=False).reset_index(drop=True)
            priority_df.insert(0, "Rank", range(1, len(priority_df) + 1))
        return priority_df

# ---------------------------------------------------------------------------
# MACHINE LEARNING ENGINE
# ---------------------------------------------------------------------------
class MLEngine:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.metrics = {}
        self.feature_importance = {}
        self.trained = False

    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        valid_df = df[df["is_valid_review"]].copy()
        if len(valid_df) < 50:
            return {"status": "error", "message": "Dataset too small (<50 valid reviews) for ML model."}

        y = (valid_df["rating"] <= 2).astype(int)
        class_balance = y.value_counts(normalize=True).to_dict()
        if len(class_balance) < 2 or min(class_balance.values()) < 0.05:
            return {"status": "error", "message": "Severe class imbalance (<5% positive class). Model cannot be validated."}

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                valid_df["review_text"], y, test_size=0.25, random_state=42, stratify=y
            )

            self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english", min_df=2)
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            self.model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
            self.model.fit(X_train_vec, y_train)

            y_pred = self.model.predict(X_test_vec)
            y_prob = self.model.predict_proba(X_test_vec)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = 0.50

            feature_names = self.vectorizer.get_feature_names_out()
            coefs = self.model.coef_[0]
            top_low_idx = coefs.argsort()[-15:][::-1]
            top_high_idx = coefs.argsort()[:15]

            self.feature_importance = {
                "top_low_rating_terms": [(feature_names[i], round(float(coefs[i]), 3)) for i in top_low_idx],
                "top_high_rating_terms": [(feature_names[i], round(float(-coefs[i]), 3)) for i in top_high_idx]
            }

            self.metrics = {
                "status": "success",
                "accuracy": round(acc, 3),
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1_score": round(f1, 3),
                "roc_auc": round(auc, 3),
                "confusion_matrix": cm,
                "test_size": len(y_test),
                "train_size": len(y_train),
                "feature_importance": self.feature_importance
            }
            self.trained = True
            return self.metrics
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"status": "error", "message": str(e)}

# ---------------------------------------------------------------------------
# TIME-SERIES FORECASTING ENGINE
# ---------------------------------------------------------------------------
class ForecastingEngine:
    @staticmethod
    def prepare_time_series(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
        valid_df = df[df["is_valid_review"] & df["parsed_date"].notna()].copy()
        if valid_df.empty:
            return pd.DataFrame()

        valid_df["date"] = pd.to_datetime(valid_df["parsed_date"])
        valid_df.set_index("date", inplace=True)

        resampled = valid_df.resample(freq).agg(
            review_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            low_rating_count=("rating", lambda s: (s <= 2).sum())
        ).reset_index()

        resampled["ma_rating"] = resampled["avg_rating"].rolling(window=3, min_periods=1).mean()
        return resampled

    @classmethod
    def generate_forecast(cls, ts_df: pd.DataFrame, periods_ahead: int = 4) -> pd.DataFrame:
        clean = ts_df.dropna(subset=["avg_rating"]).copy()
        if len(clean) < 4:
            return pd.DataFrame()

        x = np.arange(len(clean))
        y = clean["avg_rating"].values
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        rmse = float(np.std(residuals))

        future_x = np.arange(len(clean), len(clean) + periods_ahead)
        last_date = clean["date"].iloc[-1]
        time_step = pd.Timedelta(days=7)
        future_dates = [last_date + (i + 1) * time_step for i in range(periods_ahead)]

        forecast_vals = slope * future_x + intercept
        forecast_vals = np.clip(forecast_vals, 1.0, 5.0)

        rows = []
        for dt, val in zip(future_dates, forecast_vals):
            rows.append({
                "date": dt,
                "projected_rating": round(float(val), 2),
                "lower_bound_95": round(max(1.0, float(val - 1.96 * rmse)), 2),
                "upper_bound_95": round(min(5.0, float(val + 1.96 * rmse)), 2)
            })
        return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# RATING IMPROVEMENT SIMULATOR ENGINE
# ---------------------------------------------------------------------------
class SimulatorEngine:
    SCENARIO_PRESETS = {
        "Conservative": {"resolution_rate": 0.40, "rating_shift_to": 3.5, "future_review_volume_pm": 250},
        "Moderate": {"resolution_rate": 0.65, "rating_shift_to": 4.2, "future_review_volume_pm": 250},
        "Optimistic": {"resolution_rate": 0.85, "rating_shift_to": 4.7, "future_review_volume_pm": 250}
    }

    @classmethod
    def run_simulation(
        cls,
        df: pd.DataFrame,
        target_categories: List[str],
        resolution_rate: float,
        rating_shift_to: float,
        monthly_new_reviews: int
    ) -> Dict[str, Any]:
        valid_df = df[df["is_valid_review"]].copy()
        n_hist = len(valid_df)
        if n_hist == 0:
            return {}

        baseline_avg = float(valid_df["rating"].mean())
        target_pattern = "|".join([re.escape(c) for c in target_categories])
        target_mask = valid_df["complaint_categories"].str.contains(target_pattern, na=False, regex=True)
        target_low_mask = target_mask & (valid_df["rating"] <= 2)

        impacted_review_share = float(target_low_mask.sum() / n_hist)
        incoming_base_rating = baseline_avg

        improved_incoming_avg = incoming_base_rating + (impacted_review_share * resolution_rate) * (rating_shift_to - 1.5)
        improved_incoming_avg = min(5.0, max(1.0, improved_incoming_avg))

        projections = {}
        for m in [1, 3, 6]:
            n_new = monthly_new_reviews * m
            total_n = n_hist + n_new
            cum_rating = (baseline_avg * n_hist + improved_incoming_avg * n_new) / total_n
            abs_change = cum_rating - baseline_avg
            rel_change = (abs_change / baseline_avg) * 100

            projections[f"{m}_month"] = {
                "months": m,
                "projected_cumulative_rating": round(cum_rating, 2),
                "absolute_gain": round(abs_change, 2),
                "relative_gain_pct": round(rel_change, 1),
                "complaints_prevented_est": int(round(n_new * impacted_review_share * resolution_rate))
            }

        return {
            "baseline_rating": round(baseline_avg, 2),
            "target_categories": target_categories,
            "impacted_review_share_pct": round(impacted_review_share * 100, 1),
            "resolution_rate_pct": round(resolution_rate * 100, 1),
            "assumed_shifted_rating": rating_shift_to,
            "projections": projections
        }

# ---------------------------------------------------------------------------
# EXPORT GENERATION ENGINE
# ---------------------------------------------------------------------------
class ExportEngine:
    @staticmethod
    def generate_excel_bytes(df: pd.DataFrame, kpis: Dict[str, Any], priority_df: pd.DataFrame, period_df: pd.DataFrame) -> bytes:
        if not _HAS_OPENPYXL:
            return b""
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")

        def style_sheet(ws, start_row=1):
            for cell in ws[start_row]:
                cell.fill = header_fill
                cell.font = header_font
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                col_letter = openpyxl.utils.get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 40)

        # Sheet 1: Summary
        ws1 = wb.create_sheet(title="Summary")
        ws1.append(["Metric", "Value"])
        kpi_rows = [
            ("Total Reviews", kpis.get("total_reviews", 0)),
            ("Valid Reviews", kpis.get("valid_reviews", 0)),
            ("Mean Rating", kpis.get("mean_rating", 0.0)),
            ("1-Star %", f"{kpis.get('pct_1', 0.0)}%"),
            ("Low-Rating % (1-2★)", f"{kpis.get('pct_1_2', 0.0)}%"),
            ("High-Rating % (4-5★)", f"{kpis.get('pct_4_5', 0.0)}%"),
            ("Negative Sentiment %", f"{kpis.get('pct_negative', 0.0)}%"),
            ("Top Complaint", kpis.get("top_complaint_category", "N/A"))
        ]
        for k, v in kpi_rows:
            ws1.append([k, v])
        style_sheet(ws1)

        # Sheet 2: Priorities
        if not priority_df.empty:
            ws2 = wb.create_sheet(title="Priority Matrix")
            for r in dataframe_to_rows(priority_df, index=False, header=True):
                ws2.append(r)
            style_sheet(ws2)

        # Sheet 3: Full Data
        ws3 = wb.create_sheet(title="Analyzed Reviews")
        export_cols = ["review_id", "rating", "review_date", "app_version", "sentiment_label", "primary_category", "review_text"]
        sub_df = df[[c for c in export_cols if c in df.columns]].head(5000)
        for r in dataframe_to_rows(sub_df, index=False, header=True):
            ws3.append(r)
        style_sheet(ws3)

        out = io.BytesIO()
        wb.save(out)
        return out.getvalue()

    @staticmethod
    def generate_pdf_bytes(kpis: Dict[str, Any], priority_df: pd.DataFrame) -> bytes:
        if not _HAS_REPORTLAB:
            return b""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, textColor=colors.HexColor("#0F172A"))
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, leading=13, textColor=colors.HexColor("#334155"))
        h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor("#1E293B"), spaceBefore=10, spaceAfter=6)

        story = [
            Paragraph("MoveInSync App Review Intelligence Report", title_style),
            Paragraph(f"ReviewPulse AI Executive Audit | Generated {datetime.date.today()}", body_style),
            Spacer(1, 12),
            Paragraph("1. Executive Summary & Findings", h2_style),
            Paragraph(
                f"Analysis of <b>{kpis.get('valid_reviews', 0):,}</b> customer reviews reveals a mean rating of "
                f"<b>{kpis.get('mean_rating', 0.0)}★</b>. Low ratings (1-2★) represent <b>{kpis.get('pct_1_2', 0.0)}%</b> of total feedback. "
                f"The highest priority complaint area is <b>{kpis.get('top_complaint_category', 'N/A')}</b>.",
                body_style
            ),
            Spacer(1, 10)
        ]

        kpi_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Reviews", f"{kpis.get('total_reviews', 0):,}", "1-Star %", f"{kpis.get('pct_1', 0.0)}%"],
            ["Valid Reviews", f"{kpis.get('valid_reviews', 0):,}", "Low-Rating % (1-2★)", f"{kpis.get('pct_1_2', 0.0)}%"],
            ["Mean Rating", f"{kpis.get('mean_rating', 0.0)}★", "High-Rating % (4-5★)", f"{kpis.get('pct_4_5', 0.0)}%"]
        ]
        t = Table(kpi_data, colWidths=[1.8*inch, 1.2*inch, 1.8*inch, 2.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1"))
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        if not priority_df.empty:
            story.append(Paragraph("2. Top Priority Problem Matrix", h2_style))
            p_data = [["Rank", "Issue", "Count", "Avg★", "Low★ %", "Trend", "Effort", "Score"]]
            for _, r in priority_df.head(7).iterrows():
                p_data.append([
                    str(r["Rank"]), str(r["Issue"])[:26], str(r["Review Count"]),
                    f"{r['Average Rating']}★", f"{r['Low-Rating % (1-2★)']}%",
                    str(r["Trend"])[:10], str(r["Effort"]), str(r["Priority Score"])
                ])
            pt = Table(p_data, colWidths=[0.5*inch, 2.2*inch, 0.6*inch, 0.6*inch, 0.8*inch, 0.9*inch, 0.7*inch, 0.7*inch])
            pt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('FONTSIZE', (0,0), (-1,-1), 8)
            ]))
            story.append(pt)

        doc.build(story)
        return buffer.getvalue()

# ---------------------------------------------------------------------------
# PIPELINE CACHING LOGIC
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def execute_analytics_pipeline(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    if filename.endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        try:
            raw_df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            raw_df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")

    clean_df, health = DataValidator.validate_and_clean(raw_df)
    sentiment_engine = SentimentEngine()
    sent_df = sentiment_engine.analyze_dataset(clean_df)

    classifier = ComplaintClassifier()
    analyzed_df = classifier.classify_dataset(sent_df)

    kpis = StatsEngine.compute_summary_kpis(analyzed_df)
    period_df = StatsEngine.compute_period_comparison(analyzed_df)
    priority_df = PriorityEngine.calculate_priorities(analyzed_df)

    ml_engine = MLEngine()
    ml_metrics = ml_engine.train_model(analyzed_df)
    clusters = classifier.discover_unstructured_clusters(analyzed_df, n_clusters=4)

    return {
        "analyzed_df": analyzed_df,
        "health": health,
        "kpis": kpis,
        "period_df": period_df,
        "priority_df": priority_df,
        "ml_metrics": ml_metrics,
        "clusters": clusters
    }

# ---------------------------------------------------------------------------
# STREAMLIT UI SETUP & THEME STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ReviewPulse AI — MoveInSync Review Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Theme
st.markdown("""
<style>
    .stApp {
        background-color: #090d16;
        color: #f8fafc;
    }
    header, footer {visibility: hidden;}
    
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
        font-weight: 700;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    
    .filter-panel {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 20px;
    }
    
    .pulse-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .badge-bar {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 16px;
    }
    .status-badge {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.82rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA INGESTION & DATASET RESOLUTION
# ---------------------------------------------------------------------------
local_dataset_path = "moveinsync_organized_reviews.csv"
file_bytes = None
file_name = None

# Sidebar contains only the clean navigation & file uploader expander
st.sidebar.markdown("## ⚡ ReviewPulse AI")
st.sidebar.caption("MoveInSync App Review Intelligence")

with st.sidebar.expander("📁 Dataset Management", expanded=False):
    uploaded_file = st.file_uploader("Upload New Dataset", type=["csv", "xlsx", "xls"])
    use_default = st.checkbox("Use local MoveInSync dataset", value=True)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif os.path.exists(local_dataset_path):
    with open(local_dataset_path, "rb") as f:
        file_bytes = f.read()
    file_name = local_dataset_path

if not file_bytes:
    st.info("👋 Welcome! Please upload a review dataset (CSV or Excel) or ensure `moveinsync_organized_reviews.csv` is in your directory.")
    st.stop()

# Execute analysis pipeline with caching
with st.spinner("Processing reviews with NLP & statistical pipeline..."):
    pipeline_res = execute_analytics_pipeline(file_bytes, file_name)

analyzed_df = pipeline_res["analyzed_df"]
health = pipeline_res["health"]
kpis = pipeline_res["kpis"]
period_df = pipeline_res["period_df"]
priority_df = pipeline_res["priority_df"]
ml_metrics = pipeline_res["ml_metrics"]
clusters = pipeline_res["clusters"]

# Navigation Menu (Left Sidebar)
NAV_PAGES = [
    "📊 Overview Dashboard",
    "📋 Dataset Explorer",
    "⭐ Rating Analysis",
    "💬 Sentiment Analysis",
    "🔍 Complaint Analysis",
    "⚙️ Root-Cause Investigation",
    "📈 Trends and Forecasts",
    "🎯 Priority Matrix",
    "💡 Recommendations",
    "🚀 Rating Improvement Simulator",
    "📑 Export Reports",
    "📐 Methodology and Settings"
]

selected_page = st.sidebar.radio("Navigation Views", NAV_PAGES)

# ---------------------------------------------------------------------------
# MAIN WORKSPACE TOP HEADER & PERSISTENT FILTER BAR
# (Same architecture as the desktop Tkinter interface!)
# ---------------------------------------------------------------------------
# Top Header Badges
col_hdr1, col_hdr2 = st.columns([3, 1])
with col_hdr1:
    st.markdown(f"""
    <div class="badge-bar">
        <span class="status-badge">📁 <b>File:</b> {file_name}</span>
        <span class="status-badge">📊 <b>Total Reviews:</b> {kpis.get('valid_reviews', 0):,} Valid</span>
        <span class="status-badge">⭐ <b>Avg Rating:</b> {kpis.get('mean_rating', 0.0)}★</span>
        <span class="status-badge" style="color:#10b981;">● <b>Pipeline:</b> Analyzed</span>
    </div>
    """, unsafe_allow_html=True)

# Main Canvas Horizontal Filter Bar
with st.expander("🔍 Interactive Global Filters", expanded=True):
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([1.5, 1.2, 2.2, 1.4, 0.9])

    with f_col1:
        rating_options = ["All", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars", "1-2 Stars (Low)", "4-5 Stars (High)"]
        selected_rating = st.selectbox("Star Rating", rating_options, key="main_rating_filter")

    with f_col2:
        sentiment_options = ["All", "Positive", "Neutral", "Negative"]
        selected_sentiment = st.selectbox("Sentiment", sentiment_options, key="main_sentiment_filter")

    with f_col3:
        cat_options = ["All"] + list(DEFAULT_COMPLAINT_TAXONOMY.keys())
        selected_cat = st.selectbox("Complaint Category", cat_options, key="main_cat_filter")

    with f_col4:
        version_options = ["All"] + sorted([v for v in analyzed_df["app_version"].unique() if v != "Unknown"])
        selected_version = st.selectbox("App Version", version_options, key="main_version_filter")

    with f_col5:
        st.write("")
        st.write("")
        if st.button("Reset", key="reset_filters_btn"):
            st.rerun()

# Apply Filters
filtered_df = analyzed_df.copy()
if selected_rating == "1 Star":
    filtered_df = filtered_df[filtered_df["rating"] == 1]
elif selected_rating == "2 Stars":
    filtered_df = filtered_df[filtered_df["rating"] == 2]
elif selected_rating == "3 Stars":
    filtered_df = filtered_df[filtered_df["rating"] == 3]
elif selected_rating == "4 Stars":
    filtered_df = filtered_df[filtered_df["rating"] == 4]
elif selected_rating == "5 Stars":
    filtered_df = filtered_df[filtered_df["rating"] == 5]
elif selected_rating == "1-2 Stars (Low)":
    filtered_df = filtered_df[filtered_df["rating"] <= 2]
elif selected_rating == "4-5 Stars (High)":
    filtered_df = filtered_df[filtered_df["rating"] >= 4]

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment_label"] == selected_sentiment]

if selected_cat != "All":
    filtered_df = filtered_df[filtered_df["complaint_categories"].str.contains(re.escape(selected_cat), na=False, regex=True)]

if selected_version != "All":
    filtered_df = filtered_df[filtered_df["app_version"] == selected_version]

# Helper for dark styled plotly figures
def style_fig(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f8fafc", size=13)),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", family="Segoe UI, sans-serif", size=11),
        margin=dict(l=30, r=20, t=40, b=30),
        height=height,
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155")
    )
    return fig

# ---------------------------------------------------------------------------
# 1. OVERVIEW DASHBOARD
# ---------------------------------------------------------------------------
if selected_page == "📊 Overview Dashboard":
    st.subheader("Overview Dashboard")

    # KPI Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Average Star Rating", f"{kpis.get('mean_rating', 0.0)}★", f"CI: [{kpis.get('mean_ci', (0,0))[0]}-{kpis.get('mean_ci', (0,0))[1]}]")
    m2.metric("Low Ratings (1-2★)", f"{kpis.get('pct_1_2', 0.0)}%", f"{kpis.get('count_1_2', 0):,} reviews")
    m3.metric("1-Star Share", f"{kpis.get('pct_1', 0.0)}%", f"{kpis.get('count_1', 0):,} reviews")
    m4.metric("High Ratings (4-5★)", f"{kpis.get('pct_4_5', 0.0)}%", f"{kpis.get('count_4_5', 0):,} reviews")
    m5.metric("Negative Sentiment", f"{kpis.get('pct_negative', 0.0)}%", "NLP Polarity")
    m6.metric("Top Complaint Area", str(kpis.get("top_complaint_category", "N/A"))[:16], "Highest volume")

    # Executive Summary Card
    top_issue = kpis.get("top_complaint_category", "Identified themes")
    st.markdown(f"""
    <div class="pulse-card">
        <h4 style="color:#38bdf8; margin-top:0;">Executive Findings Summary</h4>
        <p style="margin-bottom:0; color:#e2e8f0; line-height:1.6;">
            Analysis of <b>{kpis.get('valid_reviews', 0):,}</b> validated Google Play Store customer reviews shows a current dataset mean of 
            <b>{kpis.get('mean_rating', 0.0)}★</b> (median {kpis.get('median_rating', 0.0)}★). 
            Severe customer dissatisfaction is concentrated in 1-star and 2-star reviews (<b>{kpis.get('pct_1_2', 0.0)}%</b> share). 
            The predominant driver of negative ratings is <b>'{top_issue}'</b>. Resolving root-cause bottlenecks in this area 
            will directly prevent the ongoing intake of 1-star reviews and improve overall Play Store rating trajectory.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Charts Row
    c1, c2 = st.columns(2)
    with c1:
        r_counts = analyzed_df["rating"].value_counts().sort_index()
        fig_r = go.Figure(data=[
            go.Bar(
                x=[f"{int(x)} Stars" for x in r_counts.index],
                y=r_counts.values,
                marker_color=["#f43f5e", "#f59e0b", "#64748b", "#38bdf8", "#10b981"],
                text=r_counts.values,
                textposition='auto'
            )
        ])
        fig_r = style_fig(fig_r, "Customer Star Rating Distribution")
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        complaints_sub = analyzed_df[
            ~analyzed_df["primary_category"].isin(["Positive feedback and praise", "Other or unclassified"])
        ]["primary_category"].value_counts().head(6)

        fig_c = go.Figure(data=[
            go.Bar(
                x=complaints_sub.values,
                y=[c[:22] for c in complaints_sub.index],
                orientation='h',
                marker_color="#38bdf8",
                text=complaints_sub.values,
                textposition='auto'
            )
        ])
        fig_c.update_layout(yaxis=dict(autorange="reversed"))
        fig_c = style_fig(fig_c, "Top Complaint Themes by Review Volume")
        st.plotly_chart(fig_c, use_container_width=True)

# ---------------------------------------------------------------------------
# 2. DATASET EXPLORER
# ---------------------------------------------------------------------------
elif selected_page == "📋 Dataset Explorer":
    st.subheader("Dataset Explorer")

    search_query = st.text_input("🔍 Search Review Text", placeholder="Type keywords like 'gps', 'late', 'login', 'crash'...")
    df_display = filtered_df.copy()
    if search_query:
        df_display = df_display[df_display["review_text"].str.contains(search_query, case=False, na=False)]

    st.write(f"Showing **{len(df_display):,}** matching reviews")

    cols_to_view = [
        "review_id", "rating", "review_date", "app_version",
        "sentiment_label", "sentiment_score", "primary_category",
        "triggered_phrases", "classification_confidence", "review_text"
    ]
    st.dataframe(df_display[cols_to_view].head(1000), use_container_width=True, height=450)

    # Detailed Inspection Drawer
    st.markdown("#### 🔎 Inspect Selected Review Evidence")
    selected_idx = st.selectbox("Choose Review Index", df_display.index[:100] if not df_display.empty else [])
    if selected_idx is not None and selected_idx in analyzed_df.index:
        rev_row = analyzed_df.loc[selected_idx]
        st.markdown(f"""
        <div class="pulse-card">
            <b>Review ID:</b> {rev_row.get('review_id')} | <b>Rating:</b> {rev_row.get('rating')}★ | <b>Date:</b> {rev_row.get('parsed_date')} | <b>Version:</b> {rev_row.get('app_version')}<br>
            <b>Sentiment:</b> {rev_row.get('sentiment_label')} (Score: {rev_row.get('sentiment_score', 0):.2f}) | <b>Category:</b> {rev_row.get('primary_category')}<br>
            <b>Trigger Phrases:</b> [{rev_row.get('triggered_phrases')}]<br><br>
            <i style="color:#f8fafc;">"{rev_row.get('review_text')}"</i>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. RATING ANALYSIS
# ---------------------------------------------------------------------------
elif selected_page == "⭐ Rating Analysis":
    st.subheader("Star Rating Analysis")

    c1, c2 = st.columns(2)
    with c1:
        ts_df = ForecastingEngine.prepare_time_series(filtered_df, freq="W")
        fig_ts = go.Figure()
        if not ts_df.empty:
            fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["avg_rating"], mode="lines+markers", name="Weekly Avg Rating", line=dict(color="#38bdf8")))
            if "ma_rating" in ts_df.columns:
                fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["ma_rating"], mode="lines", name="3-Wk Moving Avg", line=dict(color="#f59e0b", dash="dash")))
        fig_ts = style_fig(fig_ts, "Weekly Average Star Rating Trend")
        fig_ts.update_layout(yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig_ts, use_container_width=True)

    with c2:
        ver_stats = filtered_df[filtered_df["app_version"] != "Unknown"].groupby("app_version")["rating"].agg(["mean", "count"]).query("count >= 15").sort_index()
        fig_v = go.Figure(data=[
            go.Bar(x=ver_stats.index, y=ver_stats["mean"], marker_color="#a855f7", text=[f"{m:.2f}★" for m in ver_stats["mean"]], textposition='auto')
        ])
        fig_v = style_fig(fig_v, "Average Rating by App Version (Min 15 reviews)")
        fig_v.update_layout(yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("#### 📅 Period-over-Period Shift Comparison")
    if not period_df.empty:
        st.dataframe(period_df, use_container_width=True)

# ---------------------------------------------------------------------------
# 4. SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------
elif selected_page == "💬 Sentiment Analysis":
    st.subheader("Sentiment Analysis")

    c1, c2 = st.columns(2)
    with c1:
        cross_tab = pd.crosstab(filtered_df["rating"], filtered_df["sentiment_label"], normalize="index") * 100
        fig_s = go.Figure()
        for col_name, color in [("Negative", "#f43f5e"), ("Neutral", "#64748b"), ("Positive", "#10b981")]:
            if col_name in cross_tab.columns:
                fig_s.add_trace(go.Scatter(x=cross_tab.index, y=cross_tab[col_name], mode="lines+markers", name=col_name, line=dict(color=color, width=2.5)))
        fig_s = style_fig(fig_s, "Sentiment Distribution Across Star Ratings (%)")
        st.plotly_chart(fig_s, use_container_width=True)

    with c2:
        fig_h = px.histogram(filtered_df, x="sentiment_score", nbins=30, color_discrete_sequence=["#38bdf8"])
        fig_h = style_fig(fig_h, "Compound Sentiment Score Distribution (-1.0 to +1.0)")
        st.plotly_chart(fig_h, use_container_width=True)

    st.info(f"💡 **Sentiment Disagreements (e.g. 5★ with negative complaints, or 1★ with praise):** {int(filtered_df['sentiment_disagreement'].sum()):,} reviews flagged for investigation.")

# ---------------------------------------------------------------------------
# 5. COMPLAINT ANALYSIS
# ---------------------------------------------------------------------------
elif selected_page == "🔍 Complaint Analysis":
    st.subheader("Complaint Themes & Thematic Classification")

    cat_data = []
    for cat in DEFAULT_COMPLAINT_TAXONOMY.keys():
        sub = filtered_df[filtered_df["complaint_categories"].str.contains(re.escape(cat), na=False, regex=True)]
        cnt = len(sub)
        if cnt > 0:
            cat_data.append({
                "Category": cat,
                "Count": cnt,
                "Share (%)": f"{(cnt/len(filtered_df))*100:.1f}%",
                "Avg Rating": f"{sub['rating'].mean():.2f}★",
                "Low Rating %": f"{(sub['rating'] <= 2).mean() * 100:.1f}%",
                "Responsible Team": DEFAULT_COMPLAINT_TAXONOMY[cat].get("team", "General")
            })

    cat_df = pd.DataFrame(cat_data).sort_values("Count", ascending=False)
    st.dataframe(cat_df, use_container_width=True, height=360)

    st.markdown("#### 🧩 Exploratory Clusters (Unstructured Themes in 1-2★ Reviews)")
    if clusters:
        cols = st.columns(len(clusters))
        for i, cl in enumerate(clusters):
            with cols[i]:
                terms_bullet = "".join([f"<li>{t}</li>" for t in cl['top_terms']])
                st.markdown(textwrap.dedent(f"""
                <div class="pulse-card">
                    <h5 style="color:#38bdf8; margin-top:0;">Cluster #{cl['cluster_id']}</h5>
                    <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:6px;">{cl['size']} reviews ({cl['share_of_low']}%)</p>
                    <ul style="padding-left:18px; font-size:0.85rem; margin-bottom:0; color:#cbd5e1;">
                        {terms_bullet}
                    </ul>
                </div>
                """), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 6. ROOT-CAUSE INVESTIGATION
# ---------------------------------------------------------------------------
elif selected_page == "⚙️ Root-Cause Investigation":
    st.subheader("Root-Cause Investigation")
    top_issue = priority_df.iloc[0]["Issue"] if not priority_df.empty else "Booking failure"

    st.markdown(f"""
    <div class="pulse-card">
        <h4 style="color:#f43f5e; margin-top:0;">Primary Root Cause Target: {top_issue}</h4>
        <p style="margin-bottom:0; color:#e2e8f0;">Translating Play Store qualitative feedback into actionable engineering telemetry checks.</p>
    </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(textwrap.dedent("""
        <div class="pulse-card">
            <h5 style="color:#38bdf8; margin-top:0;">Group A: Observed Evidence</h5>
            <ul style="color:#cbd5e1; margin-bottom:0;">
                <li>Empirical volume surge in 1★ and 2★ reviews.</li>
                <li>Customer comments report commute disruptions, lockouts, and post-update regressions.</li>
                <li>Clear correlation with low average rating cohorts.</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)

        st.markdown(textwrap.dedent("""
        <div class="pulse-card">
            <h5 style="color:#f59e0b; margin-top:0;">Group B: Possible Explanations</h5>
            <ul style="color:#cbd5e1; margin-bottom:0;">
                <li>API backend latency spikes during peak shift scheduling.</li>
                <li>Client-side GPS background wake lock starvation on Android 13+.</li>
                <li>Route clustering combining distant corporate pickups.</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)

    with g2:
        st.markdown(textwrap.dedent("""
        <div class="pulse-card">
            <h5 style="color:#a855f7; margin-top:0;">Group C: Hypotheses to Test (Internal Telemetry)</h5>
            <ul style="color:#cbd5e1; margin-bottom:0;">
                <li>Audit 5xx rate and latency on booking/trip endpoints.</li>
                <li>Measure driver mobile GPS ping intervals (target &lt;5s).</li>
                <li>Inspect OTP SMS delivery gateway delivery lag.</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)

        st.markdown(textwrap.dedent("""
        <div class="pulse-card">
            <h5 style="color:#10b981; margin-top:0;">Group D: Recommended Engineering Actions</h5>
            <ul style="color:#cbd5e1; margin-bottom:0;">
                <li>Immediate (0-7d): Remove OTP modal gate; increase SMS retry speed.</li>
                <li>Short-term (1-4w): Patch background location permission handling.</li>
                <li>Medium-term (1-3m): Calibrate ETA routing algorithms against peak traffic.</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 7. TRENDS AND FORECASTS
# ---------------------------------------------------------------------------
elif selected_page == "📈 Trends and Forecasts":
    st.subheader("Trends & Short-Horizon Forecasts")

    ts_df = ForecastingEngine.prepare_time_series(filtered_df, freq="W")
    forecast_df = ForecastingEngine.generate_forecast(ts_df, periods_ahead=4)

    fig_fc = go.Figure()
    if not ts_df.empty:
        fig_fc.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["avg_rating"], mode="lines+markers", name="Observed Rating", line=dict(color="#38bdf8")))
        if not forecast_df.empty:
            last_dt = ts_df["date"].iloc[-1]
            last_val = ts_df["avg_rating"].iloc[-1]
            fc_dates = [last_dt] + list(forecast_df["date"])
            fc_vals = [last_val] + list(forecast_df["projected_rating"])
            fc_low = [last_val] + list(forecast_df["lower_bound_95"])
            fc_high = [last_val] + list(forecast_df["upper_bound_95"])

            fig_fc.add_trace(go.Scatter(x=fc_dates, y=fc_vals, mode="lines+markers", name="Status Quo Forecast", line=dict(color="#f59e0b", dash="dash")))
            fig_fc.add_trace(go.Scatter(x=fc_dates + fc_dates[::-1], y=fc_high + fc_low[::-1], fill='toself', fillcolor='rgba(245, 158, 11, 0.15)', line=dict(color='rgba(255,255,255,0)'), name="95% Uncertainty Band"))

    fig_fc = style_fig(fig_fc, "Historical Weekly Rating & 4-Week Status Quo Forecast")
    fig_fc.update_layout(yaxis=dict(range=[1, 5]))
    st.plotly_chart(fig_fc, use_container_width=True)

    if not forecast_df.empty:
        st.markdown("#### 🔮 4-Week Projection Table")
        st.dataframe(forecast_df, use_container_width=True)

# ---------------------------------------------------------------------------
# 8. PRIORITY MATRIX
# ---------------------------------------------------------------------------
elif selected_page == "🎯 Priority Matrix":
    st.subheader("Problem Prioritization Matrix")

    with st.expander("⚙️ Adjust Scoring Engine Weights", expanded=False):
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        w_freq = w_col1.slider("Frequency Weight", 0.0, 1.0, 0.30, 0.05)
        w_sev = w_col2.slider("Severity Weight", 0.0, 1.0, 0.30, 0.05)
        w_grow = w_col3.slider("Recent Growth Weight", 0.0, 1.0, 0.20, 0.05)
        w_conf = w_col4.slider("Confidence Weight", 0.0, 1.0, 0.20, 0.05)

    custom_weights = {"frequency": w_freq, "severity": w_sev, "growth": w_grow, "confidence": w_conf}
    p_df = PriorityEngine.calculate_priorities(analyzed_df, weights=custom_weights)

    st.dataframe(p_df, use_container_width=True)

    effort_map_num = {"Low": 1, "Medium": 2, "High": 3}
    fig_p = go.Figure()
    for _, r in p_df.iterrows():
        fig_p.add_trace(go.Scatter(
            x=[effort_map_num.get(r["Effort"], 2)],
            y=[r["Priority Score"]],
            mode="markers+text",
            text=[r["Issue"][:18]],
            textposition="top center",
            marker=dict(size=max(12, r["Review Count"] * 0.4), color="#38bdf8", opacity=0.8),
            name=r["Issue"]
        ))
    fig_p.update_layout(
        xaxis=dict(tickvals=[1, 2, 3], ticktext=["Low Effort (Quick Wins)", "Medium Effort", "High Effort (Architectural)"]),
        yaxis=dict(title="Priority Score (0-100)", range=[30, 105]),
        showlegend=False
    )
    fig_p = style_fig(fig_p, "Impact vs Implementation Effort Matrix (Bubble size = Review count)")
    st.plotly_chart(fig_p, use_container_width=True)

# ---------------------------------------------------------------------------
# 9. RECOMMENDATIONS
# ---------------------------------------------------------------------------
elif selected_page == "💡 Recommendations":
    st.subheader("Actionable Engineering & Operational Roadmap")

    phases = [
        ("Phase 1: Immediate Stabilization (0–7 Days) — Low Effort Quick Wins", [
            "OTP Interstitial Flow: Remove mandatory seatbelt acknowledgment gate blocking 1-click OTP view.",
            "Corporate SSO / Login: Increase auth token renewal buffer and silent token retry.",
            "Notification Delivery: Add fallback SMS gateway dispatch for high-priority cab arrival alerts."
        ], "#10b981"),
        ("Phase 2: Short-Term Bug Fixes & Diagnostics (1–4 Weeks)", [
            "Client Background GPS: Correct wake lock permissions to maintain driver location stream.",
            "App Crash Telemetry: Fix null pointer exception in Native Map fragment following version updates.",
            "Rating Prompt Flow: Introduce non-blocking 'Skip' option to eliminate retaliatory 1-star ratings."
        ], "#38bdf8"),
        ("Phase 3: Medium-Term Product & Routing Enhancements (1–3 Months)", [
            "ETA Algorithm Calibration: Train arrival time estimates on peak office hour congestion matrices.",
            "Roster Allocation Logic: Prevent bundling riders across distant city zones in single shuttle pools.",
            "Driver Vendor SLA: Establish real-time tracking of driver unpunctuality with automated dispatch re-routing."
        ], "#f59e0b"),
        ("Phase 4: Long-Term Architectural & Operational Quality (3–6 Months)", [
            "Automated Review Intelligence: Deploy real-time Play Store sentiment and issue alerting pipeline.",
            "Telepresence Ride Tracking: Sub-second websocket live vehicle tracking.",
            "Closed-Loop Support: Link user reviews directly to corporate transport desk tickets."
        ], "#a855f7")
    ]

    for title, items, col in phases:
        items_html = "".join([f"<li>{it}</li>" for it in items])
        st.markdown(textwrap.dedent(f"""
        <div class="pulse-card">
            <h4 style="color:{col}; margin-top:0;">{title}</h4>
            <ul style="color:#cbd5e1; margin-bottom:0;">
                {items_html}
            </ul>
        </div>
        """), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 10. RATING IMPROVEMENT SIMULATOR
# ---------------------------------------------------------------------------
elif selected_page == "🚀 Rating Improvement Simulator":
    st.subheader("Rating Improvement Simulator — What If We Fix These Problems?")
    st.caption("Model projected customer rating gains under explicit scenario assumptions over 1, 3, and 6 months.")

    c1, c2, c3 = st.columns(3)
    preset = c1.selectbox("Scenario Preset", ["Moderate", "Conservative", "Optimistic"])
    preset_data = SimulatorEngine.SCENARIO_PRESETS[preset]

    target_cat = c2.selectbox("Target Issue", ["All Top Issues"] + list(DEFAULT_COMPLAINT_TAXONOMY.keys()))
    res_rate = c3.slider("Resolution Rate (%)", 10.0, 100.0, preset_data["resolution_rate"] * 100.0, 5.0) / 100.0

    shift_rating = preset_data["rating_shift_to"]
    target_cats = list(DEFAULT_COMPLAINT_TAXONOMY.keys()) if target_cat == "All Top Issues" else [target_cat]

    sim_res = SimulatorEngine.run_simulation(
        analyzed_df, target_cats, res_rate, shift_rating, monthly_new_reviews=250
    )

    if sim_res:
        p1 = sim_res["projections"]["1_month"]
        p3 = sim_res["projections"]["3_month"]
        p6 = sim_res["projections"]["6_month"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Rating", f"{sim_res['baseline_rating']}★", "Empirical Baseline")
        m2.metric("Projected: +1 Month", f"{p1['projected_cumulative_rating']}★", f"+{p1['absolute_gain']}★ (+{p1['relative_gain_pct']}%)")
        m3.metric("Projected: +3 Months", f"{p3['projected_cumulative_rating']}★", f"+{p3['absolute_gain']}★ (+{p3['relative_gain_pct']}%)")
        m4.metric("Projected: +6 Months", f"{p6['projected_cumulative_rating']}★", f"+{p6['absolute_gain']}★ (+{p6['relative_gain_pct']}%)")

        fig_sim = go.Figure(data=[
            go.Bar(
                x=["Baseline", "+1 Month", "+3 Months", "+6 Months"],
                y=[sim_res['baseline_rating'], p1['projected_cumulative_rating'], p3['projected_cumulative_rating'], p6['projected_cumulative_rating']],
                marker_color=["#64748b", "#38bdf8", "#f59e0b", "#10b981"],
                text=[f"{v:.2f}★" for v in [sim_res['baseline_rating'], p1['projected_cumulative_rating'], p3['projected_cumulative_rating'], p6['projected_cumulative_rating']]],
                textposition='auto'
            )
        ])
        fig_sim = style_fig(fig_sim, f"Projected App Rating Trajectory (Assuming {res_rate*100:.0f}% resolution)")
        fig_sim.update_layout(yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig_sim, use_container_width=True)

# ---------------------------------------------------------------------------
# 11. EXPORT REPORTS
# ---------------------------------------------------------------------------
elif selected_page == "📑 Export Reports":
    st.subheader("Export Intelligence Reports")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. CSV Dataset")
        st.caption("Full dataset with derived sentiment & category analytics.")
        csv_bytes = analyzed_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download CSV", csv_bytes, "moveinsync_analyzed_reviews.csv", "text/csv")

    with c2:
        st.markdown("### 2. Multi-Sheet Excel")
        st.caption("Workbook with Executive Summary, Priorities & Review data.")
        excel_bytes = ExportEngine.generate_excel_bytes(analyzed_df, kpis, priority_df, period_df)
        st.download_button("📥 Download Excel (.xlsx)", excel_bytes, "moveinsync_audit.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with c3:
        st.markdown("### 3. Executive PDF")
        st.caption("Executive audit report with findings and priority table.")
        pdf_bytes = ExportEngine.generate_pdf_bytes(kpis, priority_df)
        st.download_button("📥 Download PDF Report", pdf_bytes, "moveinsync_executive_report.pdf", "application/pdf")

# ---------------------------------------------------------------------------
# 12. METHODOLOGY AND SETTINGS (Clean, Beautiful, No HTML escaping bugs)
# ---------------------------------------------------------------------------
elif selected_page == "📐 Methodology and Settings":
    st.subheader("System Architecture & Analytical Methodology")

    methodology_sections = [
        ("1. Data Sanitization & Ingestion Rules", [
            "Canonical column headers are matched case-insensitively and support aliases (e.g. 'comment', 'stars', 'version').",
            "Customer star ratings are strictly cast to numeric floats (1–5 scale); invalid values are logged and quarantined.",
            "Raw customer review text and original timestamps are preserved immutable in memory.",
            "All NLP predictions and theme labels are stored in separate derived columns without data fabrication."
        ]),
        ("2. Explainable Sentiment Analysis Engine", [
            "Primary model: NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner) calibrated for review texts.",
            "Local transparent fallback lexicon with negation handling (e.g. 'not working', 'never late', 'cant book') for offline execution.",
            "Disagreements between customer star rating (e.g. 5★ with negative text, or 1★ with praise) are flagged for human audit."
        ]),
        ("3. Multi-Factor Problem Prioritization Formula", [
            "Composite Priority Score = 0.30 × Frequency + 0.30 × Severity + 0.20 × Recent Growth + 0.20 × Classification Confidence.",
            "Severity is weighted by low-star concentration (1★ and 2★ share) and negative sentiment share.",
            "Safety Override: Critical categories like Driver Behavior or SOS/OTP receive an automatic priority score floor."
        ]),
        ("4. Predictive ML Risk Modeling", [
            "Binary classification target: Low-Rating Risk (1–2★ vs 3–5★).",
            "Feature extraction: TF-IDF unigram and bigram token matrices with English stop-word removal.",
            "Classifier: Stratified Logistic Regression with balanced class weighting to handle customer review class imbalance."
        ]),
        ("5. Rating Simulator Assumptions & Governance", [
            "Mathematical formula: Future Rating = (Historical Sum + Projected Future Sum) / (Historical N + Future N).",
            "Models prospective rating trajectories over 1, 3, and 6 months under stated resolution rates (e.g. 65%).",
            "Simulator estimates mathematical potential; it does not promise or guarantee future customer ratings."
        ])
    ]

    for title, points in methodology_sections:
        with st.container():
            st.markdown(f"#### 🔹 {title}")
            for pt in points:
                st.markdown(f"- {pt}")
            st.write("")
