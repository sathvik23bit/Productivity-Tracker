"""
URL Productivity Classifier
Step 1: Rule-based auto-labeling + TF-IDF + Logistic Regression
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from urllib.parse import urlparse

# ─────────────────────────────────────────
# 1. RULE-BASED AUTO-LABELER
# ─────────────────────────────────────────

PRODUCTIVE_DOMAINS = [
    "github.com", "stackoverflow.com", "docs.python.org", "arxiv.org",
    "medium.com", "kaggle.com", "coursera.org", "udemy.com", "edx.org",
    "leetcode.com", "geeksforgeeks.org", "developer.mozilla.org",
    "docs.google.com", "notion.so", "trello.com", "jira.atlassian.com",
    "linear.app", "figma.com", "overleaf.com", "scholar.google.com",
    "researchgate.net", "huggingface.co", "pytorch.org", "tensorflow.org",
]

UNPRODUCTIVE_DOMAINS = [
    "netflix.com", "instagram.com", "twitter.com",
    "facebook.com", "reddit.com", "tiktok.com", "twitch.tv",
    "9gag.com", "buzzfeed.com", "pinterest.com", "snapchat.com",
    "spotify.com", "primevideo.com", "disneyplus.com", "hulu.com",
]

NEUTRAL_DOMAINS = [
    "google.com", "bing.com", "yahoo.com", "wikipedia.org",
    "amazon.com", "gmail.com", "outlook.com",
]


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url if url.startswith("http") else "https://" + url)
        domain = parsed.netloc.replace("www.", "")
        return domain
    except:
        return url


def auto_label(url: str, title: str = "") -> str:
    """
    Auto-label a URL using rule-based heuristics.
    Returns: 'productive', 'unproductive', or 'neutral'
    """
    domain = extract_domain(url)

    for d in PRODUCTIVE_DOMAINS:
        if d in domain:
            return "productive"

    for d in UNPRODUCTIVE_DOMAINS:
        if d in domain:
            return "unproductive"

    for d in NEUTRAL_DOMAINS:
        if d in domain:
            return "neutral"

    # Keyword hints from URL path and title
    url_lower = (url + " " + title).lower()

    productive_keywords = [
        "tutorial", "learn", "course", "documentation", "api", "docs",
        "research", "paper", "study", "project", "code", "programming",
        "algorithm", "machine-learning", "deep-learning", "data",
        "lecture", "explained", "how to", "introduction", "beginner",
        "python", "javascript", "math", "science", "engineering",
        "for beginners", "crash course", "full course", "bootcamp",
        "lesson", "training", "workshop", "interview prep",
    ]

    unproductive_keywords = [
        "meme", "funny", "entertainment", "game", "shorts",
        "reels", "trending", "compilation", "lofi", "lo-fi",
        "reaction", "prank", "challenge", "highlights", "vlog",
    ]

    prod_score = sum(1 for kw in productive_keywords if kw in url_lower)
    unprod_score = sum(1 for kw in unproductive_keywords if kw in url_lower)

    if prod_score > unprod_score:
        return "productive"
    elif unprod_score > prod_score:
        return "unproductive"
    return "neutral"


# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────

def extract_features_text(url: str, title: str = "") -> str:
    domain = extract_domain(url)
    url_tokens = re.split(r'[/.\-_?=&]', url)
    url_tokens = [t for t in url_tokens if len(t) > 2]
    combined = domain + " " + " ".join(url_tokens) + " " + title
    return combined.lower()


# ─────────────────────────────────────────
# 3. SAMPLE DATASET
# ─────────────────────────────────────────

SAMPLE_DATA = [
    ("https://github.com/pytorch/pytorch", "PyTorch GitHub", "productive"),
    ("https://github.com/tensorflow/tensorflow", "TensorFlow GitHub", "productive"),
    ("https://stackoverflow.com/questions/12345", "How to reverse a list in Python", "productive"),
    ("https://stackoverflow.com/questions/67890", "Fix CORS error in FastAPI", "productive"),
    ("https://arxiv.org/abs/2301.00001", "Attention is All You Need", "productive"),
    ("https://arxiv.org/abs/2302.00002", "BERT Pre-training of Deep Transformers", "productive"),
    ("https://docs.python.org/3/library/re.html", "Python re module docs", "productive"),
    ("https://kaggle.com/competitions", "Kaggle ML competitions", "productive"),
    ("https://coursera.org/learn/machine-learning", "ML Course Andrew Ng", "productive"),
    ("https://leetcode.com/problems/two-sum", "Two Sum - LeetCode", "productive"),
    ("https://huggingface.co/models", "HuggingFace Models", "productive"),
    ("https://notion.so/myworkspace", "Project Notes - Notion", "productive"),
    ("https://medium.com/towards-data-science", "Towards Data Science Article", "productive"),
    ("https://youtube.com/watch?v=abc", "Python Tutorial for Beginners", "productive"),
    ("https://youtube.com/watch?v=def", "Deep Learning Crash Course", "productive"),
    ("https://www.netflix.com/watch/movie", "Breaking Bad Season 1", "unproductive"),
    ("https://www.instagram.com/explore", "Instagram Reels", "unproductive"),
    ("https://reddit.com/r/memes", "Funny memes subreddit", "unproductive"),
    ("https://twitter.com/home", "Twitter home feed", "unproductive"),
    ("https://www.twitch.tv/xqc", "xQc live stream", "unproductive"),
    ("https://tiktok.com/@user/video", "TikTok funny video", "unproductive"),
    ("https://discord.com/channels/gaming", "Gaming Discord server", "unproductive"),
    ("https://www.youtube.com/watch?v=xyz", "Funny cats compilation", "unproductive"),
    ("https://www.youtube.com/shorts/abc", "Trending shorts highlights", "unproductive"),
    ("https://wikipedia.org/wiki/Deep_learning", "Deep learning - Wikipedia", "neutral"),
    ("https://google.com/search?q=python", "python - Google Search", "neutral"),
    ("https://google.com/search?q=weather", "weather - Google Search", "neutral"),
    ("https://wikipedia.org/wiki/Neural_network", "Neural network - Wikipedia", "neutral"),
    ("https://amazon.com/products", "Amazon Shopping", "neutral"),
    ("https://gmail.com/inbox", "Gmail Inbox", "neutral"),
]


def generate_sample_csv(path="sample_browsing_data.csv"):
    rows = []
    for url, title, label in SAMPLE_DATA:
        rows.append({
            "url": url,
            "title": title,
            "time_spent_sec": np.random.randint(30, 600),
            "label": label,
            "auto_label": auto_label(url, title),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"✓ Sample dataset saved to {path} ({len(df)} rows)")
    return df


# ─────────────────────────────────────────
# 4. TRAIN THE MODEL
# ─────────────────────────────────────────

def train(csv_path="sample_browsing_data.csv", model_path="url_classifier.pkl"):
    df = pd.read_csv(csv_path)
    df["final_label"] = df["label"].fillna(df["auto_label"])
    df = df[df["final_label"].notna()].copy()

    df["text"] = df.apply(
        lambda r: extract_features_text(r["url"], str(r.get("title", ""))), axis=1
    )

    X = df["text"].values
    y = df["final_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            C=1.0,
            class_weight="balanced",
            zero_division=0,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n─── Model Evaluation ───")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, model_path)
    print(f"✓ Model saved to {model_path}")
    return pipeline


# ─────────────────────────────────────────
# 5. PREDICT
# ─────────────────────────────────────────

def predict(url: str, title: str = "", model_path="url_classifier.pkl") -> dict:
    pipeline = joblib.load(model_path)
    text = extract_features_text(url, title)
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    scores = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, proba)}
    predicted = classes[np.argmax(proba)]

    return {
        "url": url,
        "title": title,
        "predicted_label": predicted,
        "scores": scores,
        "productive_pct": scores.get("productive", 0),
    }


def predict_batch(csv_path: str, model_path="url_classifier.pkl") -> pd.DataFrame:
    pipeline = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    df["text"] = df.apply(
        lambda r: extract_features_text(r["url"], str(r.get("title", ""))), axis=1
    )
    proba = pipeline.predict_proba(df["text"].values)
    classes = pipeline.classes_
    for i, cls in enumerate(classes):
        df[f"score_{cls}"] = (proba[:, i] * 100).round(1)
    df["predicted_label"] = classes[np.argmax(proba, axis=1)]
    return df


# ─────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=== URL Productivity Classifier ===\n")

    df = generate_sample_csv()
    model = train()

    test_urls = [
        ("https://github.com/tensorflow/tensorflow", "TensorFlow GitHub"),
        ("https://youtube.com/watch?v=abc", "Python Tutorial for Beginners"),
        ("https://www.youtube.com/watch?v=xyz", "Funny cats compilation"),
        ("https://google.com/search?q=weather", "weather - Google Search"),
        ("https://leetcode.com/problems/binary-search", "Binary Search - LeetCode"),
        ("https://www.instagram.com/reels", "Instagram Reels"),
    ]

    print("\n─── Sample Predictions ───")
    for url, title in test_urls:
        result = predict(url, title)
        bar = "█" * int(result["productive_pct"] / 10) + "░" * (10 - int(result["productive_pct"] / 10))
        print(f"\n{title}")
        print(f"  Label : {result['predicted_label'].upper()}")
        print(f"  Prod% : [{bar}] {result['productive_pct']}%")
