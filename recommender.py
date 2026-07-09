"""
Cognifyz ML Internship - Task: Restaurant Recommendation System
=================================================================
Content-based filtering recommender using restaurant Cuisines,
Price range, and Aggregate rating as similarity criteria.

Steps:
1. Preprocess dataset (handle missing values, encode categorical variables)
2. Define recommendation criteria (cuisine, price range, rating)
3. Build content-based filtering model (TF-IDF on cuisines + cosine similarity)
4. Test with sample user preferences and evaluate quality
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ---------------------------------------------------------------
# STEP 1: LOAD & PREPROCESS
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 1: PREPROCESSING")
print("=" * 70)

df = pd.read_csv("/mnt/user-data/uploads/Dataset_.csv")
print(f"Original shape: {df.shape}")

# --- Handle missing values ---
# Only 'Cuisines' has missing values (9 rows) -> fill with 'Unknown'
missing_before = df.isnull().sum()
print("\nMissing values before cleaning:")
print(missing_before[missing_before > 0])

df["Cuisines"] = df["Cuisines"].fillna("Unknown")
df["Aggregate rating"] = df["Aggregate rating"].fillna(0)
df["Votes"] = df["Votes"].fillna(0)

print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")

# --- Remove exact duplicate restaurants (same name + locality) ---
before = len(df)
df = df.drop_duplicates(subset=["Restaurant Name", "Locality Verbose"]).reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows -> {len(df)} rows remain")

# --- Encode categorical variables ---
# Binary Yes/No columns -> 1/0
binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
for col in binary_cols:
    df[col + " (encoded)"] = df[col].map({"Yes": 1, "No": 0})

# Price range is already ordinal (1-4), keep as-is.
# City / Cuisines are high-cardinality categorical -> encode Cuisines via
# TF-IDF (multi-label, since a restaurant can have several cuisines) rather
# than one-hot, which would create 1800+ sparse columns for little benefit.

print("\nSample of cleaned data:")
print(df[["Restaurant Name", "City", "Cuisines", "Price range", "Aggregate rating", "Votes"]].head())


# ---------------------------------------------------------------
# STEP 2: DEFINE RECOMMENDATION CRITERIA
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: RECOMMENDATION CRITERIA")
print("=" * 70)
print("""
A restaurant's "content profile" is built from:
  1. Cuisines        -> TF-IDF vector (captures cuisine similarity, e.g.
                          'North Indian, Chinese' is close to 'Chinese, Thai')
  2. Price range      -> normalized numeric (1=cheap ... 4=expensive)
  3. Aggregate rating  -> normalized numeric (quality signal)

User preferences (query) are expressed the same way:
  - preferred cuisines (free text, e.g. "Italian, Pizza")
  - preferred price range (1-4)
  - City (optional filter)
  - minimum rating (optional filter)
""")


# ---------------------------------------------------------------
# STEP 3: CONTENT-BASED FILTERING MODEL
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 3: BUILDING CONTENT-BASED MODEL")
print("=" * 70)

tfidf = TfidfVectorizer(token_pattern=r"[^,]+")  # split only on commas, keep multi-word cuisines intact
cuisine_matrix = tfidf.fit_transform(df["Cuisines"].str.strip())
print(f"Cuisine TF-IDF matrix shape: {cuisine_matrix.shape}")

scaler = MinMaxScaler()
numeric_features = scaler.fit_transform(df[["Price range", "Aggregate rating"]])

CUISINE_WEIGHT = 1.0
NUMERIC_WEIGHT = 0.3  # smaller weight so price/rating nudge results, not dominate

combined_matrix = hstack([
    cuisine_matrix * CUISINE_WEIGHT,
    csr_matrix(numeric_features) * NUMERIC_WEIGHT
]).tocsr()

print(f"Combined feature matrix shape: {combined_matrix.shape}")


def recommend_restaurants(cuisine_pref=None, price_range=None, city=None,
                           min_rating=0.0, top_n=5):
    """
    Content-based recommender.

    Builds a query vector from user preferences using the SAME TF-IDF
    vocabulary + scaler fit on the restaurant catalogue, then ranks all
    (optionally filtered) restaurants by cosine similarity to that query.
    """
    candidates = df.copy()
    if city:
        candidates = candidates[candidates["City"].str.lower() == city.lower()]
    if min_rating:
        candidates = candidates[candidates["Aggregate rating"] >= min_rating]
    if price_range:
        candidates = candidates[candidates["Price range"] == price_range]

    if candidates.empty:
        return pd.DataFrame(), "No restaurants match the hard filters (city/price/min rating)."

    idxs = candidates.index

    cuisine_query_text = cuisine_pref if cuisine_pref else ""
    query_cuisine_vec = tfidf.transform([cuisine_query_text]) * CUISINE_WEIGHT

    query_price = price_range if price_range else df["Price range"].median()
    query_rating = 4.0  # assume user wants a good restaurant by default
    query_df = pd.DataFrame([[query_price, query_rating]], columns=["Price range", "Aggregate rating"])
    query_numeric = scaler.transform(query_df) * NUMERIC_WEIGHT

    query_vec = hstack([query_cuisine_vec, csr_matrix(query_numeric)]).tocsr()

    sims = cosine_similarity(query_vec, combined_matrix[idxs])[0]
    candidates = candidates.copy()
    candidates["similarity_score"] = sims

    result = candidates.sort_values(
        by=["similarity_score", "Aggregate rating", "Votes"],
        ascending=[False, False, False]
    ).head(top_n)

    return result[["Restaurant Name", "City", "Cuisines", "Price range",
                    "Aggregate rating", "Votes", "similarity_score"]], None


# ---------------------------------------------------------------
# STEP 4: TEST WITH SAMPLE USER PREFERENCES
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: TESTING THE RECOMMENDATION SYSTEM")
print("=" * 70)

test_cases = [
    {"desc": "User 1: wants North Indian food, mid-range price, in New Delhi",
     "params": dict(cuisine_pref="North Indian", price_range=2, city="New Delhi", top_n=5)},

    {"desc": "User 2: wants Italian/Pizza, no city restriction, any price",
     "params": dict(cuisine_pref="Italian, Pizza", top_n=5)},

    {"desc": "User 3: wants Chinese food, low budget (price range 1), rating >= 4.0",
     "params": dict(cuisine_pref="Chinese", price_range=1, min_rating=4.0, top_n=5)},

    {"desc": "User 4: wants Japanese/Sushi, high-end (price range 4)",
     "params": dict(cuisine_pref="Japanese, Sushi", price_range=4, top_n=5)},
]

for case in test_cases:
    print("\n" + "-" * 70)
    print(case["desc"])
    print("Preferences:", case["params"])
    result, err = recommend_restaurants(**case["params"])
    if err:
        print(err)
    else:
        print(result.to_string(index=False))


# ---------------------------------------------------------------
# STEP 5: EVALUATE QUALITY OF RECOMMENDATIONS
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: EVALUATION")
print("=" * 70)

def evaluate(cuisine_pref, top_n=10, **filters):
    """
    Simple precision-style check: what fraction of recommended restaurants'
    cuisine lists actually contain at least one of the requested cuisine
    keywords? This proxies relevance for a content-based system (there are
    no user click/purchase logs to compute true precision/recall against).
    """
    result, err = recommend_restaurants(cuisine_pref=cuisine_pref, top_n=top_n, **filters)
    if err or result.empty:
        return None
    keywords = [k.strip().lower() for k in cuisine_pref.split(",")]
    hits = result["Cuisines"].str.lower().apply(
        lambda c: any(k in c for k in keywords)
    )
    precision = hits.mean()
    avg_sim = result["similarity_score"].mean()
    return precision, avg_sim

print("\nRelevance check (does the recommended cuisine actually match the ask?):")
for cuisine in ["North Indian", "Italian", "Chinese", "Japanese", "Mexican"]:
    precision, avg_sim = evaluate(cuisine, top_n=10)
    print(f"  Query='{cuisine:15s}'  cuisine-match precision@10 = {precision:.2f}   avg similarity = {avg_sim:.3f}")

print("\nDone.")
