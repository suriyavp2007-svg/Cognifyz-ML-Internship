# Cognifyz Technologies – Machine Learning Internship

This repository contains the tasks completed as part of the **Machine Learning Internship** at **Cognifyz Technologies**. The tasks are based on a real-world restaurant dataset (Zomato-style data) and focus on data preprocessing, exploratory analysis, predictive modeling, and recommendation systems.

## 📂 Dataset

The dataset (`Dataset_.csv`) contains **9,551 restaurant records** with the following key attributes:
- Restaurant details (Name, City, Address, Locality)
- Cuisines served
- Average Cost for two, Currency, Price range
- Table booking / Online delivery availability
- Aggregate rating, Rating text, Votes

---

## ✅ Task: Restaurant Recommendation System

**Objective:** Build a content-based recommendation system that suggests restaurants to a user based on their stated preferences (cuisine, price range, etc.).

### Approach

1. **Data Preprocessing**
   - Handled missing values (9 rows with missing `Cuisines`, filled as `"Unknown"`)
   - Removed 93 duplicate restaurant entries
   - Encoded binary categorical columns (`Has Table booking`, `Has Online delivery`, etc.) as 0/1
   - Encoded `Cuisines` using **TF-IDF vectorization** (handles restaurants with multiple cuisines better than one-hot encoding)

2. **Recommendation Criteria**
   Each restaurant is represented as a feature vector combining:
   - Cuisine similarity (TF-IDF)
   - Normalized price range
   - Normalized aggregate rating

3. **Content-Based Filtering**
   - User preferences (cuisine, price range, city, minimum rating) are converted into a query vector using the same encoding as the restaurant catalogue
   - **Cosine similarity** ranks restaurants against the query
   - Optional hard filters (city, price range, minimum rating) narrow the candidate pool before ranking

4. **Testing & Evaluation**
   - Tested with 4 sample user profiles (e.g., "North Indian in New Delhi", "cheap Chinese with rating ≥ 4.0", "high-end Japanese/Sushi")
   - Evaluated using a **precision@10** relevance check — measuring how many of the top-10 recommendations actually matched the requested cuisine
   - Achieved **1.00 precision@10** across all tested cuisines, with average cosine similarity scores above 0.99

### Files
| File | Description |
|------|-------------|
| `recommender.py` | Full pipeline: preprocessing, TF-IDF encoding, content-based filtering model, test cases, and evaluation |
| `output_log.txt` | Console output from running the script, showing preprocessing steps, sample recommendations, and evaluation scores |
| `Dataset_.csv` | Source dataset used for the task |

### How to Run
```bash
pip install pandas numpy scikit-learn scipy
python recommender.py
```

### Sample Output
```
User: wants Chinese food, low budget (price range 1), rating >= 4.0

     Restaurant Name                   City       Cuisines  Price range  Aggregate rating  Votes  similarity_score
    Tsing Tsao South             Des Moines        Chinese             1               4.1    218          0.999983
  Ting's Red Lantern Cedar Rapids/Iowa City        Chinese             1               4.2    347          0.999934
```

### Key Learnings
- TF-IDF is an effective way to encode multi-label categorical text data (like comma-separated cuisines) for similarity-based recommendations
- Combining text-based similarity with normalized numeric features (price, rating) allows a recommender to balance "what cuisine" with "what quality/budget"
- Content-based filtering doesn't require user history — it works purely from item attributes and stated preferences, making it well-suited to cold-start scenarios

---

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- scikit-learn (TF-IDF, cosine similarity, MinMaxScaler)
- SciPy (sparse matrix operations)

## 🙏 Acknowledgment
Thanks to **Cognifyz Technologies** for the opportunity to work on this hands-on Machine Learning internship task.

Connect with me on [LinkedIn](#) to see more of my internship journey!
