# K-Drama-Recommendation

> Latest Data 2023 | Dataset from [Kaggle](https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset)

This Streamlit-based **K-Drama Recommendation System** provides personalized drama suggestions using **content-based filtering**. The model analyzes key drama attributes to recommend titles similar to the ones a user rates highly.

## 🚀 How It Works
1. **User Input** – Enter **three K-Dramas** you’ve watched and rate them.
2. **Feature Matching** – The system calculates similarity using a **weighted feature vector** approach:
   - **Genre (44%)** – Ensures alignment with preferred drama types.
   - **Main Cast (25%)** – Recognizes the impact of favorite actors.
   - **Tags (20%)** – Captures themes, tropes, and unique elements.
   - **Network (1%)** – Considers the broadcasting channel.
   - **Content Rating (10%)** – Ensures age-appropriate recommendations.
3. **Personalized Output** – The app suggests **five K-Dramas** that best match your preferences.

## 📌 Features
- **Personalized K-Drama Recommendations** based on user ratings.
- **Content-Based Filtering** using genre, main cast, tags, network, and content rating.

## 📸 Screenshot
![App Screenshot](screenshot.png)

---
### ⭐ Enjoy discovering your next favorite K-Drama!

