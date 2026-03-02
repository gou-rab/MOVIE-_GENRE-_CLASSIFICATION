# 🎬 CineGenre — Movie Genre Predictor

An NLP Machine Learning web app that predicts the **genre of a movie** from its plot summary. Built with **TF-IDF vectorization** and compares **3 classifiers** (Logistic Regression, Naive Bayes, Linear SVM), served via **Flask** with a cinematic dark-gold themed frontend.

---

## 📁 Project Structure

```
movie-genre-predictor/
│
├── train_data.txt              # IMDb dataset (download from Kaggle)
├── test_data.txt               # IMDb test set (optional)
├── genre_predictor.py          # NLP pipeline, training, model saving
├── app.py                      # Flask API backend
├── index.html                  # Cinematic web UI
├── requirements.txt            # Python dependencies
│
│   ── Generated after training ──
├── GenreModel.pkl              # Best trained classifier
├── tfidf.pkl                   # Fitted TF-IDF vectorizer
├── label_encoder.pkl           # Genre label encoder
├── eda_plots.png               # Genre distribution + description length
└── model_evaluation.png        # Accuracy comparison + confusion matrix
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Genre Classification Dataset IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) |
| File | `train_data.txt` |
| Format | `ID ::: Title ::: Genre ::: Description` |
| Genres | Action, Adult, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Game-Show, History, Horror, Music, Musical, Mystery, News, Reality-TV, Romance, Sci-Fi, Short, Sport, Thriller, War, Western |

---

## 🤖 NLP & ML Pipeline

### Text Preprocessing
1. Lowercase all text
2. Remove HTML tags
3. Remove punctuation and digits
4. Remove English stopwords
5. Filter tokens shorter than 3 characters

### Vectorization
- **TF-IDF** with 30,000 features
- **Unigrams + Bigrams** (`ngram_range=(1,2)`)
- Sublinear TF scaling (`sublinear_tf=True`)
- Minimum document frequency of 2

### Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | `C=1.0`, `class_weight='balanced'`, `solver='liblinear'` |
| Naive Bayes | `MultinomialNB(alpha=0.1)` |
| Linear SVM | `LinearSVC(C=1.0)`, `class_weight='balanced'` |

The best model is auto-selected by accuracy and saved as `GenreModel.pkl`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| NLP | TF-IDF (scikit-learn), custom stopword removal |
| ML Models | LogisticRegression, MultinomialNB, LinearSVC |
| Backend | Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Persistence | pickle |
| Visualization | matplotlib, seaborn |

---

## ⚙️ Setup & Run

### 1. Download the dataset
Go to [kaggle.com/datasets/hijest/genre-classification-dataset-imdb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) and download `train_data.txt`. Place it in your project folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python genre_predictor.py
```

Output:
```
📄 Training rows: 54,214  |  Genres: 27
⏳ Cleaning descriptions...
⏳ Fitting TF-IDF (30k features, 1-2 grams)...
⏳ Training classifiers...
   → Logistic Regression ... 58.3%
   → Naive Bayes         ... 54.1%
   → Linear SVM          ... 59.7%
🏆 Best: Linear SVM → 59.7%
✅ GenreModel.pkl
✅ tfidf.pkl
✅ label_encoder.pkl
```

### 4. Start Flask server
```bash
python app.py
```

### 5. Open `index.html` in your browser
The status badge turns green when Flask is connected.

---

## 🌐 API Reference

### `POST /predict`
**Request:**
```json
{ "plot": "A detective investigates mysterious murders in the city..." }
```

**Response:**
```json
{
  "success": true,
  "genre": "Thriller",
  "confidence": 72.4,
  "top5": [
    { "genre": "Thriller", "score": 72.4 },
    { "genre": "Crime",    "score": 18.1 },
    { "genre": "Mystery",  "score": 6.3  },
    { "genre": "Drama",    "score": 2.1  },
    { "genre": "Action",   "score": 1.1  }
  ]
}
```

### `GET /model-info`
```json
{
  "model": "LinearSVC",
  "genres": ["Action", "Comedy", "Drama", ...],
  "features": 30000
}
```

---

## 📈 Why These Models?

- **TF-IDF** is the gold standard for text classification on tabular-style NLP tasks — fast, interpretable, and effective
- **Logistic Regression** works well with high-dimensional sparse data (TF-IDF matrices)
- **Naive Bayes** is fast and surprisingly strong for text, especially with skewed class distributions
- **Linear SVM** often achieves the best accuracy on multi-class text classification tasks

---

## ⚠️ Limitations

- Genre classification is inherently ambiguous — many movies span multiple genres
- Model predicts a single genre; multi-label classification would be more realistic
- Performance varies by genre frequency (common genres like Drama/Comedy predict better)
- Very short plot summaries give less reliable predictions

---

## 📄 License
MIT License
