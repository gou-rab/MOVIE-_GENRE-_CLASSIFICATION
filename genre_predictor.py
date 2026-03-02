import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re, os, pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Working directory: {BASE_DIR}")

def load_txt(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                rows.append({'id': parts[0].strip(), 'title': parts[1].strip(),
                             'genre': parts[2].strip(), 'description': parts[3].strip()})
    return pd.DataFrame(rows)

TRAIN_PATH = os.path.join(BASE_DIR, 'train_data.txt')
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"train_data.txt not found in {BASE_DIR}\n"
        "Download from: kaggle.com/datasets/hijest/genre-classification-dataset-imdb"
    )

df = load_txt(TRAIN_PATH)
print(f"\n📄 Training rows: {len(df):,}  |  Genres: {df['genre'].nunique()}")
print(df['genre'].value_counts().to_string())

STOPWORDS = set(("a about above after again against all am an and any are as at be "
    "because been before being below between both but by can could did do does doing "
    "down during each few for from further get got had has have having he her here "
    "hers herself him himself his how i if in into is it its itself let me more most "
    "my myself no nor not of off on once only or other our ours ourselves out over own "
    "same she should so some such than that the their theirs them themselves then "
    "there these they this those through to too under until up very was we were what "
    "when where which while who whom why will with would you your yours yourself").split())

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', str(text).lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

print("\n⏳ Cleaning descriptions...")
df['clean'] = df['description'].apply(clean_text)
print("✅ Done. Sample:", df['clean'].iloc[0][:100], '...')

plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0a0f1a')

ax1 = axes[0]
ax1.set_facecolor('#0f1825')
gc = df['genre'].value_counts()
cols = plt.cm.plasma(np.linspace(0.2, 0.9, len(gc)))
ax1.barh(gc.index[::-1], gc.values[::-1], color=cols, height=0.7)
ax1.set_title('Movies per Genre', color='white', fontsize=13)
ax1.set_xlabel('Count', color='#888')
ax1.tick_params(colors='white', labelsize=9)
for s in ax1.spines.values(): s.set_edgecolor('#1e2d3d')

ax2 = axes[1]
ax2.set_facecolor('#0f1825')
df['wcount'] = df['clean'].str.split().str.len()
ax2.hist(df['wcount'], bins=50, color='#7c3aed', alpha=0.85, edgecolor='none')
ax2.axvline(df['wcount'].mean(), color='#f59e0b', linestyle='--', lw=2,
            label=f"Mean: {df['wcount'].mean():.0f} words")
ax2.set_title('Description Length (words)', color='white', fontsize=13)
ax2.set_xlabel('Word Count', color='#888')
ax2.legend(fontsize=10)
ax2.tick_params(colors='#888')
for s in ax2.spines.values(): s.set_edgecolor('#1e2d3d')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'eda_plots.png'), dpi=120,
            bbox_inches='tight', facecolor='#0a0f1a')
plt.close()
print("📊 EDA plot saved.")

y  = le.fit_transform(df['genre'])
print(f"\n🏷️  Genres: {list(le.classes_)}")

print("⏳ Fitting TF-IDF (30k features, 1-2 grams)...")
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                         sublinear_tf=True, min_df=2)
X = tfidf.fit_transform(df['clean'])
print(f"✅ Matrix: {X.shape}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)
MODELS = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000,
                             class_weight='balanced', solver='liblinear'),
    'Naive Bayes':         MultinomialNB(alpha=0.1),
    'Linear SVM':          LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
}

results = {}
print("\n⏳ Training classifiers...")
for name, clf in MODELS.items():
    print(f"   → {name} ...", end=' ', flush=True)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    results[name] = {'model': clf, 'acc': acc, 'preds': preds}
    print(f"{acc*100:.2f}%")

best_name = max(results, key=lambda k: results[k]['acc'])
best      = results[best_name]
print(f"\n🏆 Best: {best_name}  →  {best['acc']*100:.2f}%")
print(classification_report(y_te, best['preds'], target_names=le.classes_))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0a0f1a')

ax1 = axes[0]
ax1.set_facecolor('#0f1825')
names = list(results.keys())
accs  = [results[n]['acc']*100 for n in names]
bar_c = ['#f59e0b' if n == best_name else '#7c3aed' for n in names]
bars  = ax1.bar(names, accs, color=bar_c, width=0.5, edgecolor='none')
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', color='white', fontsize=11)
ax1.set_ylim(0, 100)
ax1.set_title('Model Accuracy Comparison', color='white', fontsize=13)
ax1.set_ylabel('Accuracy (%)', color='#888')
ax1.tick_params(colors='white')
ax1.set_xticklabels(names, rotation=10, ha='right')
for s in ax1.spines.values(): s.set_edgecolor('#1e2d3d')

ax2 = axes[1]
ax2.set_facecolor('#0f1825')
cm = confusion_matrix(y_te, best['preds'])
cm_n = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_n, annot=False, cmap='Blues', ax=ax2,
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.3)
ax2.set_title(f'Confusion Matrix — {best_name}', color='white', fontsize=13)
ax2.set_xlabel('Predicted', color='#888')
ax2.set_ylabel('Actual', color='#888')
ax2.tick_params(colors='white', labelsize=7)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'model_evaluation.png'), dpi=120,
            bbox_inches='tight', facecolor='#0a0f1a')
plt.close()
print("📈 Evaluation plots saved.")

pickle.dump(best['model'], open(os.path.join(BASE_DIR, 'GenreModel.pkl'),    'wb'))
pickle.dump(tfidf,         open(os.path.join(BASE_DIR, 'tfidf.pkl'),         'wb'))
pickle.dump(le,            open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'wb'))
print(f"\n✅ GenreModel.pkl   ({best_name})")
print(f"✅ tfidf.pkl")
print(f"✅ label_encoder.pkl")

def predict_genre(plot_text):
    m   = pickle.load(open(os.path.join(BASE_DIR, 'GenreModel.pkl'),    'rb'))
    tf  = pickle.load(open(os.path.join(BASE_DIR, 'tfidf.pkl'),         'rb'))
    enc = pickle.load(open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb'))
    vec  = tf.transform([clean_text(plot_text)])
    pred = m.predict(vec)[0]
    if hasattr(m, 'decision_function'):
        scores = m.decision_function(vec)[0]
        scores = (scores - scores.min()) / (scores.ptp() + 1e-9)
        top3   = sorted(zip(enc.classes_, scores), key=lambda x: -x[1])[:3]
    else:
        proba = m.predict_proba(vec)[0]
        top3  = sorted(zip(enc.classes_, proba), key=lambda x: -x[1])[:3]
    return enc.inverse_transform([pred])[0], top3

print("\n" + "="*60)
print("🎬  DEMO PREDICTIONS")
print("="*60)
demos = [
    ("A detective investigates a brutal serial killer leaving cryptic clues in New York City.",  "Thriller/Crime"),
    ("Two people fall in love aboard a luxury ship sailing across the Atlantic Ocean.",           "Romance"),
    ("A team of soldiers battles their way through enemy lines on a secret rescue mission.",     "Action"),
    ("A comedian struggles with fame while trying to reconnect with his estranged family.",      "Comedy"),
    ("Ghosts haunt an old mansion where a family moves in and strange things begin happening.",  "Horror"),
]
for plot, expected in demos:
    genre, top3 = predict_genre(plot)
    top_str = ', '.join(f'{g}({v*100:.0f}%)' for g, v in top3)
    print(f"\n  Expected : {expected}")
    print(f"  Predicted: {genre}")
    print(f"  Top 3    : {top_str}")
