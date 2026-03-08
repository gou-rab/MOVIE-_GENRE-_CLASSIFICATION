from flask import Flask, request, jsonify, send_from_directory
import pickle, re, os

app      = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

model = pickle.load(open(os.path.join(BASE_DIR, 'GenreModel.pkl'),    'rb'))
tfidf = pickle.load(open(os.path.join(BASE_DIR, 'tfidf.pkl'),         'rb'))
le    = pickle.load(open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb'))
print(f"✅ Model loaded | Genres: {list(le.classes_)}")


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        plot = str(data.get('plot', '')).strip()
        if len(plot) < 10:
            return jsonify({'success': False, 'error': 'Plot summary too short.'}), 400

        vec  = tfidf.transform([clean_text(plot)])
        pred = model.predict(vec)[0]

        if hasattr(model, 'decision_function'):
            scores = model.decision_function(vec)[0]
            scores = (scores - scores.min()) / (scores.ptp() + 1e-9)
        else:
            scores = model.predict_proba(vec)[0]

        top5 = sorted(zip(le.classes_, scores.tolist()), key=lambda x: -x[1])[:5]

        return jsonify({
            'success':    True,
            'genre':      le.inverse_transform([pred])[0],
            'confidence': round(float(max(scores)) * 100, 1),
            'top5':       [{'genre': g, 'score': round(s * 100, 1)} for g, s in top5]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    return jsonify({'model': type(model).__name__,
                    'genres': list(le.classes_),
                    'features': tfidf.max_features})


if __name__ == '__main__':
    print("\n🎬 Genre Predictor — http://127.0.0.1:5000\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
