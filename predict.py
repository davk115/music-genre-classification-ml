import pickle

from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

genres = [
    'classical',
    'country',
    'dance',
    'disco',
    'electronic',
    'hip-hop',
    'house',
    'jazz',
    'metal',
    'pop',
    'r-n-b',
    'rock']

app = Flask('genre')

@app.route('/predict', methods=['POST'])
def predict():
    genre = request.get_json()
    X = dv.transform([genre])

    y_pred = model.predict(X)

    genre_pred = genres[y_pred[0]]
    result = {
        'genre_pred': str(genre_pred)
    }
    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)