import pickle

from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

genres = [
    'acoustic',
    'afrobeat',
    'alt-rock',
    'alternative',
    'ambient',
    'anime',
    'black-metal',
    'bluegrass',
    'blues',
    'brazil',
    'breakbeat',
    'british',
    'cantopop',
    'chicago-house',
    'children',
    'chill',
    'classical',
    'club',
    'comedy',
    'country',
    'dance',
    'dancehall',
    'death-metal',
    'deep-house',
    'detroit-techno',
    'disco',
    'disney',
    'drum-and-bass',
    'dub',
    'dubstep',
    'edm',
    'electro',
    'electronic',
    'emo',
    'folk',
    'forro',
    'french',
    'funk',
    'garage',
    'german',
    'gospel',
    'goth',
    'grindcore',
    'groove',
    'grunge',
    'guitar',
    'happy',
    'hard-rock',
    'hardcore',
    'hardstyle',
    'heavy-metal',
    'hip-hop',
    'honky-tonk',
    'house',
    'idm',
    'indian',
    'indie-pop',
    'indie',
    'industrial',
    'iranian',
    'j-dance',
    'j-idol',
    'j-pop',
    'j-rock',
    'jazz',
    'k-pop',
    'kids',
    'latin',
    'latino',
    'malay',
    'mandopop',
    'metal',
    'metalcore',
    'minimal-techno',
    'mpb',
    'new-age',
    'opera',
    'pagode',
    'party',
    'piano',
    'pop-film',
    'pop',
    'power-pop',
    'progressive-house',
    'psych-rock',
    'punk-rock',
    'punk',
    'r-n-b',
    'reggae',
    'reggaeton',
    'rock-n-roll',
    'rock',
    'rockabilly',
    'romance',
    'sad',
    'salsa',
    'samba',
    'sertanejo',
    'show-tunes',
    'singer-songwriter',
    'ska',
    'sleep',
    'songwriter',
    'soul',
    'spanish',
    'study',
    'swedish',
    'synth-pop',
    'tango',
    'techno',
    'trance',
    'trip-hop',
    'turkish',
    'world-music']

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

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)