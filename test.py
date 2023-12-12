import requests

url = 'http://localhost:9696/predict'

genre_1 = {'acousticness': 0.000708,
         'danceability': 0.652,
         'duration_ms': 251520,
         'energy': 0.798,
         'instrumentalness': 0.0327,
         'key': 5,
         'liveness': 0.112,
         'loudness': -7.224,
         'mode': 1,
         'speechiness': 0.0325,
         'tempo': 124.01,
         'time_signature': 4,
         'valence': 0.336}

genre_2 = {'acousticness': 0.0036,
           'danceability': 0.744,
           'duration_ms': 185918,
           'energy': 0.789,
           'instrumentalness': 0.00144,
           'key': 9,
           'liveness': 0.0947,
           'loudness': -4.876,
           'mode': 0,
           'speechiness': 0.059,
           'tempo': 116.985,
           'time_signature': 4,
           'valence': 0.866}

genre_3 = {'acousticness': 0.558,
           'danceability': 0.543,
           'duration_ms': 223840,
           'energy': 0.485,
           'instrumentalness': 0,
           'key': 8,
           'liveness': 0.12,
           'loudness': -6.85,
           'mode': 1,
           'speechiness': 0.0305,
           'tempo': 136.961,
           'time_signature': 4,
           'valence': 0.371}

for genre_data in [genre_1, genre_2, genre_3]:
    response = requests.post(url, json=genre_data).json()
    print('This track is from the "' +response['genre_pred']+ '" genre.')