import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Importing Data
df_full = pd.read_csv('data.csv')

df = df_full

# Cleaning Data
del df['Unnamed: 0']
del df['track_id']
del df['artists']
del df['album_name']
del df['track_name']
del df['popularity']
del df['explicit']
del df['key']
del df['mode']
del df['time_signature']

df.rename(columns = {'track_genre':'genre'}, inplace = True)

drop_genres = ['acoustic','alternative','afrobeat','alt-rock','ambient','anime','black-metal','bluegrass','blues','brazil','breakbeat','british','cantopop','chicago-house','children','chill','club','comedy','dancehall','death-metal','deep-house','detroit-techno','disney','drum-and-bass','dub','dubstep','edm','electro','emo','folk','forro','french','funk','garage','german','gospel','goth','grindcore','groove','grunge','guitar','happy','hard-rock','hardcore','hardstyle','heavy-metal','honky-tonk','idm','indian','indie-pop','indie','industrial','iranian','j-dance','j-idol','j-pop','j-rock','k-pop','kids','latin','latino','malay','mandopop','metalcore','minimal-techno','mpb','new-age','opera','pagode','party','piano','pop-film','power-pop','progressive-house','psych-rock','punk-rock','punk','reggae','reggaeton','rock-n-roll','rockabilly','romance','sad','salsa','samba','sertanejo','show-tunes','singer-songwriter','ska','sleep','songwriter','soul','spanish','study','swedish','synth-pop','tango','techno','trance','trip-hop','turkish','world-music']
df.drop(df[df['genre'].isin(drop_genres)].index, inplace = True)

# Training the model on the full dataset
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train_full = df_train_full.reset_index(drop=True)

y_train_full = df_train_full.genre.values
del df_train_full['genre']

dicts_train_full = df_train_full.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train_full = dv.fit_transform(dicts_train_full)

le = LabelEncoder()
y_train_full = le.fit_transform(y_train_full)

rf = RandomForestClassifier(n_estimators=125,
                            max_depth=15,
                            min_samples_leaf=3,
                            random_state=1)

rf.fit(X_train_full, y_train_full)

# Export the model

with open('model1.bin', 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is exported to model.bin')