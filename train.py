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

df.rename(columns = {'track_genre':'genre'}, inplace = True)

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

rf = RandomForestClassifier(n_estimators=175,
                            max_depth=15,
                            min_samples_leaf=5,
                            random_state=1)

rf.fit(X_train_full, y_train_full)

# Export the model

with open('model1.bin', 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is exported to model.bin')