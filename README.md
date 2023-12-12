# Music Genre Classification Model

This project is part of the ML Zoomcamp course by DataTalksClub.
You can access the course [here](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem Description

This project is to train a model to predict which genre a song is part of from its Spotify audio features.

## Dataset

Dataset used - https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset.

This dataset contains information on 114,000 songs including their audio features from Spotify and 114 genres.

## Training

I first completed the EDA and cleaned up the data. I think started training the models. I used Logistic Regression, Decision Tree and Random Forest. I tuned all models and spend time selecting the correct one. In the end, Random Forest had the best outcome and was chosen as the final model.

## Export

I built out a `train.py` script to train the model and save it to a `model.bin` file.

I then created a `predict.py` script to load the model and process the function with song info using a POST request to determine the correct genre.

I created a `test.py` script to be able to test giving the predict app song data and then processing it and giving back a genre prediction.

## Setup locally

1. Install Pipenv:
````
pip install pipenv
````


2. Install the requirements:
    - flask
    - numpy
    - scikit-learn==1.3.0
    - gunicorn
````
pipenv install gunicorn flask numpy scikit-learn==1.3.0
````


3. Clone the repo to the Pipenv directory


4. Run the `train.py` script to train the model and export the model file:
````
python train.py
````


5. Run the `predict.py` script to run the webservice using gunicorn and is then ready for input:
````
python predict.py
````


6. You can now run the `test.py` script in another terminal window to pass the song data to the predict app and get back the genre prediction:
````
python test.py
````

7. Edit the `genre_1` json in the `test.py` file to be able to test out other variations and get different results.


## Setup using Docker


1. Build a docker image:
````
docker build -t NAME_OF_PROJECT .
````

2. Run the Docker image:
````
docker run -it --rm -p 9696:9696 NAME_OF_PROJECT
````

3. You can now run the `test.py` script in another terminal window to pass the song data to the predict app and get back the genre prediction:
````
python test.py
````

4. Edit the `genre_1` json in the `test.py` file to be able to test out other variations and get different results.


## Deploying to the cloud

I used https://render.com to deploy my Docker image.

I created a webservice that can be accessed through https://genrepred.onrender.com.

**Note, it might take a few seconds to start up if it has been idle.**


You can use this service by editing the url variable in the `test.py` script and changing it to ``https://genrepred.onrender.com/predict``.