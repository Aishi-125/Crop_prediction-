# Importing essential libraries and modules
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

# Import NLTK for text processing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

# Load trained models and dictionaries
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic

# Load chatbot model
import json
import random
from tensorflow import keras

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('final.json').read())
model = keras.models.load_model('chatbot.h5')
lemmatizer = WordNetLemmatizer()

def preprocess_input(user_input):
    user_input_words = word_tokenize(user_input)
    user_input_words = [lemmatizer.lemmatize(word.lower()) for word in user_input_words]
    input_bag = [1 if word in user_input_words else 0 for word in words]
    return input_bag

def get_chatbot_response(user_input):
    input_bag = preprocess_input(user_input)
    prediction = model.predict(np.array([input_bag]))
    predicted_class = classes[np.argmax(prediction)]
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            return response

# Function to convert text to speech using gTTS and save to disk
from gtts import gTTS
def text_to_speech(text):
    tts = gTTS(text)
    audio_file = 'audio.wav'
    tts.save(audio_file)
    return audio_file

def chatbot():
    title = 'AgroSoln - Chatbot'
    return render_template('chatbot.html', title=title)

# Flask app initialization
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    title = 'AgroSoln - Home'
    return render_template('index.html', title=title)

# Route for crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'AgroSoln - Crop Recommendation'
    return render_template('crop.html', title=title)

# Route for fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgroSoln - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

# Route for disease prediction input page
@app.route('/disease-predict')
def disease_prediction():
    title = 'AgroSoln - Disease Detection'
    return render_template('disease.html', title=title)

# Route for crop recommendation result page
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'AgroSoln - Crop Recommendation'
    # Process form data
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        # Fetch weather data
        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)

# Route for fertilizer recommendation result page
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgroSoln - Fertilizer Suggestion'
    # Process form data
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])
    df = pd.read_csv('Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]
    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"
    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)

# Route for disease prediction result page
@app.route('/disease-predict', methods=['POST'])
def disease_prediction_result():
    title = 'AgroSoln - Disease Detection'
    # Process form data
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            return render_template('error.html', error=e, title=title)

# Route for chatbot search
@app.route('/', methods=['POST'])
def chatbot_search():
    title = 'AgroSoln - Home'
    user_input = request.form['user_input']
    if user_input:
        chatbot_response = get_chatbot_response(user_input)
        return render_template('index.html', user_input=user_input, chatbot_response=chatbot_response, title=title)
    else:
        return render_template('index.html', title=title)

# Function to fetch weather data
def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

# Run the Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print("An error occurred:", e)
