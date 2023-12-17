from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import pickle
import random
import os
from string import punctuation

app = Flask(__name__)

# Load the model
model_path = 'chatbot'
load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
model = tf.keras.models.load_model(model_path, options=load_options)

# Load the label encoder
with open("label_encoder.pickle", "rb") as le_filename:
    le = pickle.load(le_filename)

# Load the data
with open("content.json") as data_file:
    data = json.load(data_file)

def preprocess_string(string):
    string = string.lower()
    exclude = set(punctuation)
    string = ''.join(ch for ch in string if ch not in exclude)
    return string

def get_response(input):
    input = preprocess_string(input)
    prob = model.predict([input])
    results = le.classes_[prob.argmax()]

    if prob.max() == 0.9999302625656128:
        return random.choice(["Waduh kayaknya pertanyaan anda blom ada di database saya", "Maaf, sepertinya saya blom belajar tentang itu."])
    else:
        for tg in data['intents']:
            if tg['tag'] == results:
                responses = tg['responses']
        if results == '':
            return "END CHAT"
        return random.choice(responses)

@app.route('/chat', methods=['POST'])
def chat():
    input = request.data.decode('utf-8')  # Mendekode data dari byte menjadi string
    response = get_response(input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
