import librosa 
import numpy as np 
import tensorflow as tf 
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Audio received successfully'

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_file.save('audio.mp3')
    y, sr = librosa.load("C:\\Users\\druth\\Desktop\\stress detection\\audio.mp3", duration=5.0)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)  
    mfcc = np.expand_dims(mfcc, axis=0) 

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']
    emotion_model = tf.keras.models.load_model('C:\\Users\\druth\\Desktop\\stress detection\\Voice-Emotion-Backend\\model.h5') # load the model
    predictions = emotion_model.predict(mfcc) 
    emotion = np.argmax(predictions)
    print(f'The emotion of the audio file is {emotion_labels[emotion]}.')
    return jsonify(emotion_labels[emotion]), 200


if __name__ == '__main__':
    app.run(debug=True,port=8000)

    