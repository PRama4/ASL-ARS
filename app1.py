from flask import Flask, render_template, request, jsonify
import base64
import io
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from gtts import gTTS

app = Flask(__name__)

model = keras.models.load_model(os.path.join('models', 'cnn_lstm_seq_model1.keras'))
sequence_length = 5  # Adjust to your actual model's requirement
img_size = (100, 100)  # Set according to your training input size
num_classes = 29
class_to_letter = {
    **{i: chr(65 + i) for i in range(26)},
    26: 'del',
    27: 'space',
    28: ''
}

frames = []  # Global frame buffer for sequences

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global frames
    data = request.json['image']
    img_str = data.split(",", 1)[1]
    img_bytes = base64.b64decode(img_str)

    # Convert to image and preprocess
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(img_size)
    img_np = np.asarray(img).astype(np.float32) / 255.0  # Normalized

    frames.append(img_np)
    # Keep only the latest sequence_length frames
    if len(frames) > sequence_length:
        frames = frames[-sequence_length:]

    predicted_text = ""
    audio_b64 = ""
    confidence = 0.0
    if len(frames) == sequence_length:
        input_sequence = np.expand_dims(np.array(frames), axis=0)
        predictions = model.predict(input_sequence)
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_text = class_to_letter.get(predicted_class_index, "")
        confidence = float(np.max(predictions[0]))  # Model confidence for this frame sequence
        frames = []

        # gTTS: Only speak if a valid prediction was made
        if predicted_text and predicted_text not in ['del', 'space', '']:
            tts = gTTS(predicted_text)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_b64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
    return jsonify({
        'text': predicted_text if predicted_text else "Show sign and capture", 
        'audio': audio_b64,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
