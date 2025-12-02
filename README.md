# Real-Time American Sign Language Alphabet Recognition and Speech Translation

## 📖 Project Overview
This project aims to bridge the communication gap between the hearing and speech-impaired community and others by recognizing American Sign Language (ASL) alphabet gestures in real-time using a webcam. The recognized hand gesture is converted into corresponding text and speech output for easy understanding, enabling smoother human-computer interaction and inclusive communication.

## 🛠 Technology Stack
* **Frontend:** HTML5, CSS3, JavaScript
* **Backend Framework:** Flask (Python micro web framework)
* **Deep Learning Framework:** TensorFlow / Keras
* **Model Type:** CNN-LSTM (Convolutional Neural Network combined with Long Short-Term Memory)
* **Text-to-Speech Engine:** Google Text-to-Speech (gTTS)
* **Image Processing:** Pillow (PIL)
* **Programming Language:** Python
* **Deployment:** Local Flask Server

## ⚙️ System Workflow
1.  **Start:** The user opens the web application and clicks "Start Camera" to activate the webcam.
2.  **Input:** The user displays a single hand sign clearly.
3.  **Capture:** The user clicks "Capture Sequence," which records a short frame sequence.
4.  **Processing:** * Frames are sent to the Flask backend.
    * The system pre-processes frames (resize, normalize) and feeds them into the CNN-LSTM model.
5.  **Prediction:** The model predicts the character and calculates a confidence score.
6.  **Output:** The result is displayed on the UI, and the converted speech is played aloud automatically.

## 📊 Performance & Results
The model demonstrates strong predictive power and effective generalization, optimized through hyperparameter tuning (adding Dropout and setting Adam learning rate to 0.0005).

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | 92.41% |
| **Test Loss** | 0.2188 |
| **Overall Classification Accuracy** | 91.4% |

The system shows consistent learning with accuracy rising above 80% on training/validation sets and high precision/recall across most alphabet classes.

## 👥 Team Members
* **Dhruti P Shetty** (Reg no: 230968025)
* **P Pattabhi Ram** (Reg no: 230968033)

## 🚀 Future Scope
While currently hosted locally, future enhancements include:
* Cloud deployment.
* Mobile integration.
* Multi-hand gesture recognition.