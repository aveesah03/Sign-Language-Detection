from flask import Flask, Response, request, redirect
from tensorflow.keras.utils import custom_object_scope
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

app = Flask(__name__)

stream_ready = True

# ===============================
# Language → Actions → Models
# ===============================
LANGUAGES = {
    "isl": {
        "actions": [
            'Good Morning',
            'See you tomorrow',
            'Nice to meet you',
            'I am sorry for this mistake',
            'How are you'
        ],
        "models": {
            "cnn": "cnn_isl.h5",
            "gru": "gru_isl.h5",
            "lstm": "lstm_isl.h5",
            "transformer": "transformer_isl.h5"
        }
    },
    "asl": {
        "actions": [
            'Nice to meet you',
            'I am learning sign',
            'Whats your name',
            'Where are you from',
            'How are you'
        ],
        "models": {
            "cnn": "cnn_asl.h5",
            "gru": "gru_asl.h5",
            "lstm": "lstm_asl.h5",
            "transformer": "transformer_asl.h5"
        }
    },
    "bsl": {
        "actions": [
            'Hello, welcome',
            'Nice to see you',
            'What is your name',
            'How can I help you',
            'Be careful please'
        ],
        "models": {
            "cnn": "cnn_bsl.h5",
            "gru": "gru_bsl.h5",
            "lstm": "lstm_bsl.h5",
            "transformer": "transformer_bsl.h5"
        }
    }
}

# ===============================
# Current selection
# ===============================
current_language = "asl"
current_model_type = "cnn"
actions = LANGUAGES[current_language]["actions"]
model = tf.keras.models.load_model(
    LANGUAGES[current_language]["models"][current_model_type]
)

model_version = 0  

# ===============================
# MediaPipe
# ===============================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



SEQUENCE_LENGTH = 120

def extract_keypoints(results):
    pose = np.array([[p.x, p.y, p.z, p.visibility]
                     for p in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)

    face = np.array([[f.x, f.y, f.z]
                     for f in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[l.x, l.y, l.z]
                   for l in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[r.x, r.y, r.z]
                   for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])

# ===============================
# Custom Transformer Layer
# ===============================
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(seq_len, d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_emb(positions)

# ===============================
# Video stream
# ===============================
def generate_frames():
    global model_version, stream_ready
    local_version = model_version

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    sequence = []
    predictions = []

    while True:
        if local_version != model_version:
            break

        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            current_model = model
            res = current_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            pred = np.argmax(res)
            confidence = res[pred]

            predictions.append(pred)
            predictions = predictions[-10:]

            if predictions.count(pred) > 6 and confidence > 0.6:
                cv2.putText(
                    frame,
                    f"{actions[pred]} ({confidence:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() +
               b"\r\n")
        
    stream_ready = True
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

# ===============================
# Routes
# ===============================
@app.route("/set_options")
def set_options():
    global model, actions, current_language, current_model_type, model_version, stream_ready

    language = request.args.get("language")
    model_type = request.args.get("model")

    if language in LANGUAGES and model_type in LANGUAGES[language]["models"]:
        stream_ready = False
        current_language = language
        current_model_type = model_type
        actions = LANGUAGES[language]["actions"]
        tf.keras.backend.clear_session()
        model_path = LANGUAGES[language]["models"][model_type]
        if model_type == "transformer":
            with custom_object_scope({"PositionalEmbedding": PositionalEmbedding}):
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )
        else:
            model = tf.keras.models.load_model(model_path)
        model_version += 1

    return redirect("/")

@app.route("/")
def index():
    signs = LANGUAGES[current_language]["actions"]

    def selected(v, c):
        return "selected" if v == c else ""

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                padding: 20px;
            }}
            h2 {{ text-align: center; }}
            form {{
                display: flex;
                gap: 12px;
                justify-content: center;
                flex-wrap: wrap;
                background: #fff;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            select, button {{
                padding: 8px;
                border-radius: 6px;
            }}
            button {{
                background: #007bff;
                color: white;
                border: none;
                cursor: pointer;
            }}
            .container {{
                display: flex;
                gap: 20px;
                justify-content: center;
                flex-wrap: wrap;
            }}
            .signs {{
                background: #fff;
                padding: 15px;
                border-radius: 8px;
                width: 260px;
            }}
            .video img {{
                width: 720px;
                max-width: 100%;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>

        <h2>Sign Language Recognition</h2>

        <form action="/set_options">
            <select name="language">
                <option value="isl" {selected("isl", current_language)}>ISL</option>
                <option value="asl" {selected("asl", current_language)}>ASL</option>
                <option value="bsl" {selected("bsl", current_language)}>BSL</option>
            </select>

            <select name="model">
                <option value="cnn" {selected("cnn", current_model_type)}>CNN</option>
                <option value="gru" {selected("gru", current_model_type)}>GRU</option>
                <option value="lstm" {selected("lstm", current_model_type)}>LSTM</option>
                <option value="transformer" {selected("transformer", current_model_type)}>Transformer</option>
            </select>

            <button type="submit">Apply</button>
        </form>

        <div class="container">
            <div class="signs">
                <h3>Possible Signs ({current_language.upper()})</h3>
                <ul>
                    {''.join(f"<li>{s}</li>" for s in signs)}
                </ul>
            </div>

            <div class="video">
                <img id="videoFeed">
            </div>
        </div>

        <script>
            const img = document.getElementById("videoFeed");
            const version = {model_version};

            setTimeout(() => {{
                img.src = "/video_feed?v=" + version + "&t=" + Date.now();
            }}, 300);
        </script>

    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)