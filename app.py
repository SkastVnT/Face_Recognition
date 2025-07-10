from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import pickle
from mtcnn.mtcnn import MTCNN
from data_processing import get_embedding
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)

# Load model
with open('models/classifier.pkl', 'rb') as f:
    clf, le = pickle.load(f)

detector = MTCNN()
attended_names = set()
attendance_data = []
THRESHOLD = 0.6

@app.route('/')
def index():
    return render_template('client.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    global attendance_data, attended_names
    data = request.json
    image_data = data['image'].split(',')[1]  # Remove base64 header
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(frame_rgb)
    recognized = []

    for result in results:
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        face = frame_rgb[y1:y2, x1:x2]

        if face.size == 0:
            continue

        try:
            face_resized = cv2.resize(face, (224, 224))
        except:
            continue

        embedding = get_embedding(face_resized)
        probs = clf.predict_proba([embedding])[0]
        best_idx = np.argmax(probs)
        prob = probs[best_idx]

        if prob < THRESHOLD:
            name = "Không xác định"
        else:
            name = le.inverse_transform([best_idx])[0]
            if name not in attended_names:
                attended_names.add(name)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                attendance_data.append({'Họ tên': name, 'Thời gian': timestamp})

        recognized.append(name)

    return jsonify({'recognized': recognized})

@app.route('/save')
def save_attendance():
    if attendance_data:
        os.makedirs('checkin', exist_ok=True)
        today_str = datetime.now().strftime('%Y-%m-%d')
        file_path = os.path.join('checkin', f'attendance_{today_str}.xlsx')
        df = pd.DataFrame(attendance_data)

        if os.path.exists(file_path):
            old_df = pd.read_excel(file_path)
            df = pd.concat([old_df, df], ignore_index=True)

        df.to_excel(file_path, index=False)
        return f"Đã lưu điểm danh vào: {file_path}"
    return "Không có ai được điểm danh."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
