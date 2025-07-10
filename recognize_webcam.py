import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from data_processing import get_embedding
from datetime import datetime
import pandas as pd
import os

# Tải mô hình SVM và LabelEncoder
with open('models/classifier.pkl', 'rb') as f:
    clf, le = pickle.load(f)

detector = MTCNN()
cap = cv2.VideoCapture(0)

THRESHOLD = 0.6  # ngưỡng xác suất để xác định là người trong dataset

# Danh sách người đã điểm danh trong phiên
attended_names = set()

# Tạo dataframe để ghi điểm danh
attendance_data = []

print("Bắt đầu điểm danh... Nhấn ESC để kết thúc.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(frame_rgb)

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
            name = "Khong co trong dataset"
        else:
            name = le.inverse_transform([best_idx])[0]
            # Nếu chưa điểm danh người này, thêm vào danh sách
            if name not in attended_names:
                attended_names.add(name)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"📌 {name} - đã điểm danh lúc {timestamp}")
                attendance_data.append({'Họ tên': name, 'Thời gian': timestamp})

        # Hiển thị lên webcam
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Diem danh bang khuon mat", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if attendance_data:
    # Tạo tên file theo ngày
    today_str = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join('checkin', f'attendance_{today_str}.xlsx')

    df = pd.DataFrame(attendance_data)

    # Nếu file đã tồn tại, nối thêm dữ liệu
    if os.path.exists(file_path):
        old_df = pd.read_excel(file_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_excel(file_path, index=False)
    print(f"\n✅ Danh sách điểm danh đã được lưu vào: {file_path}")
else:
    print("❗Không có người nào được điểm danh trong phiên này.")
