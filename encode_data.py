import os
import cv2
import pickle
import numpy as np
from data_processing import extract_face, get_embedding

# Thư mục chứa ảnh training (mỗi người 1 thư mục con)
dataset_dir = 'dataset/known_faces'
encodings = {'names': [], 'embeddings': []}

# Duyệt qua từng thư mục con (tương ứng mỗi người)
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    # Duyệt qua từng ảnh của người đó
    for filename in os.listdir(person_dir):
        file_path = os.path.join(person_dir, filename)
        # Đọc ảnh bằng OpenCV (mặc định BGR), chuyển sang RGB
        img = cv2.imread(file_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Phát hiện và trích xuất khuôn mặt
        face = extract_face(img_rgb)
        if face is None:
            print(f"Không phát hiện khuôn mặt trong ảnh {file_path}")
            continue
        # Trích xuất embedding
        embedding = get_embedding(face)
        encodings['names'].append(person_name)
        encodings['embeddings'].append(embedding)
        print(f"Đã xử lý ảnh: {file_path}")

# Lưu embeddings và tên vào file pickle
os.makedirs('encodings', exist_ok=True)
with open('encodings/embeddings.pkl', 'wb') as f:
    pickle.dump(encodings, f)

print("Hoàn thành tạo embeddings và lưu vào 'encodings/embeddings.pkl'")
