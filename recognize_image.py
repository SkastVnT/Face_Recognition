import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from data_processing import extract_face, get_embedding

# Tải model SVM và LabelEncoder
with open('models/classifier.pkl', 'rb') as f:
    clf, le = pickle.load(f)

# Đường dẫn ảnh cần nhận diện
image_path = 'test.jpg'  # Thay bằng đường dẫn thật
# Đọc ảnh
img = cv2.imread(image_path)
if img is None:
    print(f"Không thể mở ảnh: {image_path}")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detector = MTCNN()
results = detector.detect_faces(img_rgb)

for result in results:
    x1, y1, width, height = result['box']
    x2, y2 = x1 + width, y1 + height
    face = img_rgb[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (224, 224))
    embedding = get_embedding(face_resized)
    # Dự đoán với SVM
    probs = clf.predict_proba([embedding])[0]
    best_idx = np.argmax(probs)
    prob = probs[best_idx]
    if prob < 0.5:
        name = "Không nhận diện được"
    else:
        name = le.inverse_transform([best_idx])[0]
    # Vẽ bounding box và tên lên ảnh (dùng OpenCV, màu xanh)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# Hiển thị kết quả
cv2.imshow("Kết quả nhận diện", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
