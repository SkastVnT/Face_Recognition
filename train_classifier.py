import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from scipy.spatial.distance import cosine

# Đường dẫn đến dữ liệu embedding đã tạo
encodings_path = 'encodings/embeddings.pkl'
with open(encodings_path, 'rb') as f:
    data = pickle.load(f)

X = np.array(data['embeddings'])
y = np.array(data['names'])

# Mã hóa nhãn (chuyển tên thành số)
le = LabelEncoder()
y_num = le.fit_transform(y)

# Huấn luyện SVM (kernel tuyến tính), có xác suất
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y_num)

# Lưu model và encoder
os.makedirs('models', exist_ok=True)
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump((clf, le), f)

print("Huấn luyện xong. Model SVM và LabelEncoder đã lưu tại 'models/classifier.pkl'")
