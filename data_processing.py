import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image

# Tạo detector MTCNN (phát hiện khuôn mặt)
detector = MTCNN()

# Tải model VGGFace2 (ResNet50) để trích xuất embedding khuôn mặt
# include_top=False để không lấy lớp phân loại cuối cùng, pooling='avg' để lấy vector 2048
vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_face(img_rgb, required_size=(224, 224)):
    """
    Phát hiện khuôn mặt trên ảnh RGB và trích xuất vùng khuôn mặt đã resize.
    Trả về mảng numpy kích thước (224,224,3) của khuôn mặt.
    """
    results = detector.detect_faces(img_rgb)
    if results == []:
        return None
    # Lấy bounding box của khuôn mặt đầu tiên
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # Cắt ảnh khuôn mặt và resize về kích thước chuẩn
    face = img_rgb[y1:y2, x1:x2]
    face_image = Image.fromarray(face)
    face_image = face_image.resize(required_size)
    face_array = np.asarray(face_image)
    return face_array

def get_embedding(face_pixels):
    """
    Chuyển ảnh khuôn mặt (đã resize) thành vector embedding 2048 chiều.
    Tiến hành chuẩn hóa đầu vào và L2 normalization.
    """
    # Chuyển đổi sang float và nhân thêm 1 chiều batch
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, axis=0)
    # Chuẩn hóa theo preprocess của VGGFace2 (version=2)
    samples = preprocess_input(samples, version=2)
    # Trích xuất embedding (dạng [1, 2048])
    yhat = vgg_model.predict(samples)
    embedding = yhat[0]
    # L2 normalize để so sánh cosine sau này
    embedding = embedding / np.linalg.norm(embedding)
    return embedding
