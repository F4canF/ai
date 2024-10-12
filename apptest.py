import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # type: ignore

# 모델 불러오기
model = tf.keras.models.load_model('handwriting_model.keras')  # 모델 경로 수정

# 이미지 경로
img_path = 'ㄱ.jpeg'  # 이미지 경로 수정

# 이미지 전처리 및 예측
test_image = image.load_img(img_path, target_size=(64, 64))  # 이미지 불러오기
test_image = image.img_to_array(test_image)  # 이미지를 배열로 변환
test_image = np.expand_dims(test_image, axis=0)  # 배치 차원 추가
test_image /= 255.0  # 정규화

# 예측 수행
result = model.predict(test_image)

# 예측된 클래스 인덱스
predicted_class_index = np.argmax(result, axis=1)

# 클래스 인덱스와 문자 매핑 (예시)
class_indices = {
    1: 'ㄱ', 2: 'ㄴ', 3: 'ㄷ', 
    4: 'ㅏ', 5: 'ㅑ', 
    # 나머지 문자 매핑 추가...
    138: '!', 139: '@'
}

# 예측된 라벨 얻기
if predicted_class_index[0] in class_indices:
    predicted_label = class_indices[predicted_class_index[0]]
else:
    predicted_label = "알 수 없는 문자"

print(f"예측된 결과: {predicted_label}")
