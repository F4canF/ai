from dotenv import load_dotenv
import os
import cv2
import numpy as np
import tensorflow as tf
import pymysql
import pyautogui
import time

# MariaDB 연결 정보
load_dotenv()
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

# 모델 경로 설정
keras_model_path = 'handwriting_model.keras'

# 모델 불러오기
model = tf.keras.models.load_model(keras_model_path)

# 자모음 매핑
class_indices = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 
    5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 
    15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ',
    20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
    24: 'A', 25: 'B', 26: 'C', 27: 'D', 28: 'E',
}

chosung_list = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ'
jungsung_list = 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ'
jongsung_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# DB에서 이미지 가져오기
def get_image_from_db(image_colums_id):
    conn = None
    try:
        conn = pymysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        query = "SELECT image_data FROM handwriting_images WHERE id = %s"
        cursor.execute(query, (image_colums_id,))
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            image_data = np.frombuffer(result[0], dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return image
        else:
            print("이미지를 찾을 수 없습니다.")
            return None
            
    except pymysql.Error as e:
        print(f"MariaDB Error: {e}")
        return None
    finally:
        if conn:
            conn.close()

# 텍스트 추출 함수
def extract_text_from_image(image):
    roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)

    result = model.predict(roi)
    predicted_class_index = np.argmax(result, axis=1)[0]
    predicted_label = class_indices.get(predicted_class_index, "알 수 없는 문자")
    return predicted_label

# 자모음을 받아 조합하여 한글을 만드는 함수
def combine_characters(chosung, jungsung, jongsung=''):
    chosung_index = chosung_list.index(chosung)
    jungsung_index = jungsung_list.index(jungsung)
    jongsung_index = jongsung_list.index(jongsung) if jongsung else 0

    korean_char_code = 0xAC00 + (chosung_index * 588) + (jungsung_index * 28) + jongsung_index
    return chr(korean_char_code)

# 조합된 글자를 저장할 변수
chosung = None
jungsung = None
jongsung = None

# 이미지 ID 초기화
image_colums_id = 1

def process_image(image_colums_id):
    global chosung, jungsung, jongsung

    image = get_image_from_db(image_colums_id)
    if image is not None:
        predicted_character = extract_text_from_image(image)

        if predicted_character in chosung_list:
            chosung = predicted_character
        elif predicted_character in jungsung_list and chosung:
            jungsung = predicted_character
        elif predicted_character in jongsung_list and chosung and jungsung:
            jongsung = predicted_character
        
        if chosung and jungsung:
            final_char = combine_characters(chosung, jungsung, jongsung)
            print(f"조합된 글자: {final_char}")
            pyautogui.typewrite(final_char)
            chosung, jungsung, jongsung = None, None, None
        else:
            pyautogui.typewrite(predicted_character)
    else:
        print("이미지를 처리할 수 없습니다.")

# 사용자 버튼 클릭 이벤트 처리 (예시)
def on_complete_button_click():
    global image_colums_id
    process_image(image_colums_id)
    image_colums_id += 1  # 다음 이미지 ID로 업데이트

# 사용 예시 (버튼 클릭 이벤트 호출)
# 실제로는 GUI 라이브러리와 연결해야 함
if __name__ == "__main__":
    while True:
        on_complete_button_click()
        time.sleep(1)  # 1초 대기
