import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 저장된 모델 불러오기
model = load_model("dog_cat_classifier.h5")

# 판별할 이미지 경로
img_path = "predict/img/img2.webp"

# 이미지 전처리
img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 예측
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("이 이미지는 개입니다.")
else:
    print("이 이미지는 고양이입니다.")


