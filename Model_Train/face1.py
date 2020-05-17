import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# 훈련데이터 로드
train_data_dir = pathlib.Path('C:/Users/user/PycharmProjects/project(0505)/train')

train_park_image_count = len(list(train_data_dir.glob('park/*.jpg')))
train_ma_image_count = len(list(train_data_dir.glob('ma/*.jpg')))
train_lee_image_count = len(list(train_data_dir.glob('lee/*.jpg')))

print(train_park_image_count)
print(train_ma_image_count)
print(train_lee_image_count)

class_names = ['park', 'ma', 'lee']
train_images = []

for i in range(train_park_image_count):
  img = Image.open('C:/Users/user/PycharmProjects/project(0505)/train/park/{0}.jpg'.format(i+1))
  img = img.resize((150, 150))
  img = np.array(img)
  train_images.append(img)

for i in range(train_ma_image_count):
    img = Image.open('C:/Users/user/PycharmProjects/project(0505)/train/ma/{0}.jpg'.format(i + 1))
    img = img.resize((150, 150))
    img = np.array(img)
    train_images.append(img)

for i in range(train_lee_image_count):
    img = Image.open('C:/Users/user/PycharmProjects/project(0505)/train/lee/{0}.jpg'.format(i + 1))
    img = img.resize((150, 150))
    img = np.array(img)
    train_images.append(img)

train_images = np.array(train_images)

print(train_images.shape)

train_labels = []

for i in range(train_park_image_count):
  train_labels.append(0)

for i in range(train_ma_image_count):
  train_labels.append(1)

for i in range(train_lee_image_count):
  train_labels.append(2)

train_labels = np.array(train_labels, dtype=np.uint8)
print(train_labels.shape)

temp = [[x, y] for x, y in zip(train_images, train_labels)]
random.shuffle(temp)

train_images = [n[0] for n in temp]
train_labels = [n[1] for n in temp]

train_images = np.array(train_images)
train_labels = np.array(train_labels, dtype=np.uint8)

# 이미지 크기 조절
train_images = train_images / 255.0

# 이미지 시각화
def plotImages(images, labels):
  plt.figure(figsize=(10,10))
  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[i]])
  plt.show()

plotImages(train_images, train_labels)

# 모델 구성
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(150, 150, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 학습
model.fit(train_images, train_labels, epochs=10, batch_size=100)

# 모델 저장
model.save('my_model.h5')