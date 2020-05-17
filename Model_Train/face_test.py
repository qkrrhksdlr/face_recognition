import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import random
from tensorflow import keras

# 테스트데이터 로드
test_data_dir = pathlib.Path('C:/Users/user/PycharmProjects/project(0505)/test')

test_park_image_count = len(list(test_data_dir.glob('park/*.jpg')))
test_ma_image_count = len(list(test_data_dir.glob('ma/*.jpg')))
test_lee_image_count = len(list(test_data_dir.glob('lee/*.jpg')))

print(test_park_image_count)
print(test_ma_image_count)
print(test_lee_image_count)

class_names = ['park', 'ma', 'lee']
test_images = []

for i in range(test_park_image_count):
  img = Image.open('C:/Users/user/PycharmProjects/project(0505)/test/park/{0}.jpg'.format(i+1))
  img = img.resize((150, 150))
  img = np.array(img)
  test_images.append(img)

for i in range(test_ma_image_count):
  img = Image.open('C:/Users/user/PycharmProjects/project(0505)/test/ma/{0}.jpg'.format(i+1))
  img = img.resize((150, 150))
  img = np.array(img)
  test_images.append(img)

for i in range(test_lee_image_count):
  img = Image.open('C:/Users/user/PycharmProjects/project(0505)/test/lee/{0}.jpg'.format(i+1))
  img = img.resize((150, 150))
  img = np.array(img)
  test_images.append(img)

test_images = np.array(test_images)
print(test_images.shape)

test_labels = []

for i in range(test_park_image_count):
  test_labels.append(0)

for i in range(test_ma_image_count):
  test_labels.append(1)

for i in range(test_lee_image_count):
  test_labels.append(2)

test_labels = np.array(test_labels, dtype=np.uint8)

print(test_labels.shape)

temp2 = [[x, y] for x, y in zip(test_images, test_labels)]
random.shuffle(temp2)

test_images = [n[0] for n in temp2]
test_labels = [n[1] for n in temp2]

test_images = np.array(test_images)
test_labels = np.array(test_labels, dtype=np.uint8)

# 이미지 크기 조절
test_images = test_images / 255.0

# 모델 로드
model = keras.models.load_model('my_model.h5')

# 모델 예측
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)
predictions = model.predict(test_images)

# 첫번째 이미지 예측
print(predictions[0])

np.argmax(predictions[0])

# 예측 시각화
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# 테스트 데이터 예측하기
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
plt.show()