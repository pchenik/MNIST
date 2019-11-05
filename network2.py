from mnist import MNIST
import numpy as np
import math
import matplotlib.pyplot as plt
import random


mndata = MNIST('samples')
img_train, lbl_train = mndata.load_training()
img_test, lbl_test = mndata.load_testing()

for i in range(0, len(img_test)):
    img_test[i] = np.array(img_test[i]) / 255

for i in range(0, len(img_train)):
    img_train[i] = np.array(img_train[i]) / 255

def S(x):
    return 1 / (1 + math.e ** (-x))

def dsigm(y):
 return S(y) * (1 - S(y))

def calc(img):
    r = list(range(0, 10))
    for j in range(0, 10):
        sum = 0.0
        for i in range(0, 784):
            sum = sum + img[i] * w[j][i]
        r[j] = S(sum)

    return np.argmax(r)


w = (np.random.rand(10, 784) - 1) / 10
yy = []

for n in range(len(img_train) // 3):

      img = img_train[n]
      true_ind = lbl_train[n]

      #Вектор функцкия активации для 10 нейронов во внутреннем слое
      r = [0.0] * 10

      #Возбуждение скрытого слоя, прямой шаг распространения
      for j in range(0, 10):
        sum = 0.0
        for i in range(0, 784):
            sum = sum + img[i] * w[j][i]
        r[j] = S(sum)


      #Предсказанный вектор-классификации
      predict_ind = np.argmax(r)
      resp = np.zeros(10, dtype=np.float32)
      resp[predict_ind] = 1.0


      #Вектор
      y = np.zeros(10, dtype=np.float32)
      y[true_ind] = 1.0

      loss = 0.0
      for j in range(0, 10):
          loss = loss + (r[j] - y[j]) ** 2

      #yy.append(loss)

      #Вычисление поправки для скрытого слоя
      delta = [0.0] * 10
      for k in range(0, 10):
        error = y[k] - resp[k]
        delta[k] = dsigm(resp[k]) * error

      #Коэффициент, задающий скорость градиентного спуска
      N = 0.5

      #Обновление весов связеи между узлами входного и скрытого слоя
      for j in range(0, 784):
        for k in range(0, 10):
            delta_w = delta[k] * img[j]
            w[k][j] = w[k][j] + N * delta_w

#Проверяем точность на тестовом наборе
true = 0
#x = np.arange(0, 20000, 1)
#plt.plot(x, yy)
#plt.show()

for i in range(0, len(img_test)):
    img = img_test[i]
    predict = calc(img)
    if predict == lbl_test[i]:
        true = true + 1

print("Accuracy = ", true / len(img_test), '\n')