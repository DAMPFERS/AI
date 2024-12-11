from tensorflow.keras.models import Sequential  # Используем для создания последовательной модели
from tensorflow.keras import layers  # Импортируем слои для построения нейросети
import numpy as np  # Для работы с массивами
import random  # Для добавления случайного шума
import math  # Для работы с математическими функциями
import matplotlib.pyplot as plt  # Для построения графиков

# Определяем вспомогательную функцию для создания дополнительного сигнала (функция из вашего варианта)
def func(i):
    i = i % 31  # Ограничиваем индекс значениями от 0 до 30 (повторяющийся паттерн)
    return ((i - 15) ** 2) / 100 - 4  # Параболическая функция с вертикальным смещением

# Генерация последовательности с заданной длиной
def gen_sequence(seq_len=1000):
    # Формируем последовательность, состоящую из суммы косинуса, функции func и случайного шума
    seq = [math.cos(i / 2) + func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)  # Возвращаем последовательность в виде массива numpy

# Функция для подготовки данных из последовательности
def gen_data_from_sequence(seq_len=1006, lookback=10):
    """
    - seq_len: длина последовательности
    - lookback: размер окна данных (количество прошлых значений для предсказания следующего)
    """
    seq = gen_sequence(seq_len)  # Генерируем последовательность
    # Создаем входные данные (прошлые значения для каждого окна)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    # Создаем ожидаемые результаты (следующее значение после окна)
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future  # Возвращаем входы и выходы

# Генерация данных и их разбиение
data, res = gen_data_from_sequence()  # Генерируем данные (входы и целевые значения)
dataset_size = len(data)  # Размер всего набора данных
train_size = (dataset_size // 10) * 7  # 70% данных - тренировочная выборка
val_size = (dataset_size - train_size) // 2  # 15% - валидационная выборка
# Оставшиеся 15% - тестовая выборка

# Разбиваем данные на тренировочные, валидационные и тестовые
train_data, train_res = data[:train_size], res[:train_size]  # Тренировочная выборка
val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]  # Валидация
test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]  # Тестовая выборка

# Создаем нейросеть
model = Sequential()  # Инициализация последовательной модели
# Добавляем первый слой GRU
model.add(layers.GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
# Параметры:
#  - 32 - количество нейронов
#  - recurrent_activation='sigmoid' - функция активации для рекуррентного состояния
#  - input_shape=(None, 1) - входная размерность (длина окна и 1 канал)
#  - return_sequences=True - возвращаем последовательности для следующего слоя

# Добавляем слой LSTM
model.add(layers.LSTM(32, activation='relu', return_sequences=True, dropout=0.2))
# Параметры:
#  - 32 - количество нейронов
#  - activation='relu' - функция активации
#  - return_sequences=True - возвращаем последовательности для следующего слоя
#  - dropout=0.2 - добавляем регуляризацию Dropout для уменьшения переобучения

# Добавляем еще один слой GRU
model.add(layers.GRU(32, recurrent_dropout=0.2))
# Параметры:
#  - 32 - количество нейронов
#  - recurrent_dropout=0.2 - регуляризация для рекуррентных соединений

# Выходной слой
model.add(layers.Dense(1))  # Один выходной нейрон для предсказания следующего значения

# Компилируем модель
model.compile(optimizer='nadam', loss='mse')  
# Параметры:
#  - optimizer='nadam' - алгоритм оптимизации Nadam
#  - loss='mse' - функция потерь (среднеквадратичная ошибка)

# Обучаем модель
history = model.fit(train_data, train_res, epochs=30, validation_data=(val_data, val_res))
# Параметры:
#  - train_data и train_res - обучающие данные
#  - epochs=50 - количество эпох обучения
#  - validation_data - данные для оценки валидационной ошибки

# Построение графика функции потерь
loss = history.history['loss']  # История ошибки на тренировочных данных
val_loss = history.history['val_loss']  # История ошибки на валидации
plt.plot(range(len(loss)), loss, label='Train Loss')  # График ошибки на тренировке
plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')  # График ошибки на валидации
plt.legend()
plt.show()

# Предсказание на тестовой выборке
predicted_res = model.predict(test_data)  # Предсказания модели

# Построение графика предсказанных значений и реальных значений
pred_length = range(len(predicted_res))  # Диапазон значений для оси X
plt.plot(pred_length, predicted_res, label='Predicted')  # Предсказанные значения
plt.plot(pred_length, test_res, label='Actual')  # Реальные значения
plt.legend()
plt.show()
