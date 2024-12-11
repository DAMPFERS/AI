import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.models import load_model

# 1. Генерация данных
# Генерируем выборку X из нормального распределения N(0, 10)
# и шум e из нормального распределения N(0, 0.3)
np.random.seed(42)  # Фиксируем сид для воспроизводимости
X = np.random.normal(loc=0, scale=10, size=1000).reshape(-1, 1)  # Входные данные
e = np.random.normal(loc=0, scale=0.3, size=1000).reshape(-1, 1)  # Шум
Y = np.sin(X) * X + e  # Целевая переменная

# Сохраняем данные в CSV
data = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten()})
data.to_csv("generated_data.csv", index=False, sep=';')

# 2. Определение модели
# Входной слой
input_layer = Input(shape=(1,), name="Input")

# Кодировщик: несколько скрытых слоев для уменьшения размерности
encoded = Dense(64, activation='relu', name="Encoder_Layer1")(input_layer)
encoded = Dense(32, activation='relu', name="Encoder_Layer2")(encoded)

# Закодированные данные
bottleneck = Dense(16, activation='relu', name="Bottleneck")(encoded)

# Декодировщик: восстанавливаем исходные данные
decoded = Dense(32, activation='relu', name="Decoder_Layer1")(bottleneck)
decoded = Dense(64, activation='relu', name="Decoder_Layer2")(decoded)
decoded_output = Dense(1, activation='linear', name="Decoded_Output")(decoded)

# Регрессионная часть: предсказание целевой переменной
regression = Dense(64, activation='relu', name="Regression_Layer1")(bottleneck)
regression = Dense(32, activation='relu', name="Regression_Layer2")(regression)
regression_output = Dense(1, activation='linear', name="Regression_Output")(regression)

# Полная модель: на вход подаются X, на выходе закодированные данные, декодированные данные и регрессия
full_model = Model(inputs=input_layer, outputs=[decoded_output, regression_output], name="Autoencoder_Regression")

# 3. Компиляция модели
# Оптимизатор Adam, функция потерь MSE, метрика MSE
full_model.compile(optimizer=Adam(learning_rate=0.001),
                   loss=[MeanSquaredError(), MeanSquaredError()],  # Функции потерь для декодирования и регрессии
                   metrics=[[MeanSquaredError()], [MeanSquaredError()]])  # Метрики для каждого выхода

# 4. Обучение модели
# Обучаем модель на данных, разбиваем на 80% обучающей и 20% валидационной выборки
history = full_model.fit(X, [X, Y], epochs=100, batch_size=32, validation_split=0.2)

# 5. Сохранение модели
full_model.save("full_model.h5")  # Сохраняем полную модель в формате h5

# 6. Загрузка модели
# Загружаем модель для проверки
loaded_model = load_model("full_model.h5", custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError})

# 7. Разделение на части
# Модель кодирования (Входные данные -> Закодированные данные)
encoder_model = Model(inputs=input_layer, outputs=bottleneck, name="Encoder")
encoder_model.save("encoder_model.h5")  # Сохраняем модель кодирования в формате h5

# Модель декодирования (Закодированные данные -> Декодированные данные)
encoded_input = Input(shape=(16,), name="Encoded_Input")  # Вход закодированных данных
decoder_layer = full_model.get_layer("Decoder_Layer1")(encoded_input)
decoder_layer = full_model.get_layer("Decoder_Layer2")(decoder_layer)
decoder_output = full_model.get_layer("Decoded_Output")(decoder_layer)
decoder_model = Model(inputs=encoded_input, outputs=decoder_output, name="Decoder")
decoder_model.save("decoder_model.h5")  # Сохраняем модель декодирования в формате h5

# Регрессионная модель (Входные данные -> Результат регрессии)
regression_output = full_model.get_layer("Regression_Layer1")(bottleneck)
regression_output = full_model.get_layer("Regression_Layer2")(regression_output)
regression_output = full_model.get_layer("Regression_Output")(regression_output)
regression_model = Model(inputs=input_layer, outputs=regression_output, name="Regression")
regression_model.save("regression_model.h5")  # Сохраняем регрессионную модель в формате h5

# 8. Генерация и сохранение выходов
# Генерируем закодированные данные
encoded_data = encoder_model.predict(X)
pd.DataFrame(encoded_data).to_csv("encoded_data.csv", index=False, sep=';')

# Генерируем декодированные данные
decoded_data = decoder_model.predict(encoded_data)
pd.DataFrame(decoded_data).to_csv("decoded_data.csv", index=False, sep=';')

# Генерируем результаты регрессии
regression_results = regression_model.predict(X)
results = pd.DataFrame({'True_Y': Y.flatten(), 'Predicted_Y': regression_results.flatten()})
results.to_csv("regression_results.csv", index=False, sep=';')

print("Код завершён успешно. Данные и модели сохранены.")
