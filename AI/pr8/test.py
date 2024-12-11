import numpy as np                                      # Для работы с массивами
from tensorflow.keras.utils import to_categorical       # Для преобразования меток в one-hot кодировку
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split    # Для разделения данных на тренировочную и тестовую выборки
from var4 import gen_data                               # Импорт функции для генерации данных
import pr8                                              # Ваш кастомный CallBack для сохранения моделей
import matplotlib.pyplot as plt                         # Для построения графиков
import os
from tensorflow.keras.models import load_model

# Генерация данных
data, labels = gen_data(size=1000, img_size=50)  # Генерация 1000 изображений размером 50x50 пикселей

# Преобразование строковых меток в числовые
class_map = {'Cross': 0, 'Line': 1}  # Определяем соответствие между строковыми и числовыми метками
numeric_labels = np.vectorize(class_map.get)(labels.flatten())  # Преобразуем метки в числовой формат

# Нормализация данных
data = data / 255.0  # Приводим значения пикселей к диапазону [0, 1]
data = data[..., np.newaxis]  # Добавляем измерение для канала (1 для черно-белых изображений)

# Проверяем распределение классов
unique, counts = np.unique(numeric_labels, return_counts=True)
print(f"Распределение классов: {dict(zip(unique, counts))}")

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    data, numeric_labels, test_size=0.2, random_state=42
)

# Преобразование меток в формат one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Импорт необходимых классов для построения модели
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Создание модели сверточной нейронной сети
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),  # Сверточный слой с 32 фильтрами
    MaxPooling2D((2, 2)),  # Слой подвыборки (пулинга) для уменьшения размерности
    Dropout(0.25),  # Слой регуляризации для предотвращения переобучения

    Conv2D(64, (3, 3), activation='relu'),  # Еще один сверточный слой с 64 фильтрами
    MaxPooling2D((2, 2)),  # Еще один слой подвыборки
    Dropout(0.25),  # Регуляризация

    Conv2D(128, (3, 3), activation='relu'),  # Углубляем сеть (128 фильтров)
    MaxPooling2D((2, 2)),  # Подвыборка
    Dropout(0.25),

    Flatten(),  # Преобразуем данные в одномерный массив для Dense-слоев
    Dense(256, activation='relu'),  # Полносвязный слой с 256 нейронами
    Dropout(0.5),  # Регуляризация
    Dense(2, activation='softmax')  # Выходной слой с 2 классами
])

# Компиляция модели
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Используем оптимизатор Adam с пониженной скоростью обучения
    loss='categorical_crossentropy',  # Функция потерь для задачи классификации
    metrics=['accuracy']  # Метрика для отслеживания точности
)

# Использование кастомного CallBack
save_epochs = [5, 25, 45]  # Эпохи, на которых нужно сохранять модели
custom_callback = pr8.CustomModelSaver(save_epochs, prefix="cross_line_model", directory="saved_models1")

# Обучение модели
history = model.fit(
    X_train, y_train,  # Обучающие данные
    validation_split=0.2,  # Используем 20% обучающих данных для проверки
    epochs=50,  # Количество эпох
    batch_size=32,  # Размер батча
    verbose=1,  # Показ прогресса обучения
    callbacks=[custom_callback]
)

# Оценка модели на тестовой выборке
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность на тестовой выборке: {test_accuracy * 100:.2f}%")

# --- ВИЗУАЛИЗАЦИЯ --- #

# Функция для построения графиков истории обучения
def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Построение графиков обучения
plot_training_history(history)

# Функция для оценки сохранённых моделей
def evaluate_saved_models(directory, X_test, y_test):
    models = sorted(os.listdir(directory))
    epochs = []
    accuracies = []

    for model_file in models:
        filepath = os.path.join(directory, model_file)
        model = load_model(filepath)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        epochs.append(int(model_file.split('_')[-1].split('-')[-1].split('.')[0]))
        accuracies.append(accuracy)
        print(f"Модель {model_file} -> Точность: {accuracy * 100:.2f}%")
    
    epochs.sort()
    accuracies.sort()
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='blue')
    plt.title('Test Accuracy of Saved Models')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Оценка сохранённых моделей
evaluate_saved_models("saved_models", X_test, y_test)
