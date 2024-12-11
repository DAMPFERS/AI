import os
import datetime
from tensorflow.keras.callbacks import Callback

class CustomModelSaver(Callback):
    def __init__(self, save_epochs, prefix="model", directory="models"):
        """
        Инициализация кастомного CallBack для сохранения моделей.
        :param save_epochs: список эпох, на которых нужно сохранять модель.
        :param prefix: префикс для имени файла.
        :param directory: папка для сохранения моделей.
        """
        super().__init__()
        self.save_epochs = save_epochs
        self.prefix = prefix
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)  # Создать папку, если её нет

    def on_epoch_end(self, epoch, logs=None):
        """
        Сохраняет модель на указанных эпохах.
        :param epoch: текущая эпоха.
        :param logs: данные, собранные за текущую эпоху.
        """
        if (epoch + 1) in self.save_epochs:  # Проверяем, входит ли эпоха в список
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            filename = f"{current_date}_{self.prefix}_epoch-{epoch+1}.h5"
            filepath = os.path.join(self.directory, filename)
            self.model.save(filepath)
            print(f"Модель сохранена: {filepath}")
