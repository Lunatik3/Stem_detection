# !pip install ultralytics
# !pip install scikit-learn
import cv2
import os
from matplotlib import pyplot as plt
from ultralytics import YOLO


# Загрузка дообученной модели
model = YOLO('best.pt')

results = model.predict(
    source='images/test',
    save=True,
    project='results',  # Указать корневую папку для сохранения
    name='results/results',      # Имя подпапки для текущей сессии
)

# Путь к папке с изображениями
folder_path = 'results/results'

# Список всех файлов в папке
image_files = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Отображение всех изображений
for image_file in image_files:
    # Полный путь к изображению
    img_path = os.path.join(folder_path, image_file)

    # Загрузка изображения
    img = cv2.imread(img_path)

    # Проверяем, что файл - изображение
    if img is not None:
        # Отображение изображения
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(image_file)
        plt.axis('off')
        plt.show()
