# !pip install ultralytics
# !pip install scikit-learn

from ultralytics import YOLO

y_type = 's'
y_epochs = 100
y_batch = 16
y_size = 640
n = f'yolov8{y_type}_{y_epochs}_{y_batch}_{y_size}'

model = YOLO(f'yolov8{y_type}.pt')

data_yaml_path = 'data.yaml'

results = model.train(
    data=data_yaml_path,        # Путь к конфигурационному файлу с разметкой данных
    epochs=y_epochs,                 # Количество эпох для обучения
    imgsz=y_size,                 # Размер входных изображений
    batch=y_batch,                   # Размер батча
    conf=0.3,
    iou=0.25,
    lr0=0.005,                 # Начальная скорость обучения
    optimizer='Adam',          # Оптимизатор
    name=n,
    val=True
)

print("Результаты дообучения модели:")
print(f"Точность (Precision): {results.box.mp:.4f}")
print(f"Полнота (Recall): {results.box.mr:.4f}")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")
