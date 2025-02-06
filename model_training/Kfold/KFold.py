# !pip install ultralytics
# !pip install scikit-learn
import cv2
import os
import shutil
from ultralytics import YOLO
import yaml
from sklearn.model_selection import KFold

# Пути к папкам с изображениями и метками
images_path = "dataset/images"
labels_path = "dataset/labels"

y_type = 'n'
y_epochs = 150
y_batch = 16
y_size = 640
k = 4

# Список всех изображений и меток
all_images = sorted(os.listdir(images_path))
all_labels = sorted(os.listdir(labels_path))

# Проверка на колво меток и изображений
assert len(all_images) == len(all_labels), "Количество изображений и меток не совпадает!"

# Инициалиализация KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Список для хранения результатов каждого fold
fold_results = []

model_path = f"yolov8{y_type}.pt"

# k-fold
for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    print(f"Fold {fold + 1}/{k}")

    # Создаем временные папки для текущего fold
    fold_images_train = "images/train"
    fold_images_val = "images/val"
    fold_labels_train = "labels/train"
    fold_labels_val = "labels/val"

    os.makedirs(fold_images_train, exist_ok=True)
    os.makedirs(fold_images_val, exist_ok=True)
    os.makedirs(fold_labels_train, exist_ok=True)
    os.makedirs(fold_labels_val, exist_ok=True)

    # Перемещаем файлы для тренировки
    for idx in train_idx:
        shutil.copy(os.path.join(images_path, all_images[idx]), fold_images_train)
        shutil.copy(os.path.join(labels_path, all_labels[idx]), fold_labels_train)

    # Перемещаем файлы для валидации
    for idx in val_idx:
        shutil.copy(os.path.join(images_path, all_images[idx]), fold_images_val)
        shutil.copy(os.path.join(labels_path, all_labels[idx]), fold_labels_val)

    # Генерация .yaml файла для текущего fold
    fold_data_yaml_path = f"fold_{fold}_data.yaml"
    fold_data = {
        "train": fold_images_train,
        "val": fold_images_val,
        "nc": 2,
        "names": ["background_top", "target_top"]
    }
    with open(fold_data_yaml_path, "w") as yaml_file:
        yaml.dump(fold_data, yaml_file, default_flow_style=False)

    # Загрузка модели YOLO
    print(f"Загрузка предобученной модели: {model_path}")
    model = YOLO(model_path)

    # Обучение модели на текущем fold
    results = model.train(
        data=fold_data_yaml_path,
        epochs=y_epochs,
        imgsz=y_size,
        batch=y_batch,
        conf=0.3,
        iou=0.25,
        lr0=0.005,
        optimizer="Adam",
        name=f"yolov8{y_type}_{y_epochs}_{y_batch}_{y_size}_fold{fold + 1}_2class",
        val=True
    )

    # Сохраняем путь к лучшей модели текущего фолда
    best_model_path = f"runs/detect/yolov8{y_type}_{y_epochs}_{y_batch}_{y_size}_fold{fold + 1}/weights/best.pt"
    print(f"Лучшие веса для Fold {fold + 1} сохранены: {best_model_path}")

    # Сохраняем метрики для каждого класса
    metrics = {
        "precision": results.box.p.tolist(),
        "recall": results.box.r.tolist(),
        "ap50": results.box.ap50.tolist(),
        "ap": results.box.ap.tolist()
    }

    fold_results.append(metrics)

    # Выводим метрики для каждого класса
    class_names = results.names
    for class_id, class_name in class_names.items():
        print(f"Класс {class_id} ({class_name}):")
        print(f"  Precision: {metrics['precision'][class_id]:.4f}")
        print(f"  Recall: {metrics['recall'][class_id]:.4f}")
        print(f"  AP@0.5: {metrics['ap50'][class_id]:.4f}")
        print(f"  AP@[0.5:0.95]: {metrics['ap'][class_id]:.4f}")

    # Удаляем временные папки, чтобы не занимать лишнее место
    shutil.rmtree(fold_images_train)
    shutil.rmtree(fold_images_val)
    shutil.rmtree(fold_labels_train)
    shutil.rmtree(fold_labels_val)

# Усреднение метрик по фолдам для каждого класса
average_metrics = {
    "precision": [],
    "recall": [],
    "ap50": [],
    "ap": []
}

num_classes = len(fold_results[0]["precision"])  # Количество классов

for class_id in range(num_classes):
    avg_precision = sum([fold["precision"][class_id] for fold in fold_results]) / k
    avg_recall = sum([fold["recall"][class_id] for fold in fold_results]) / k
    avg_ap50 = sum([fold["ap50"][class_id] for fold in fold_results]) / k
    avg_ap = sum([fold["ap"][class_id] for fold in fold_results]) / k

    average_metrics["precision"].append(avg_precision)
    average_metrics["recall"].append(avg_recall)
    average_metrics["ap50"].append(avg_ap50)
    average_metrics["ap"].append(avg_ap)

# Вывод усредненных метрик по каждому классу
print("\nУсредненные метрики по всем фолдам:")
for class_id, class_name in class_names.items():
    print(f"Класс {class_id} ({class_name}):")
    print(f"  Precision: {average_metrics['precision'][class_id]:.4f}")
    print(f"  Recall: {average_metrics['recall'][class_id]:.4f}")
    print(f"  AP@0.5: {average_metrics['ap50'][class_id]:.4f}")
    print(f"  AP@[0.5:0.95]: {average_metrics['ap'][class_id]:.4f}")
