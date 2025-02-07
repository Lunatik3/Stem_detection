import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, UnidentifiedImageError
from transformers import LevitFeatureExtractor, LevitModel


def iou_xyxy(boxA, boxB):
    (Ax1, Ay1, Ax2, Ay2) = boxA
    (Bx1, By1, Bx2, By2) = boxB

    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = (Ax2 - Ax1) * (Ay2 - Ay1)
    areaB = (Bx2 - Bx1) * (By2 - By1)
    union_area = areaA + areaB - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def filter_overlapped_bboxes(bboxes, iou_threshold, imgW, imgH):
    xyxy_list = []
    for (cid, x_c, y_c, w, h) in bboxes:
        x_center_abs = x_c * imgW
        y_center_abs = y_c * imgH
        w_abs = w * imgW
        h_abs = h * imgH

        xmin = x_center_abs - w_abs / 2
        ymin = y_center_abs - h_abs / 2
        xmax = x_center_abs + w_abs / 2
        ymax = y_center_abs + h_abs / 2
        xyxy_list.append((cid, xmin, ymin, xmax, ymax))

    # Функция площади
    def box_area(bx):
        cid, x1, y1, x2, y2 = bx
        return (x2 - x1) * (y2 - y1)

    # Сортируем по площади (крупные первые)
    xyxy_list.sort(key=box_area, reverse=True)

    keep_indices = []
    removed_indices = set()

    for i in range(len(xyxy_list)):
        if i in removed_indices:
            continue
        cidA, Ax1, Ay1, Ax2, Ay2 = xyxy_list[i]
        boxA = (Ax1, Ay1, Ax2, Ay2)

        for j in range(i + 1, len(xyxy_list)):
            if j in removed_indices:
                continue
            cidB, Bx1, By1, Bx2, By2 = xyxy_list[j]
            boxB = (Bx1, By1, Bx2, By2)
            iou_val = iou_xyxy(boxA, boxB)
            if iou_val > iou_threshold:
                removed_indices.add(j)

        keep_indices.append(i)

    final_bboxes = []
    for i in keep_indices:
        if i not in removed_indices:
            cid, xmin, ymin, xmax, ymax = xyxy_list[i]
            w_abs = xmax - xmin
            h_abs = ymax - ymin
            x_center_abs = xmin + w_abs / 2
            y_center_abs = ymin + h_abs / 2

            x_c_norm = x_center_abs / imgW
            y_c_norm = y_center_abs / imgH
            w_norm = w_abs / imgW
            h_norm = h_abs / imgH
            final_bboxes.append((cid, x_c_norm, y_c_norm, w_norm, h_norm))

    return final_bboxes


def yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height):
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    w_abs = w * img_width
    h_abs = h * img_height
    xmin = x_center_abs - w_abs / 2
    ymin = y_center_abs - h_abs / 2
    xmax = x_center_abs + w_abs / 2
    ymax = y_center_abs + h_abs / 2
    return xmin, ymin, xmax, ymax


def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    w_abs = xmax - xmin
    h_abs = ymax - ymin
    x_center_abs = xmin + w_abs / 2
    y_center_abs = ymin + h_abs / 2

    x_center = x_center_abs / img_width
    y_center = y_center_abs / img_height
    w = w_abs / img_width
    h = h_abs / img_height
    return x_center, y_center, w, h


def load_bboxes_yolo(txt_path):
    bboxes = []
    if not os.path.isfile(txt_path):
        return bboxes

    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                bboxes.append((class_id, x_c, y_c, w, h))
    return bboxes


def init_levit_extractor(model_name="facebook/levit-256"):
    feature_extractor = LevitFeatureExtractor.from_pretrained(model_name)
    model = LevitModel.from_pretrained(model_name)
    model.eval()
    return feature_extractor, model


def get_bbox_embedding(img_path, bbox, feature_extractor, levit_model, device="cpu"):
    with Image.open(img_path).convert("RGB") as img:
        img_width, img_height = img.size
        _, x_c, y_c, bw, bh = bbox
        xmin, ymin, xmax, ymax = yolo_to_xyxy(x_c, y_c, bw, bh, img_width, img_height)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)

        # Если bbox некорректен/пуст
        if xmax <= xmin or ymax <= ymin:
            return torch.zeros(256)

        cropped = img.crop((xmin, ymin, xmax, ymax))

    # Подготовка входа
    inputs = feature_extractor(images=cropped, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = levit_model(**inputs)
        if outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs.last_hidden_state[:, 0, :]

    emb = emb.squeeze(0).cpu()
    return emb


def cosine_similarity(emb1, emb2):
    if emb1.dim() == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 1:
        emb2 = emb2.unsqueeze(0)
    sim = torch.nn.functional.cosine_similarity(emb1, emb2)
    return sim.item()


def deduplicate_bboxes_levit(
    original_images_folder="dataset/images",
    labels_folder="dataset/labels",
    output_folder="deduplicated_images_levit",
    similarity_threshold=0.6,
    model_name="facebook/levit-256",
    device="cpu"
):
    import csv
    """
    1) Инициализируем LeViT-256 (public или fine-tuned).
    2) Читаем все изображения + YOLO bbox.
    3) Убираем внутрикадровые дубли (IoU).
    4) Межкадрово: (i)->(i+1) сравниваем фичи, удаляем дубликаты.
    5) Сохраняем результаты в output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    feature_extractor, levit_model = init_levit_extractor(model_name)
    levit_model.to(device)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    all_files = glob.glob(os.path.join(original_images_folder, "*"))
    image_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]
    image_paths = sorted(image_paths)

    if len(image_paths) < 2:
        print("[INFO] Недостаточно изображений (>=2) для дедубликации.")
        return

    all_bboxes = {}
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(labels_folder, base_name + ".txt")

        bboxes = load_bboxes_yolo(txt_path)

        with Image.open(img_path) as im:
            w, h = im.size
        bboxes = filter_overlapped_bboxes(bboxes, iou_threshold=0.4, imgW=w, imgH=h)

        all_bboxes[img_path] = bboxes

        total_removed = 0

        for i in range(len(image_paths) - 1):
            img_path_i = image_paths[i]
            img_path_next = image_paths[i+1]

            bboxes_i = all_bboxes[img_path_i]
            bboxes_next = all_bboxes[img_path_next]

            if not bboxes_i or not bboxes_next:
                continue

            features_i = []
            for bbox in bboxes_i:
                emb = get_bbox_embedding(
                    img_path_i, bbox,
                    feature_extractor, levit_model,
                    device=device
                )
                features_i.append(emb)

            features_next = []
            for bbox in bboxes_next:
                emb = get_bbox_embedding(
                    img_path_next, bbox,
                    feature_extractor, levit_model,
                    device=device
                )
                features_next.append(emb)

            duplicates_idx = set()
            for idx1, emb1 in enumerate(features_i):
                for idx2, emb2 in enumerate(features_next):
                    sim = cosine_similarity(emb1, emb2)
                    # Запишем в CSV
                    writer.writerow([
                        os.path.basename(img_path_i), idx1,
                        os.path.basename(img_path_next), idx2,
                        f"{sim:.4f}"
                    ])
                    if sim > similarity_threshold:
                        duplicates_idx.add(idx2)

            new_bboxes_next = []
            for idx2, bbox2 in enumerate(bboxes_next):
                if idx2 not in duplicates_idx:
                    new_bboxes_next.append(bbox2)

            removed_count = len(bboxes_next) - len(new_bboxes_next)
            total_removed += removed_count
            if removed_count > 0:
                print(f"[{os.path.basename(img_path_i)} -> {os.path.basename(img_path_next)}]: "
                      f"Удалено bbox: {removed_count}")
            else:
                print(f"[{os.path.basename(img_path_i)} -> {os.path.basename(img_path_next)}]: "
                      f"Дубликатов не найдено.")

            all_bboxes[img_path_next] = new_bboxes_next

    for img_path in image_paths:
        final_bboxes = all_bboxes[img_path]
        try:
            with Image.open(img_path).convert("RGB") as im:
                draw = ImageDraw.Draw(im)
                w, h = im.size
                for (class_id, x_c, y_c, bw, bh) in final_bboxes:
                    xmin, ymin, xmax, ymax = yolo_to_xyxy(x_c, y_c, bw, bh, w, h)
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=16)

                out_name = os.path.basename(img_path)
                out_path = os.path.join(output_folder, out_name)
                im.save(out_path)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_out_path = os.path.join(output_folder, base_name + ".txt")
            with open(txt_out_path, "w") as f:
                for (class_id, x_c, y_c, bw, bh) in final_bboxes:
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        except UnidentifiedImageError:
            print(f"[WARNING] Не удалось открыть '{img_path}'. Пропуск.")
            continue

    print("\n[INFO] Дедубликация с LeViT-256 завершена!")
    print(f"[INFO] Удалено всего bbox: {total_removed}")
    print(f"[INFO] Результаты (изображения и .txt) в папке: '{output_folder}'")

if __name__ == "__main__":
    deduplicate_bboxes_levit(
        original_images_folder="dataset/images",
        labels_folder="dataset/labels",
        output_folder="deduplicated_images_levit",
        similarity_threshold=0.6,
        model_name="facebook/levit-256",
        device="cpu"
    )
