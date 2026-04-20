import os
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split


def parse_yolo_segmentation(label_path, img_width, img_height):
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            segment = np.array(parts[1:]).reshape(-1, 2)
            segment[:, 0] *= img_width
            segment[:, 1] *= img_height
            polygons.append({
                'class_id': class_id,
                'segmentation': segment.astype(np.int32).tolist()
            })
    return polygons


def format_yolo_segmentation(polygons, img_width, img_height):
    yolo_lines = []
    for poly_data in polygons:
        class_id = poly_data['class_id']
        segment = np.array(poly_data['segmentation']).astype(np.float32)
        segment[:, 0] /= img_width
        segment[:, 1] /= img_height
        segment_flat = segment.flatten().tolist()
        line = f"{class_id} {' '.join(map(str, segment_flat))}"
        yolo_lines.append(line)
    return "\n".join(yolo_lines)


def augment_dataset(
        original_images_dir,
        original_labels_dir,
        output_base_dir,
        num_augmentations_per_image=5,
        split_ratio=(0.8, 0.2),
        include_original=True
):
    os.makedirs(output_base_dir, exist_ok=True)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        A.Blur(p=0.2, blur_limit=3),
        A.RGBShift(p=0.2, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        A.RandomGamma(p=0.2),
        A.MotionBlur(p=0.2, blur_limit=(3, 5)),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(height=640, width=640, p=0.2)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    image_files = [f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    all_samples = []  # список (image, polygons, filename)

    for img_idx, img_file in enumerate(image_files):
        img_path = os.path.join(original_images_dir, img_file)
        label_path = os.path.join(original_labels_dir, os.path.splitext(img_file)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"⚠️ Нет разметки для {img_file}, пропускаем")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Ошибка загрузки {img_file}, пропускаем")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        original_polygons_data = parse_yolo_segmentation(label_path, w, h)

        # добавляем оригинал в список, если нужно
        if include_original:
            all_samples.append((image, original_polygons_data, img_file))

        # готовим keypoints для аугментации
        keypoints = []
        for poly_data in original_polygons_data:
            for x, y in poly_data['segmentation']:
                keypoints.append((x, y))

        for i in range(num_augmentations_per_image):
            augmented = transform(image=image, keypoints=keypoints)
            aug_image = augmented['image']
            aug_keypoints = augmented['keypoints']

            augmented_polygons_data = []
            kp_idx = 0
            for original_poly_data in original_polygons_data:
                num_points = len(original_poly_data['segmentation'])
                transformed_segment = aug_keypoints[kp_idx: kp_idx + num_points]
                if len(transformed_segment) > 0:
                    augmented_polygons_data.append({
                        'class_id': original_poly_data['class_id'],
                        'segmentation': transformed_segment
                    })
                kp_idx += num_points

            new_img_filename = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
            all_samples.append((aug_image, augmented_polygons_data, new_img_filename))

    # --- делим на train/val ---
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=split_ratio[1],
        random_state=42
    )

    def save_samples(samples, split):
        img_dir = os.path.join(output_base_dir, "images", split)
        lbl_dir = os.path.join(output_base_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for aug_image, polygons, filename in samples:
            new_img_path = os.path.join(img_dir, filename)
            new_label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + '.txt')
            cv2.imwrite(new_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            if polygons:
                h_aug, w_aug = aug_image.shape[:2]
                yolo_annotations = format_yolo_segmentation(polygons, w_aug, h_aug)
                with open(new_label_path, 'w') as f:
                    f.write(yolo_annotations)
            else:
                open(new_label_path, 'w').close()

    save_samples(train_samples, "train")
    save_samples(val_samples, "val")

    print(f"✅ Датасет сохранён в {output_base_dir}")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")


# --- пример использования ---
if __name__ == "__main__":
    augment_dataset(
        original_images_dir='dataset/images/Train',
        original_labels_dir='dataset/labels/Train',
        output_base_dir='augmented_dataset',
        num_augmentations_per_image=5,
        split_ratio=(0.8, 0.2),   # 80% train, 20% val
        include_original=True     # добавляем оригинальные изображения
    )
