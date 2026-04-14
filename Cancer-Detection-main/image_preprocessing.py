import cv2
import os
import numpy as np
from tqdm import tqdm
import albumentations as A

# ==============================
# PREPROCESSING FUNCTIONS
# ==============================
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return inpainted


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess_image(img_source, img_size, augment=False, aug_pipeline=None):
    if isinstance(img_source, str):
        image = cv2.imread(img_source)
    else:
        image = img_source
        
    if image is None:
        return None

    image = cv2.resize(image, (img_size, img_size))
    image = remove_hair(image)
    image = enhance_contrast(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    if augment and aug_pipeline is not None:
        image = aug_pipeline(image=image)["image"]

    image = image / 255.0  # Normalize
    return image


# ==============================
# AUGMENTATION PIPELINE
# ==============================
def get_augmentation(image_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0))
            
        ], p=0.5),

        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=2)
        ], p=0.3),

        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.4
        ),

        A.CoarseDropout(
            max_holes=8,
            max_height=int(image_size * 0.1),
            max_width=int(image_size * 0.1),
            fill_value=0,
            p=0.5
        )

    ])


# ==============================
# MAIN DATASET PROCESSING FUNCTION
# ==============================
def preprocess_dataset(
    input_dirs,
    output_base,
    img_size=224,
    augment_per_image=2
):
    os.makedirs(output_base, exist_ok=True)
    aug_pipeline = get_augmentation(img_size)

    for label, folder in input_dirs.items():
        output_dir = os.path.join(output_base, label)
        os.makedirs(output_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(folder), desc=f"Processing {label}"):
            img_path = os.path.join(folder, img_name)

            clean_img = preprocess_image(img_path, img_size)
            if clean_img is None:
                continue

            base, ext = os.path.splitext(img_name)

            # Save cleaned image
            clean_path = os.path.join(output_dir, img_name)
            cv2.imwrite(clean_path, (clean_img * 255).astype(np.uint8))

            # Save augmented images
            for i in range(augment_per_image):
                aug_img = preprocess_image(
                    img_path,
                    img_size,
                    augment=True,
                    aug_pipeline=aug_pipeline
                )

                aug_name = f"{base}_aug{i}{ext}"
                aug_path = os.path.join(output_dir, aug_name)
                cv2.imwrite(aug_path, (aug_img * 255).astype(np.uint8))

    print("✅ Preprocessing and augmentation completed successfully!")
