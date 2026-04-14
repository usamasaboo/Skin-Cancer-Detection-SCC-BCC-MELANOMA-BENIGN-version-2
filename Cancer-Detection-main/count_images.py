import os

base_dir = r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train"
classes = ['bcc', 'melanoma', 'scc']

for cls in classes:
    path = os.path.join(base_dir, cls)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Class {cls}: {count} images")
    else:
        print(f"Class {cls}: Path not found")
