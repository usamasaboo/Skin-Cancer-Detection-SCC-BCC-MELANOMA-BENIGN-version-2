from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
# =====================================
# YOLOv9 MODEL CONFIGURATION
# =====================================

def yolo_model_parameter(selected_model):
    model_list = {
        "yolov9_c": {
            "architecture": "yolov9-c.pt",
            "task": "detection",
            "num_classes": 3,  # bcc, scc, melanoma
            "img_size": 640,
            "epochs": 100,
            "batch_size": 16,
            "initial_lr": 0.01,
            "optimizer": "SGD",
            "weight_decay": 5e-4,
            "momentum": 0.937,
            "augmentation": True,
            "mosaic": 1.0,
            "mixup": 0.2,
            "device": "cuda",
            "save_dir": "runs/yolo/yolov9_c",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.6,
            "print_hyper_parameter": True,
            "visualise_augmented_data": False
        },

        "yolov9_e": {
            "architecture": "yolov9-e.pt",
            "task": "detection",
            "num_classes": 3,
            "img_size": 1024,
            "epochs": 150,
            "batch_size": 8,
            "initial_lr": 0.005,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "augmentation": True,
            "mosaic": 1.0,
            "mixup": 0.3,
            "device": "cuda",
            "save_dir": "runs/yolo/yolov9_e",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.6,
            "print_hyper_parameter": True,
            "visualise_augmented_data": False
        }
    }

    return model_list[selected_model]
# =====================================
# FASTER R-CNN MODEL CONFIGURATION
# =====================================

def rcnn_model_parameter(selected_model):
    model_list = {
        "faster_rcnn_resnet50_fpn": {
            "architecture": "fasterrcnn_resnet50_fpn",
            "backbone": "resnet50",
            "pretrained": True,
            "num_classes": 4,  # background + 3 lesions
            "input_image_size": 800,
            "epochs": 30,
            "train_batch_size": 4,
            "validation_batch_size": 2,
            "initial_lr": 0.005,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "lr_scheduler": "StepLR",
            "step_size": 10,
            "gamma": 0.1,
            "device": "cuda",
            "save_model_path": "models/faster_rcnn_resnet50.pth",
            "log_file": "logs/faster_rcnn_resnet50.csv",
            "print_hyper_parameter": True,
            "print_model_summary": False,
            "visualise_augmented_data": False
        },

        "faster_rcnn_resnet101_fpn": {
            "architecture": "fasterrcnn_resnet101_fpn",
            "backbone": "resnet101",
            "pretrained": True,
            "num_classes": 4,
            "input_image_size": 1024,
            "epochs": 40,
            "train_batch_size": 2,
            "validation_batch_size": 2,
            "initial_lr": 0.003,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "lr_scheduler": "CosineAnnealingLR",
            "device": "cuda",
            "save_model_path": "models/faster_rcnn_resnet101.pth",
            "log_file": "logs/faster_rcnn_resnet101.csv",
            "print_hyper_parameter": True,
            "print_model_summary": False,
            "visualise_augmented_data": False
        }
    }

    return model_list[selected_model]
