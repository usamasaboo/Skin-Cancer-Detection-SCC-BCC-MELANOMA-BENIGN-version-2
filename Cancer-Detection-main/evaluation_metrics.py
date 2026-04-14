import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch

def plot_confusion_matrix(y_true, y_pred, classes, save_path='static/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_roc_curve(y_true, y_probs, classes, save_path='static/roc_curve.png'):
    # y_true should be one-hot or index
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        # binary labels for this class
        y_true_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC {class_name} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="bottom right")
    plt.savefig(save_path)
    plt.close()
    return save_path

def generate_evaluation_report(model, dataloader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    cm_path = plot_confusion_matrix(all_labels, all_preds, classes)
    roc_path = plot_roc_curve(all_labels, all_probs, classes)
    
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("Classification Report:\n", report)
    
    return cm_path, roc_path, report
