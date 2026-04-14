import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activations(module, input, output):
            self.activations = output

        self.handlers.append(self.target_layer.register_forward_hook(save_activations))
        self.handlers.append(self.target_layer.register_full_backward_hook(save_gradients))

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()

def apply_heatmap(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay context: BGR to RGB if needed, but here we assume BGR for cv2
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        # Check if original is 0-1 or 0-255
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
            
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return superimposed_img
