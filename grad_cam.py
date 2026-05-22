"""
================================================================================
MODEL INTERPRETABILITY & EXPLAINABLE AI (XAI)

OVERVIEW:
This module implements Gradient-weighted Class Activation Mapping (Grad-CAM) for a custom-headed ResNet18 convolutional neural network. It functions as a diagnostic layer to look inside the network's final convolutional layer block (layer4[-1]) to visually verify the network's attention mechanisms.

Grad-CAM acts as an X-ray for our AI's brain. It looks at the absolute last 
convolutional layer of our loaded ResNet18 model to figure out exactly which 
pixels the AI focused on before making its final traffic sign prediction.

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image

class GradCAMEngine:
    def __init__(self, model_path, num_classes=85):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(self, image_path, output_path="gradcam_output.png"):
        original_img = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(original_img).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(input_tensor)
        target_class = torch.argmax(outputs, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = outputs[0, target_class]
        class_loss.backward()
        
        # Compute weights and accumulate activations
        weights = torch.mean(self.gradients, dim=[2, 3])[0]
        spatial_heatmap = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(self.device)
        
        for i, w in enumerate(weights):
            spatial_heatmap += w * self.activations[0, i, :, :]
            
        # FIXED: Moved outside the for loop block
        heatmap = torch.clamp(spatial_heatmap, min=0).cpu().numpy()
        
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
            
        img_cv = cv2.imread(image_path)
        img_cv = cv2.resize(img_cv, (224, 224))
        
        resized_heatmap = cv2.resize(heatmap, (224, 224))
        resized_heatmap = np.uint8(255 * resized_heatmap)
        color_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
        
        overlay_img = cv2.addWeighted(img_cv, 0.6, color_heatmap, 0.4, 0)
        cv2.imwrite(output_path, overlay_img)
        print(f" Grad-CAM heatmap generated successfully as '{output_path}'")
        cv2.imshow("Grad-CAM Heatmap", overlay_img)
        print(" Press any key on the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return output_path
    
# ==========================================
# 🚀 LOCAL TEST RUNNER EXECUTABLE
# ==========================================
if __name__ == "__main__":
    # Placeholder 1: Ensure these relative paths match your local workspace!
    MODEL_PATH = r"D:\Traffic Sign Classification using CNN\Models\native_indian_baseline_e4.pth"
    #SAMPLE_IMAGE = r"D:\Traffic Sign Classification using CNN\sample_sign_02.png"
    #SAMPLE_IMAGE = r"D:\Traffic Sign Classification using CNN\sample_sign_01.png" 
    SAMPLE_IMAGE = r"D:\Traffic Sign Classification using CNN\sample_sign_03.png" 
    
    try:
        # Initialize the engine with your trained baseline weights
        cam = GradCAMEngine(model_path=MODEL_PATH)
        
        # Execute the forward/backward tracking loop on your test image
        cam.generate_heatmap(image_path=SAMPLE_IMAGE)
        
    except Exception as e:
        print(f" Run Alert: Make sure '{SAMPLE_IMAGE}' is dropped in your project folder. Log: {e}")