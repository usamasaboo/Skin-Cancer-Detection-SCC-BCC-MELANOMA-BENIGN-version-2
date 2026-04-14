import torch
import timm
from PIL import Image
from torchvision import transforms
import os

MODEL_PATH = "saved_models/efficientnet_b0_skin_cancer_4class.pth"
CLASSES = ['bcc', 'benign', 'melanoma', 'scc']

def test_on_downloads():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = [
        'bcc1.webp', 'bcc2.webp', 'bcc3.webp',
        'scc1.jpg', 'scc2.jpg', 'scc3.webp',
        'mel1.webp', 'mel2.webp', 'mel3.webp'
    ]

    print(f"--- Testing Old Model on {len(files)} Downloaded Samples ---")
    with open("results.txt", "w") as out:
        out.write(f"--- Testing Old Model on {len(files)} Downloaded Samples ---\n")
        for f in files:
            path = os.path.join(r"c:\Users\Usama\Downloads", f)
            if not os.path.exists(path):
                continue
            
            try:
                image = Image.open(path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, idx = torch.max(probs, 0)
                    pred = CLASSES[idx]
                
                line = f"File: {f:12} | Pred: {pred:10} | Conf: {conf.item()*100:5.2f}%\n"
                print(line.strip())
                out.write(line)
            except Exception as e:
                out.write(f"Error testing {f}: {e}\n")

if __name__ == "__main__":
    test_on_downloads()
