import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# CONFIG
image_dir = r"C:\Users\aadi2\OneDrive\Desktop\Temp\test"
model_path = r"C:\Users\aadi2\OneDrive\Desktop\Temp\best_fish_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['fresh', 'medium', 'rotten']

# TRANSFORM (same as during training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# LOAD MODEL
model = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# INFERENCE + DISPLAY
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
    return class_names[pred], img

# LOOP THROUGH IMAGES
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    full_path = os.path.join(image_dir, filename)
    pred_class, img = predict_image(full_path)

    # DISPLAY
    plt.imshow(img)
    plt.title(f"🧠 Predicted: {pred_class}\n📷 File: {filename}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
