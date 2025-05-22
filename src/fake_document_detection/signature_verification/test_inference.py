import torch
import torchvision.transforms as transforms
from PIL import Image
from siamese_model import SiameseNetwork
import os

# Load model
model = SiameseNetwork()
model.load_state_dict(torch.load("siamese_signature_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image loader
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0)  # Add batch dimension

# Load images
img1_path = "data/full_org_001/org_001_08.png" # genuine
img2_path = "data/full_org_001/for_001_08.png" # forged

img1 = load_image(img1_path)
img2 = load_image(img2_path)

# Predict
with torch.no_grad():
    output1, output2 = model(img1, img2)
    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
    print("Euclidean Distance:", euclidean_distance.item())

# Simple Threshold
if euclidean_distance.item() < 1.0:
    print("Signature MATCH")
else:
    print("Signature FORGERY")
