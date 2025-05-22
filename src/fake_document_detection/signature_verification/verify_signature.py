import os
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from .siamese_model import SiameseNetwork  # make sure this relative import works in your project structure

# Constants
MODEL_PATH = "siamese_signature_model.pth"
DATA_DIR = "data"
# Define transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

def load_image(source):
    try:
        if hasattr(source, "read"):
            image = Image.open(source).convert("L")
        else:
            image = Image.open(str(source)).convert("L")

        tensor_img = transform(image).unsqueeze(0)
        print("Image loaded and transformed:", tensor_img.shape)
        return tensor_img
    except UnidentifiedImageError:
        print(f"❌ Could not read image: {source}")
        return None


def verify_signature_pair(img1, img2, model):
    """
    Takes two transformed images and a loaded Siamese model,
    returns similarity score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        similarity = 1 - distance.item()

    return similarity
