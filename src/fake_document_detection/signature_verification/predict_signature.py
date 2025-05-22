import os
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from siamese_model import SiameseNetwork

# Constants
MODEL_PATH = "siamese_signature_model.pth"

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

def load_image(source):
    """
    Loads an image from a file path or Streamlit file uploader and applies required transform.
    """
    try:
        if hasattr(source, "read"):  # Streamlit file uploader
            image = Image.open(source).convert("L")
        else:  # File path
            image = Image.open(str(source)).convert("L")

        return transform(image).unsqueeze(0)  # shape: (1, 1, 105, 105)
    except UnidentifiedImageError:
        print(f"❌ Could not read image: {source}")
        return None

def verify_signature_pair(img1, img2, model_path=MODEL_PATH):
    """
    Takes two transformed images and returns similarity score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        similarity = 1 - distance.item()

    return similarity

    
def run_batch_signature_verification(data_dir=DATA_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Using device: {device}")

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    folders = [f for f in os.listdir(data_dir) if f.startswith("full_org_")]

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        files = os.listdir(folder_path)

        # Find one genuine image
        genuine_files = [f for f in files if f.startswith("org_")]
        if not genuine_files:
            print(f"⚠️ No genuine image in {folder}")
            continue
        genuine_path = os.path.join(folder_path, genuine_files[0])
        img1 = load_image(genuine_path)
        if img1 is None:
            continue
        img1 = img1.to(device)

        print(f"\n📁 Folder: {folder}")
        forged_files = [f for f in files if f.startswith("for_")]
        if not forged_files:
            print("⚠️ No forged images found.")
            continue

        for f in forged_files:
            forged_path = os.path.join(folder_path, f)
            img2 = load_image(forged_path)
            if img2 is None:
                continue
            img2 = img2.to(device)

            try:
                with torch.no_grad():
                    output1, output2 = model(img1, img2)
                    distance = torch.nn.functional.pairwise_distance(output1, output2)
                    similarity = 1 - distance.item()  # closer to 1 = more similar
                    print(f"🖊️ {f} → Similarity Score: {similarity:.4f}")
            except RuntimeError as e:
                print(f"❌ RuntimeError comparing {f}: {e}")

if __name__ == "__main__":
    run_batch_signature_verification()
