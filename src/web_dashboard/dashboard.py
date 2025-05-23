import os
import sys

import streamlit as st
import pandas as pd
from PIL import Image
import torch
from pymongo import MongoClient
from torchvision import transforms

# ========== Project Root Setup ==========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

# ========== Internal Module Imports ==========
from src.rag_chatbot.rag_chatbot import chat_with_bot
from src.fake_document_detection.signature_verification.verify_signature import verify_signature_pair
from src.fake_document_detection.signature_verification.siamese_model import SiameseNetwork

# ========== Device Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Signature Verification Model ==========
SIGNATURE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "siamese_signature_model.pth")

signature_model = None
if os.path.exists(SIGNATURE_MODEL_PATH):
    signature_model = SiameseNetwork()
    print("SIGNATURE_MODEL_PATH:", SIGNATURE_MODEL_PATH)
    print("Is file exist?", os.path.exists(SIGNATURE_MODEL_PATH))
    checkpoint = torch.load(SIGNATURE_MODEL_PATH, map_location=device, weights_only=True)
    print(checkpoint.keys())

    signature_model.load_state_dict(checkpoint)
    signature_model.to(device)
    signature_model.eval()
else:
    st.warning(f"⚠️ Signature model file missing at: {SIGNATURE_MODEL_PATH}")

# ========== Helper function ==========
def safe_path(*args):
    return os.path.normpath(os.path.join(PROJECT_ROOT, *args))

def load_image(source):
    """Load and transform image from Streamlit uploader"""
    try:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((105, 105)),
            transforms.ToTensor()
        ])
        image = Image.open(source).convert("L")
        tensor_img = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor_img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def main():
    st.set_page_config(page_title="Immigration AI Dashboard", layout="wide")
    st.title("🚨 Illegal Immigration Monitoring Dashboard")
    st.markdown("---")

    tabs = st.tabs([
        "📊 Alerts Dashboard",
        "🗺️ Heatmaps",
        "📑 Reports",
        "🎥 CCTV Alerts",
        "🤖 RAG Chatbot (Beta - Improving Accuracy)",
        "🧾 Fake Document Check",
        "✍️ Signature Verification",
        "🛰️ Satellite Intrusion"
    ])

    # --- 0. Alerts Dashboard ---
    with tabs[0]:
        st.header("📊 Alerts Dashboard")

        MONGO_URI = "mongodb://localhost:27017/"
        client = MongoClient(MONGO_URI)
        db = client["immigration_monitoring"]
        alerts_collection = db["alerts"]

        alerts_data = list(alerts_collection.find({}, {"_id": 0}))
        if alerts_data:
            df_alerts = pd.DataFrame(alerts_data)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Alerts by Source")
                st.bar_chart(df_alerts["source"].value_counts())

            with col2:
                st.subheader("Alerts by Severity")
                st.bar_chart(df_alerts["severity"].value_counts())

            st.subheader("📋 Alert Details")
            st.dataframe(df_alerts)
        else:
            st.warning("No alerts found in MongoDB.")

    # --- 1. Heatmaps ---
    with tabs[1]:
        st.header("🗺️ Geospatial Heatmaps")
        map_paths = [
            safe_path("reports", "india_heatmap.html"),
            safe_path("reports", "darkweb_forged_heatmap.html"),
            safe_path("reports", "illegal_hotspot_heatmap.html")
        ]
        for path in map_paths:
            st.markdown(f"**Map:** `{os.path.basename(path)}`")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=600)
            except FileNotFoundError:
                st.warning(f"Map not found: {path}")
            st.markdown("---")

    # --- 2. Reports ---
    with tabs[2]:
        st.header("📑 Statistical Reports")
        report_category = st.selectbox("Select Report Category", ["Reddit", "Dark Web", "Twitter", "Other"])

        def show_images_in_folder(folder):
            import glob
            for img_path in glob.glob(folder + "/*.png"):
                st.image(img_path, use_column_width=True)

        if report_category == "Reddit":
            folder = safe_path("reports", "reddit")
            show_images_in_folder(folder)
        elif report_category == "Dark Web":
            folder = safe_path("reports", "darkweb")
            show_images_in_folder(folder)
        elif report_category == "Twitter":
            folder = safe_path("reports", "twitter")
            show_images_in_folder(folder)
        else:
            st.info("No reports available for this category yet.")

    # --- 3. CCTV Alerts ---
    with tabs[3]:
        st.header("🎥 CCTV Intrusion Alerts")
        video_path = safe_path("data", "final", "final_alert_overlay.mp4")
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.error("Video file not found.")

    # --- 4. RAG Chatbot ---
    with tabs[4]:
        st.header("🤖 RAG Chatbot (Local LLM)")
        st.warning("🚧 Beta version - improvements ongoing.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question:")
            submitted = st.form_submit_button("Send")

            if submitted and user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("Generating response..."):
                    response = chat_with_bot(user_input)
                st.session_state.chat_history.append({"role": "bot", "content": response})

        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**Bot:** {chat['content']}")

    # --- 5. Fake Document Check ---
    with tabs[5]:
        st.header("🧾 Fake Document Check")
        st.info("Feature coming soon.")

    # --- 6. Signature Verification ---
    with tabs[6]:
        st.header("✍️ Signature Verification (Siamese Network)")
        st.warning("⚠️ Signature verification feature is under improvement. Similarity scores might not be fully accurate yet. Thank you for your patience!")

        org_image = st.file_uploader("Upload Original Signature", type=["jpg", "jpeg", "png"], key="org")
        forged_image = st.file_uploader("Upload Forged Signature", type=["jpg", "jpeg", "png"], key="forged")

        if org_image and forged_image:
            col1, col2 = st.columns(2)
            with col1:
                st.image(org_image, caption="Original Signature", use_column_width=True)
            with col2:
                st.image(forged_image, caption="Forged Signature", use_column_width=True)

            if st.button("🔍 Compare Signatures"):
                if signature_model is None:
                    st.error("Signature verification model not loaded.")
                else:
                    img1 = load_image(org_image)
                    img2 = load_image(forged_image)

                    if img1 is not None and img2 is not None:
                        similarity = verify_signature_pair(img1, img2, signature_model)
                        st.success(f"Similarity Score: {similarity:.4f}")

                        if similarity > 0.8:
                            st.info("🟢 Likely the same signer.")
                        else:
                            st.warning("🔴 Possibly forged signature.")
                    else:
                        st.error("Error loading images for verification.")

    # --- 7. Satellite Intrusion Detection ---
    with tabs[7]:
        st.header("🛰️ Satellite Image Border Intrusion Detection")
        st.info("Feature coming soon.")

if __name__ == "__main__":
    main()
