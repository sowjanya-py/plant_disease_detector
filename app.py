import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError
from inference_sdk import InferenceHTTPClient
import os
import tempfile

# Disease information dictionary
DISEASE_INFO = {
    "0": {
        "name": "Leaf Spot",
        "description": "Dark circular or irregular spots on leaves caused by fungal infection. Remove affected leaves and apply appropriate fungicides."
    },
    "1": {
        "name": "Powdery Mildew",
        "description": "White powdery fungal growth on leaves and stems. Use sulfur-based fungicides or neem oil spray for control."
    },
    "2": {
        "name": "Rust",
        "description": "Orange to reddish pustules on leaf undersides caused by rust fungus. Improve air circulation and apply fungicides."
    },
    "3": {
        "name": "Blight",
        "description": "Rapid browning and death of leaf tissue, usually due to bacterial or fungal infection. Remove infected plants and use copper-based sprays."
    }
}

# Initialize Roboflow client securely
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["general"]["api_key"]
)

# Page setup
st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")

# Custom CSS for leafy theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255,255,255,0.3), rgba(255,255,255,0.3)),
            url("https://i.pinimg.com/736x/ab/bc/1d/abbc1d5062585092c10bc928f099fa8e.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .title {
        color: #2c6b2f;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.2em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: #3a763a;
        font-size: 1.3rem;
        margin-bottom: 1.5em;
        font-weight: 500;
    }
    .disease-info {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(44,107,47,0.25);
        margin-bottom: 1em;
        color: #004d00;
    }
    .footer {
        font-size: 0.9rem;
        color: #3a663a;
        margin-top: 3rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="title">🌿 Plant Disease Detection App</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Upload one or more plant leaf images to detect diseases.</h2>', unsafe_allow_html=True)

# File upload
uploaded_files = st.file_uploader("📄 Upload one or more leaf images (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"### Processing: {uploaded_file.name}")
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name

            with st.spinner("🔍 Detecting diseases..."):
                result = client.run_workflow(
                    workspace_name="oreo-kfw1b",
                    workflow_id="custom-workflow",
                    images={"image": temp_image_path},
                    use_cache=True
                )

            if result and "predictions" in result[0] and "predictions" in result[0]["predictions"]:
                detections = result[0]["predictions"]["predictions"]
                if detections:
                    draw = ImageDraw.Draw(image)
                    detected_diseases = set()

                    for pred in detections:
                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                        class_id = str(pred.get("class_id", ""))
                        confidence = pred.get("confidence", 0)
                        disease_name = DISEASE_INFO.get(class_id, {}).get("name", "Unknown")
                        disease_desc = DISEASE_INFO.get(class_id, {}).get("description", "No description available.")
                        detected_diseases.add((disease_name, disease_desc, confidence))

                        label = f"{disease_name} ({confidence*100:.1f}%)"
                        draw.rectangle(
                            [(x - w/2, y - h/2), (x + w/2, y + h/2)],
                            outline="#8B0000",
                            width=4
                        )
                        draw.text((x - w/2, y - h/2 - 20), label, fill="#8B0000")

                    st.image(image, caption="🪧 Detected Disease(s)", use_column_width=True)
                    st.markdown("### Disease Information & Care Advice:")

                    for name, desc, conf in detected_diseases:
                        st.markdown(f'<div class="disease-info"><b>{name}</b> - Confidence: {conf*100:.1f}%<br>{desc}</div>', unsafe_allow_html=True)
                else:
                    st.success("🌿 The leaf appears to be healthy. No disease detected.")
                    st.image(image, caption="🌱 Healthy Leaf", use_column_width=True)
            else:
                st.warning("⚠️ Could not get a valid response. Please try again.")

            os.remove(temp_image_path)

        except UnidentifiedImageError:
            st.error("❌ The uploaded file is not a valid image.")
        except Exception as e:
            st.error(f"❌ Error during inference: {e}")
else:
    st.info("Please upload one or more plant leaf images to get started.")

st.markdown('<div class="footer">Developed by Sowjanya Surada | Powered by Roboflow & Streamlit 🌱</div>', unsafe_allow_html=True)
