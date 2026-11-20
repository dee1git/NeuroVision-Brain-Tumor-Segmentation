
import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroVision | MRI Segmentation",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "neurovision_final_unet.pt"
THRESHOLD = 0.20
IMAGE_SIZE = 128


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080f20 0%, #101a35 55%, #17132f 100%);
    color: #eef3ff;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f1f5ff !important;
}

p, label, li {
    color: #c4cee4;
}

.hero {
    padding: 28px;
    border: 1px solid #34466e;
    border-radius: 20px;
    background: linear-gradient(110deg, #172c52, #242047);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
}

.hero-sub {
    color: #b9c9ec;
    font-size: 16px;
}

.metric-card {
    background: #172440;
    border: 1px solid #34466e;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    color: #78aaff;
    font-size: 27px;
    font-weight: 800;
}

.metric-label {
    color: #b7c7e5;
    font-size: 13px;
}

.section-card {
    background: #121f39;
    border: 1px solid #2c3e63;
    border-radius: 16px;
    padding: 18px;
}

div[data-testid="stFileUploader"] {
    background: #14233e;
    border: 1px dashed #5575ad;
    border-radius: 14px;
    padding: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #3979e8, #7656d8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 700;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 64-CHANNEL U-NET ARCHITECTURE
# --------------------------------------------------

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet64(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = double_conv(1, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = double_conv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = UNet64().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, device, checkpoint


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):

    image = image.convert("L")

    original = np.array(image)

    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    array = np.array(resized).astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(array)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return original, tensor


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">🧠 NeuroVision</div>
    <div class="hero-sub">
        AI-powered Brain Tumor Segmentation Platform
        <br>
        MRI Analysis • U-Net Architecture • Pixel-level Segmentation
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🧠 NeuroVision")

    st.caption("Model Control Center")

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**Architecture:** 64-channel U-Net")
    st.write("**Input size:** 128 × 128")
    st.write("**Input channels:** 1 (grayscale)")
    st.write("**Parameters:** 31.04 million")
    st.write("**Prediction threshold:** 0.20")

    st.markdown("---")

    st.markdown("### Processing")

    st.write("Grayscale conversion")
    st.write("128 × 128 resizing")
    st.write("Pixel normalization")
    st.write("Sigmoid probability")
    st.write("Binary segmentation")


# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

try:

    model, device, checkpoint = load_model()

    st.success(
        f"Model loaded successfully • Running on {device.type.upper()}"
    )

except Exception as error:

    st.error(f"Could not load the trained model: {error}")

    st.stop()


# --------------------------------------------------
# PROJECT TEST RESULTS
# --------------------------------------------------

st.markdown("## 📊 Model Performance")

st.caption(
    "Reported held-out test-set results from the project evaluation. "
    "These are dataset-level metrics, not measurements of the uploaded image."
)

metrics = checkpoint.get("test_metrics", {})

if metrics:

    cols = st.columns(4)

    labels = [
        ("Dice Score", "dice"),
        ("IoU Score", "iou"),
        ("Precision", "precision"),
        ("Recall", "recall")
    ]

    for col, (label, key) in zip(cols, labels):

        value = metrics.get(key)

        with col:

            if value is not None:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">
                            {value:.4f}
                        </div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.info(f"{label}: Not available")


st.markdown("---")


# --------------------------------------------------
# MRI IMAGE INPUT
# --------------------------------------------------

st.markdown("## 🔬 MRI Segmentation Workspace")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    key="mri_upload"
)


# --------------------------------------------------
# IMAGE PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        original, input_tensor = preprocess_image(image)

        st.markdown("### Prediction Settings")

        st.write(
            f"Segmentation threshold: **{THRESHOLD:.2f}**"
        )

        if st.button(
            "🚀 Run Tumor Segmentation",
            use_container_width=True
        ):

            with st.spinner("Analyzing MRI image..."):

                input_tensor = input_tensor.to(device)

                with torch.no_grad():

                    logits = model(input_tensor)

                    probabilities = torch.sigmoid(logits)

                    prediction = (
                        probabilities >= THRESHOLD
                    ).float()

                probability_array = (
                    probabilities.squeeze().cpu().numpy()
                )

                prediction_array = (
                    prediction.squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                resized_mri = np.array(
                    image.convert("L").resize(
                        (IMAGE_SIZE, IMAGE_SIZE),
                        Image.Resampling.BILINEAR
                    )
                )

                # CREATE RGB OVERLAY

                overlay = np.stack(
                    [resized_mri] * 3,
                    axis=-1
                ).astype(np.uint8)

                overlay[prediction_array == 1] = (
                    overlay[prediction_array == 1] * 0.45
                    + np.array([255, 55, 95]) * 0.55
                ).astype(np.uint8)

                st.success("Segmentation completed.")

                # ------------------------------------------
                # PREDICTION RESULTS
                # ------------------------------------------

                st.markdown("### 🖼️ Prediction Results")

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.image(
                        resized_mri,
                        caption="Input MRI",
                        width="stretch",
                        clamp=True
                    )

                with c2:

                    st.image(
                        prediction_array * 255,
                        caption="Predicted Tumor Mask",
                        width="stretch",
                        clamp=True
                    )

                with c3:

                    st.image(
                        overlay,
                        caption="MRI + Tumor Overlay",
                        width="stretch"
                    )

                # ------------------------------------------
                # PREDICTION SUMMARY
                # ------------------------------------------

                st.markdown("### Prediction Summary")

                tumor_pixels = int(prediction_array.sum())

                total_pixels = prediction_array.size

                tumor_percentage = (
                    tumor_pixels / total_pixels
                ) * 100

                a, b, c = st.columns(3)

                a.metric(
                    "Predicted Tumor Pixels",
                    f"{tumor_pixels:,}"
                )

                b.metric(
                    "Image Pixels",
                    f"{total_pixels:,}"
                )

                c.metric(
                    "Predicted Area",
                    f"{tumor_percentage:.2f}%"
                )

                # ------------------------------------------
                # PREDICTION INTERPRETATION
                # ------------------------------------------

                st.markdown("### Prediction Interpretation")

                st.markdown(
                    f"""
                    The U-Net model has processed the uploaded MRI image
                    and generated a predicted tumour segmentation mask.

                    - **Predicted tumour pixels:** {tumor_pixels:,}
                    - **Total image pixels:** {total_pixels:,}
                    - **Predicted area percentage:** {tumor_percentage:.2f}%

                    The predicted area percentage represents the proportion
                    of image pixels classified as tumour by the model.

                    The red overlay highlights the regions identified by
                    the model in the MRI image.
                    """
                )

                # ------------------------------------------
                # DOWNLOAD PREDICTED MASK
                # ------------------------------------------

                mask_image = Image.fromarray(
                    (prediction_array * 255).astype(np.uint8)
                )

                mask_buffer = io.BytesIO()

                mask_image.save(
                    mask_buffer,
                    format="PNG"
                )

                st.download_button(
                    "⬇️ Download Predicted Mask",
                    data=mask_buffer.getvalue(),
                    file_name="neurovision_predicted_mask.png",
                    mime="image/png"
                )

    except Exception as error:

        st.error(f"Error processing MRI image: {error}")

else:

    st.info("Upload an MRI image above to begin segmentation.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8799bd; padding:15px;">
        <b>NeuroVision</b>
        <br>
        Brain Tumor Segmentation using U-Net
        <br>
        Team 5
    </div>
    """,
    unsafe_allow_html=True
)
