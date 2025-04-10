import streamlit as st
import cv2
import numpy as np
import pydicom
import tempfile
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from module import (
    preprocess, extract_dicom_metadata, generate_watermark_image,
    embed_watermark_dual, extract_watermark_dual,
    apply_jpeg, apply_noise, apply_crop,
    apply_rotate, apply_scale, apply_shift,
    evaluate_dual
)

st.set_page_config(layout="wide")
st.title("Medical Image Watermarking System (DICOM-based)")

uploaded_file = st.file_uploader("Upload DICOM file", type=["dcm"])

if uploaded_file:
    # Read DICOM
    ds = pydicom.dcmread(uploaded_file)
    dicom_img = ds.pixel_array.astype(np.float32)
    img_norm = cv2.normalize(dicom_img, None, 0, 255, cv2.NORM_MINMAX)
    if len(img_norm.shape) == 2:
        img_color = cv2.cvtColor(img_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_norm.astype(np.uint8)

    st.image(img_color, caption="Original DICOM Image", use_column_width=True)

    if st.button("Generate & Embed Watermark"):
        with st.spinner("Embedding watermark..."):
            metadata = extract_dicom_metadata(ds)
            wm_image = generate_watermark_image(metadata)

            cv2.imwrite("cover.png", img_color)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                cv2.imwrite(f.name, wm_image)
                cover_path = "cover.png"
                wm_path = f.name

            wm_img, Uw, Vw, SH, S_HL, alpha, public_key, wm_ref = embed_watermark_dual(
                cover_path, wm_path, alpha=0.02, k=50)

            cv2.imwrite("watermarked_output.png", wm_img)

        st.success("Watermark embedded.")
        st.image(wm_img, caption="Watermarked Image", use_column_width=True)

        # Quality comparison
        psnr_val = psnr(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(wm_img, cv2.COLOR_BGR2GRAY))
        ssim_val = ssim(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(wm_img, cv2.COLOR_BGR2GRAY))

        st.write(f"**PSNR:** {psnr_val:.2f} dB")
        st.write(f"**SSIM:** {ssim_val:.4f}")

        # Extraction from clean image
        if st.button("Extract from Clean Image"):
            wm_extr, valid = extract_watermark_dual("watermarked_output.png", Uw, Vw, SH, S_HL, alpha, public_key)
            st.image(wm_extr, caption="Extracted Watermark")
            st.write("Authentication Valid:" if valid else "**FAILED**")

        # Attack simulation
        st.markdown("---")
        st.header("Attack Simulation & Robustness Testing")

        attack_type = st.selectbox("Choose Attack", ["JPEG", "Noise", "Crop", "Rotate", "Scale", "Shift"])

        if st.button("Run Attack and Evaluate"):
            if attack_type == "JPEG":
                attacked = apply_jpeg(wm_img, 40)
            elif attack_type == "Noise":
                attacked = apply_noise(wm_img, std=10)
            elif attack_type == "Crop":
                attacked = apply_crop(wm_img.copy(), p=0.2)
            elif attack_type == "Rotate":
                attacked = apply_rotate(wm_img.copy(), angle=10)
            elif attack_type == "Scale":
                attacked = apply_scale(wm_img.copy(), s=0.9)
            elif attack_type == "Shift":
                attacked = apply_shift(wm_img.copy(), dx=10, dy=10)

            st.image(attacked, caption=f"Attacked Image ({attack_type})")
            ncc, ber, valid, extracted = evaluate_dual(attacked, Uw, Vw, SH, S_HL, alpha, public_key, wm_ref)

            st.image(extracted, caption="Extracted Watermark (Post-Attack)")
            st.write(f"**NCC:** {ncc:.4f}")
            st.write(f"**BER:** {ber:.4f}")
            st.write("Authentication Valid:" if valid else "**Authentication Failed**")
