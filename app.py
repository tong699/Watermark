import streamlit as st
import numpy as np
import cv2
import pydicom
import tempfile
from module import (
    preprocess,
    generate_watermark_image,
    embed_watermark_dual,
    extract_watermark_dual,
    psnr, ssim,
    apply_jpeg, apply_noise, apply_crop,
    apply_rotate, apply_scale, apply_shift,
    evaluate_dual,
)

# === App Setup ===
st.set_page_config("Medical Watermarking", layout="wide")
st.title("🩺 Medical Image Watermarking with RSA Authentication")

# === Upload Section ===
dcm_file = st.file_uploader("Upload a DICOM (.dcm) file", type=["dcm"])
text_input = st.text_area("Enter Watermark Text", placeholder="e.g. Patient Name, ID, Hospital Info")

if dcm_file and text_input.strip():
    # Load and save temporary DICOM
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_dcm:
        tmp_dcm.write(dcm_file.read())
        dicom_path = tmp_dcm.name

    ds = pydicom.dcmread(dicom_path)
    try:
        img = ds.pixel_array
        img = preprocess(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        st.error(f"❌ Failed to load DICOM image.\n\n**Error:** {e}")
        st.stop()

    st.image(img, caption="Preprocessed Medical Image", use_column_width=True)

    # === Watermark Generation ===
    wm_img = generate_watermark_image(text_input)
    st.image(wm_img, caption="Generated Watermark Image", clamp=True, width=256)

    if st.button("🔐 Embed Watermark"):
        # Save watermark
        cv2.imwrite("temp_watermark.png", wm_img)

        # Run embedding
        watermarked, Uw, Vw, S_LL, S_HL, alpha, pub = embed_watermark_dual(
            cover_path=dicom_path,
            wm_path="temp_watermark.png",
            alpha=0.02,
            k=50
        )
        cv2.imwrite("watermarked_output.png", watermarked)
        st.image(watermarked, caption="✅ Watermarked Image", use_column_width=True)

        # === Quality Evaluation ===
        psnr_val = psnr(img, cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY))
        ssim_val = ssim(img, cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY))
        st.markdown(f"**PSNR:** {psnr_val:.2f} dB")
        st.markdown(f"**SSIM:** {ssim_val:.4f}")

        # === Extraction
        extracted_wm, valid = extract_watermark_dual(
            "watermarked_output.png", Uw, Vw, S_LL, S_HL, alpha, pub
        )
        st.image(extracted_wm, caption=f"Extracted Watermark (Auth: {'✅' if valid else '❌'})", clamp=True)

        # === Robustness Tests
        st.subheader("🔁 Optional: Attack Testing")
        col1, col2 = st.columns(2)
        with col1:
            jpeg = apply_jpeg(watermarked)
            ncc, ber, auth, wm = evaluate_dual(jpeg, Uw, Vw, S_LL, S_HL, alpha, pub, wm_img)
            st.image(wm, caption=f"JPEG Attack | NCC: {ncc:.3f} | BER: {ber:.3f} | Auth: {'✓' if auth else '✗'}")

            noise = apply_noise(watermarked)
            ncc, ber, auth, wm = evaluate_dual(noise, Uw, Vw, S_LL, S_HL, alpha, pub, wm_img)
            st.image(wm, caption=f"Noise Attack | NCC: {ncc:.3f} | BER: {ber:.3f} | Auth: {'✓' if auth else '✗'}")

        with col2:
            crop = apply_crop(watermarked)
            ncc, ber, auth, wm = evaluate_dual(crop, Uw, Vw, S_LL, S_HL, alpha, pub, wm_img)
            st.image(wm, caption=f"Crop Attack | NCC: {ncc:.3f} | BER: {ber:.3f} | Auth: {'✓' if auth else '✗'}")

            rotate = apply_rotate(watermarked)
            ncc, ber, auth, wm = evaluate_dual(rotate, Uw, Vw, S_LL, S_HL, alpha, pub, wm_img)
            st.image(wm, caption=f"Rotate Attack | NCC: {ncc:.3f} | BER: {ber:.3f} | Auth: {'✓' if auth else '✗'}")

else:
    st.info("📥 Please upload a DICOM file and enter watermark text to begin.")
