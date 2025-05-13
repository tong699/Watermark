# streamlit_app.py
import pydicom
import streamlit as st
import numpy as np
import cv2
import pywt
from PIL import Image
import io
import zipfile

from watermark_core import (
    preprocess_medical_image,
    extract_dicom_metadata_text,  
    generate_text_watermark,
    logistic_encrypt,
    logistic_decrypt,
    perform_watermark_embedding,
    perform_watermark_extraction,
    evaluate_watermark_quality,
    calculate_ber,
    calculate_ncc
)

st.set_page_config(page_title="Digital Medical Image Watermarking", layout="wide")
st.title("Digital Medical Image Watermarking (DWT + SVD)")

mode = st.radio("Select Mode", ["Embed Watermark", "Extract Watermark"])

if mode == "Embed Watermark":
    uploaded_image = st.file_uploader("Upload DICOM or Grayscale Image", type=["dcm", "png", "jpg"])
    use_adaptive = st.checkbox("Use Adaptive Alpha", value=True)
    alpha = st.slider("Embedding Strength (α)", min_value=0.01, max_value=0.5, value=0.1)
    use_dcm_metadata = st.checkbox("Use DICOM Metadata as Watermark", value=False)
    watermark_text = ""

    if uploaded_image:
        if uploaded_image.name.endswith(".dcm"):
            dicom_data = pydicom.dcmread(uploaded_image)
            image_data = dicom_data.pixel_array.astype(np.float64)
            image_data = ((image_data - image_data.min()) / (np.ptp(image_data) + 1e-5) * 255).astype(np.uint8)

            if use_dcm_metadata:
                watermark_text = extract_dicom_metadata_text(dicom_data)
            else:
                watermark_text = st.text_area("Watermark Text", "Sample DICOM metadata")
        else:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image_data = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            watermark_text = st.text_area("Watermark Text", "Sample DICOM metadata")

        preprocessed = preprocess_medical_image(image_data)
        wm_image, alpha_used, Uw, Vwh, S, encrypted_wm, ll_shape, orig_wm = perform_watermark_embedding(
            preprocessed,
            watermark_text,
            alpha,
            use_adaptive
        )

        wm_display = np.clip(wm_image, 0, 255).astype(np.uint8)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(preprocessed.astype(np.uint8), caption="Original Image", width=512, use_container_width=True)
        with col2:
            st.image(encrypted_wm.astype(np.uint8), caption="Watermark", width=512, use_container_width=True)
        with col3:
            st.image(wm_display.astype(np.uint8), caption="Watermarked Image", width=512, use_container_width=True)

        psnr, ssim = evaluate_watermark_quality(preprocessed, wm_display)
        st.text(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | Alpha: {alpha_used:.4f}")

        img_bytes = cv2.imencode(".png", wm_display)[1].tobytes()
        st.download_button("Download Watermarked Image", img_bytes, "watermarked_image.png", "image/png")

        memfile = io.BytesIO()
        np.savez(memfile, Uw=Uw, Vwh=Vwh, S=S, alpha=alpha_used, shape=ll_shape, wm=orig_wm)
        st.download_button("Download Extraction Metadata (.npz)", memfile.getvalue(), "metadata.npz", "application/zip")

elif mode == "Extract Watermark":
    uploaded_wm_img = st.file_uploader("Upload Watermarked Image", type=["png", "jpg"])
    uploaded_npz = st.file_uploader("Upload Metadata (.npz)", type=["npz"])

    if uploaded_wm_img and uploaded_npz:
        file_bytes = np.asarray(bytearray(uploaded_wm_img.read()), dtype=np.uint8)
        wm_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        npz_data = np.load(uploaded_npz)

        Uw = npz_data["Uw"]
        Vwh = npz_data["Vwh"]
        S = npz_data["S"]
        alpha = float(npz_data["alpha"])
        shape = tuple(npz_data["shape"])
        original_wm = npz_data["wm"]

        extracted_wm = perform_watermark_extraction(
            wm_image,
            alpha,
            Uw,
            Vwh,
            S,
            original_wm.shape,
            shape
        )

        st.image(extracted_wm.astype(np.uint8), caption="Extracted Watermark", width=512, use_container_width=True)
        ber = calculate_ber(original_wm, extracted_wm)
        ncc = calculate_ncc(original_wm, extracted_wm)
        st.text(f"BER: {ber:.6f} | NCC: {ncc:.6f}")
