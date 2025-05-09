import streamlit as st
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import os
from watermarking import (
    preprocess_medical_image, extract_dicom_metadata_text, generate_text_watermark,
    perform_watermark_embedding, perform_watermark_extraction, apply_attack,
    evaluate_watermark_quality, calculate_ber, calculate_ncc
)

# Streamlit page configuration
st.set_page_config(page_title="Medical Image Watermarking Dashboard", layout="wide")

# Title and description
st.title("📷 Medical Image Watermarking Dashboard")
st.markdown("""
This dashboard allows you to watermark DICOM medical images with patient metadata using IWT and SVD.
Upload a DICOM file, configure parameters, apply attacks, and visualize results.
""")

# Sidebar for inputs
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload DICOM File (.dcm)", type=["dcm"])

# Watermarking parameters
watermark_size = st.sidebar.slider("Watermark Size (pixels)", 64, 256, 128, step=8)
font_size = st.sidebar.slider("Watermark Font Size", 8, 20, 10)
embedding_strength = st.sidebar.slider("Embedding Strength (α)", 0.01, 1.0, 0.1, step=0.01)
use_adaptive_alpha = st.sidebar.checkbox("Use Adaptive Alpha", value=False)
lambda_strength = st.sidebar.slider("Lambda Strength (if adaptive)", 0.01, 0.1, 0.05, step=0.01)

# Attack selection
attack_types = st.sidebar.multiselect(
    "Select Attacks for Robustness Testing",
    ["no_attack", "salt_pepper", "gaussian_noise", "jpeg_compression", "rotation", "scaling", "cropping"],
    default=["no_attack", "salt_pepper", "jpeg_compression"]
)
attack_params = {
    "no_attack": {},
    "salt_pepper": {"amount": st.sidebar.slider("Salt & Pepper Amount", 0.001, 0.1, 0.01, step=0.001)},
    "gaussian_noise": {"mean": 0, "std": st.sidebar.slider("Gaussian Noise Std Dev", 5, 50, 15)},
    "jpeg_compression": {"quality": st.sidebar.slider("JPEG Quality", 10, 100, 90)},
    "rotation": {"angle": st.sidebar.slider("Rotation Angle (degrees)", 1, 45, 15)},
    "scaling": {"scale": st.sidebar.slider("Scaling Factor", 0.1, 1.0, 0.7, step=0.1)},
    "cropping": {"percent": st.sidebar.slider("Cropping Percentage", 0.05, 0.5, 0.1, step=0.05)}
}

# Process button
process_button = st.sidebar.button("Process Watermarking")

# Main content
if uploaded_file and process_button:
    with st.spinner("Processing..."):
        # Read DICOM file
        try:
            dicom_dataset = pydicom.dcmread(uploaded_file)
            host_pixel_array = dicom_dataset.pixel_array.astype(np.float64)
            # Normalize to [0, 255]
            min_val, max_val = host_pixel_array.min(), host_pixel_array.max()
            if max_val == min_val:
                host_normalized_uint8 = np.zeros_like(host_pixel_array, dtype=np.uint8)
            else:
                host_normalized = (host_pixel_array - min_val) / (max_val - min_val) * 255.0
                host_normalized_uint8 = host_normalized.astype(np.uint8)
        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
            st.stop()

        # Preprocess image
        Y_host_preprocessed = preprocess_medical_image(host_normalized_uint8)

        # Generate watermark
        dicom_metadata_str = extract_dicom_metadata_text(dicom_dataset)
        original_text_watermark_img = generate_text_watermark(
            dicom_metadata_str, size=(watermark_size, watermark_size), font_size=font_size
        )

        # Perform watermark embedding
        (Y_watermarked_float, alpha_final, Uw, Vwh, S,
         W_encrypted_embedded, original_shape) = perform_watermark_embedding(
            Y_host_preprocessed,
            original_text_watermark_img,
            embedding_strength=embedding_strength,
            use_adaptive_alpha=use_adaptive_alpha,
            lambda_strength=lambda_strength
        )
        Y_watermarked_uint8 = np.clip(Y_watermarked_float, 0, 255).astype(np.uint8)

        # Evaluate watermark quality
        psnr_val, ssim_val = evaluate_watermark_quality(Y_host_preprocessed, Y_watermarked_uint8)

        # Extract watermark (no attack)
        decrypted_watermark_no_attack = perform_watermark_extraction(
            Y_watermarked_float,
            alpha_final, Uw, Vwh, S,
            original_text_watermark_img.shape,
            original_shape
        )
        ber_no_attack = calculate_ber(original_text_watermark_img, decrypted_watermark_no_attack)
        ncc_no_attack = calculate_ncc(original_text_watermark_img, decrypted_watermark_no_attack)

        # Display main results
        st.header("Watermarking Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(Y_host_preprocessed, caption="Preprocessed Host Image", use_column_width=True)
        with col2:
            st.image(W_encrypted_embedded, caption="Encrypted Watermark", use_column_width=True)
        with col3:
            st.image(Y_watermarked_uint8, caption=f"Watermarked Image\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}", use_column_width=True)
        with col4:
            st.image(decrypted_watermark_no_attack, caption=f"Extracted Watermark (No Attack)\nBER: {ber_no_attack:.4f}, NCC: {ncc_no_attack:.4f}", use_column_width=True)

        # Metrics plot
        st.subheader("Quality Metrics")
        metrics_fig = go.Figure()
        metrics_fig.add_trace(go.Bar(
            x=["PSNR (dB)", "SSIM", "BER (No Attack)", "NCC (No Attack)"],
            y=[psnr_val, ssim_val, ber_no_attack, ncc_no_attack],
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        ))
        metrics_fig.update_layout(
            title="Watermarking Quality Metrics",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )
        st.plotly_chart(metrics_fig, use_container_width=True)

        # Robustness analysis
        st.header("Robustness Analysis")
        attack_results = []
        for attack_name in attack_types:
            attacked_image_float = apply_attack(Y_watermarked_float.copy(), attack_name, **attack_params[attack_name])
            decrypted_watermark_attacked = perform_watermark_extraction(
                attacked_image_float,
                alpha_final, Uw, Vwh, S,
                original_text_watermark_img.shape,
                original_shape
            )
            ber_attacked = calculate_ber(original_text_watermark_img, decrypted_watermark_attacked)
            ncc_attacked = calculate_ncc(original_text_watermark_img, decrypted_watermark_attacked)
            attack_results.append({
                "Attack": attack_name,
                "Attacked Image": np.clip(attacked_image_float, 0, 255).astype(np.uint8),
                "Extracted Watermark": decrypted_watermark_attacked,
                "BER": ber_attacked,
                "NCC": ncc_attacked
            })

        # Display attack results
        for result in attack_results:
            st.subheader(f"Attack: {result['Attack']}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(result["Attacked Image"], caption=f"Attacked Image: {result['Attack']}", use_column_width=True)
            with col2:
                st.image(result["Extracted Watermark"], caption=f"Extracted Watermark\nBER: {result['BER']:.4f}, NCC: {result['NCC']:.4f}", use_column_width=True)

        # Robustness plot
        st.subheader("Robustness Metrics")
        robustness_fig = go.Figure()
        attack_names = [r["Attack"] for r in attack_results]
        ber_values = [r["BER"] for r in attack_results]
        ncc_values = [r["NCC"] for r in attack_results]
        robustness_fig.add_trace(go.Bar(
            x=attack_names,
            y=ber_values,
            name="BER",
            marker_color="#2ca02c"
        ))
        robustness_fig.add_trace(go.Bar(
            x=attack_names,
            y=ncc_values,
            name="NCC",
            marker_color="#d62728"
        ))
        robustness_fig.update_layout(
            title="Robustness Against Attacks",
            yaxis_title="Value",
            barmode="group",
            height=400
        )
        st.plotly_chart(robustness_fig, use_container_width=True)

else:
    st.info("Please upload a DICOM file and click 'Process Watermarking' to start.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Medical Image Watermarking using IWT and SVD")
