import streamlit as st
import pydicom
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

# Import your watermarking methods
from watermark_module import (
    preprocess_medical_image,
    extract_dicom_metadata_text,
    generate_text_watermark,
    logistic_encrypt,
    apply_dwt, hessenberg, apply_svd,
    blend_singular_values, reconstruct_matrix,
    reconstruct_y_channel, evaluate_watermark_quality,
    decompose_watermarked_image, apply_svd_extraction,
    extract_encrypted_watermark_singular,
    reconstruct_encrypted_watermark,
    logistic_decrypt, calculate_ber, calculate_ncc,
    apply_attack
)


st.set_page_config(layout="wide")
st.title("🔐 Medical Image Watermarking with Robustness Testing")

uploaded_file = st.file_uploader("Upload DICOM file (.dcm)", type=["dcm"])

if uploaded_file:
    # Read DICOM
    ds = pydicom.dcmread(uploaded_file)
    cover_image = ds.pixel_array.astype(np.float64)
    normalized = ((cover_image - cover_image.min()) / (cover_image.max() - cover_image.min()) * 255).astype(np.uint8)

    # Step 1: Preprocess
    Y = preprocess_medical_image(normalized)

    # Step 2: Generate watermark
    text_input = extract_dicom_metadata_text(ds)
    watermark_img = generate_text_watermark(text_input)
    encrypted_watermark = logistic_encrypt(watermark_img)

    U_wm, S_wm, Vt_wm = np.linalg.svd(encrypted_watermark.astype(np.float64), full_matrices=False)
    S_wm_matrix = np.diag(S_wm)

    alpha = 0.18  # Or compute_adaptive_alpha(Y, lam=0.18)
    LL, HL, LH, HH = apply_dwt(Y)
    H, P = hessenberg(LL, calc_q=True)
    U_H, S_H, Vt_H = apply_svd(H.astype(np.float64))
    S_H = np.diag(S_H)

    S_H_blended = blend_singular_values(S_H, S_wm_matrix, alpha)
    H_prime = reconstruct_matrix(U_H, S_H_blended, Vt_H)
    LL_prime = P @ H_prime @ P.T
    Y_prime = reconstruct_y_channel(LL_prime, HL, LH, HH).astype(np.uint8)

    psnr, ssim = evaluate_watermark_quality(Y, Y_prime)
    st.success(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # Show images
    col1, col2 = st.columns(2)
    col1.image(Y, caption="Original Image", use_column_width=True, clamp=True)
    col2.image(Y_prime, caption="Watermarked Image", use_column_width=True, clamp=True)

    # Attacks section
    st.header("💥 Robustness Evaluation (Attacks)")
    attack_types = [
        ("no_attack", {}),
        ("salt_pepper", {"amount": 0.01}),
        ("gaussian_noise", {"mean": 0, "std": 15}),
        ("jpeg_compression", {"quality": 90}),
        ("rotation", {"angle": 5}),
        ("scaling", {"scale": 0.7}),
        ("cropping", {"percent": 0.1}),
    ]

    for attack_name, params in attack_types:
        st.subheader(f"🧪 Attack: {attack_name}")
        attacked_image = apply_attack(Y_prime.copy(), attack_name, **params)

        # Extraction
        LL_att, _, _, _ = decompose_watermarked_image(attacked_image)
        H_att = P.T @ LL_att @ P
        U_att, S_att, Vt_att = apply_svd_extraction(H_att)
        S_W_prime_att = extract_encrypted_watermark_singular(S_att, S_H, alpha)
        S_W_prime_crop = S_W_prime_att[:U_wm.shape[0], :U_wm.shape[0]]

        W_E_att = reconstruct_encrypted_watermark(U_wm, S_W_prime_crop, Vt_wm)
        decrypted_att = logistic_decrypt(W_E_att, x0=0.5, r=4)

        ber = calculate_ber(watermark_img, decrypted_att)
        ncc = calculate_ncc(watermark_img, decrypted_att)

        st.write(f"🔐 BER: {ber:.4f} | NCC: {ncc:.4f}")
        st.image(decrypted_att, caption="Decrypted Watermark", use_column_width=False, clamp=True)
