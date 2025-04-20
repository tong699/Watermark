import streamlit as st
import pydicom
import numpy as np
import cv2
import os
from PIL import Image
from scipy.linalg import hessenberg
import matplotlib.pyplot as plt

# Import your watermarking methods
from watermark_module import (
    preprocess_medical_image,
    extract_dicom_metadata_text,
    generate_text_watermark,
    logistic_encrypt,
    compute_adaptive_alpha,
    apply_dwt, apply_svd,
    blend_singular_values, reconstruct_matrix,
    reconstruct_y_channel, evaluate_watermark_quality,
    decompose_watermarked_image, apply_hessenberg_extraction,
    apply_svd_extraction, singular_value_power_correction,
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

    normalized = (cover_image - cover_image.min()) / (cover_image.max() - cover_image.min()) * 255
    normalized = normalized.astype(np.uint8)

    # Apply preprocessing
    Y = preprocess_medical_image(normalized)

    text_input = extract_dicom_metadata_text(ds)
    watermark_img = generate_text_watermark(text_input)
    encrypted_watermark = logistic_encrypt(watermark_img)
    cv2.imwrite("encrypted_text_watermark.png", encrypted_watermark)

    U_wm, S_wm, Vt_wm = np.linalg.svd(encrypted_watermark.astype(np.float64), full_matrices=False)
    S_wm_matrix = np.diag(S_wm)

    alpha = compute_adaptive_alpha(Y, lam=0.18)
    LL, HL, LH, HH = apply_dwt(Y)
    H, P = hessenberg(LL, calc_q=True)
    U_H, S_H, Vt_H = np.linalg.svd(H.astype(np.float64), full_matrices=False)
    S_H = np.diag(S_H)

    S_H_blended = blend_singular_values(S_H, S_wm_matrix, alpha)
    H_prime = reconstruct_matrix(U_H, S_H_blended, Vt_H)
    LL_prime = P @ H_prime @ P.T
    Y_prime = reconstruct_y_channel(LL_prime, HL, LH, HH)

    np.save("watermarked_image.npy", Y_prime)
    
    psnr, ssim = evaluate_watermark_quality(Y.astype(np.uint8), Y_prime)
    print(f"📊 PSNR: {psnr:.2f} dB")
    print(f"📊 SSIM: {ssim:.4f}")
    
    #Extraction
    Y_wm = np.load("watermarked_image.npy")
    LL_WM, HL_WM, LH_WM, HH_WM = decompose_watermarked_image(Y_wm)
    print("LL_WM shape:", LL_WM.shape)
    H_WM = P.T @ LL_WM @ P
    U_WM, S_WM, Vt_WM = apply_svd_extraction(H_WM)
    # Use U_WM, S_WM, Vt_WM from Step 4
    # corrected_watermark = singular_value_power_correction(U_WM, S_WM, Vt_WM)
    corrected_watermark = U_WM @ np.diag(np.diag(S_WM)) @ Vt_WM
    
    # Convert to uint8 image
    watermark_extracted = corrected_watermark.astype(np.uint8)
    cv2.imwrite("extracted_watermark.png", watermark_extracted)
    S_W_prime = extract_encrypted_watermark_singular(S_WM, S_H, alpha)
    # Assume S_W_prime is 256x256 from the host, but watermark was 64x64
    wm_size = U_wm.shape[0]  
    S_W_prime_cropped = S_W_prime[:wm_size, :wm_size]

    # Reconstruct the encrypted watermark image
    W_E_prime = reconstruct_encrypted_watermark(U_wm, S_W_prime_cropped, Vt_wm)

    # Optional: Save it
    cv2.imwrite("extracted_encrypted_watermark.png", W_E_prime)

    # W_E_prime: encrypted watermark image from Step 7
    # Use same x0 and r as in Step 1 of embedding
    decrypted_watermark = logistic_decrypt(W_E_prime, x0=0.5, r=4)

    # Save or display
    cv2.imwrite("final_decrypted_watermark.png", decrypted_watermark)

    ber = calculate_ber(watermark_img, decrypted_watermark)
    ncc = calculate_ncc(watermark_img, decrypted_watermark)

    print(f"📊 BER:  {ber:.4f}")
    print(f"📊 NCC:  {ncc:.4f}")
    

    print("\n🔍 Robustness Testing Under Attacks")
    attack_types = [
        ("no_attack", {}),
        ("salt_pepper", {"amount": 0.01}),
        ("gaussian_noise", {"mean": 0, "std": 15}),
        ("jpeg_compression", {"quality": 90}),
        ("rotation", {"angle": 5}),
        ("scaling", {"scale": 0.7}),
        ("cropping", {"percent": 0.1})
    ]

    for attack_name, params in attack_types:
        print(f"\n🧪 Attack: {attack_name} | Params: {params}")

        attacked_image = apply_attack(Y_prime.copy(), attack_name, **params)

        # Extraction from attacked image
        LL_att, HL_att, LH_att, HH_att = decompose_watermarked_image(attacked_image)
        H_att = P.T @ LL_att @ P
        U_att, S_att, Vt_att = apply_svd_extraction(H_att)

        # Singular value extraction
        S_W_prime_att = extract_encrypted_watermark_singular(S_att, S_H, alpha)
        S_W_prime_crop = S_W_prime_att[:wm_size, :wm_size]
        W_E_att = reconstruct_encrypted_watermark(U_wm, S_W_prime_crop, Vt_wm)
        decrypted_att = logistic_decrypt(W_E_att, x0=0.5, r=4)

        # Quality metrics
        ber_att = calculate_ber(watermark_img, decrypted_att)
        ncc_att = calculate_ncc(watermark_img, decrypted_att)

        # Save result
        cv2.imwrite(f"decrypted_{attack_name}.png", decrypted_att)

        print(f"🔐 BER after {attack_name}:  {ber_att:.4f}")
        print(f"🔐 NCC after {attack_name}:  {ncc_att:.4f}")
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(attacked_image.astype(np.uint8), cmap='gray')
        plt.title(f"Attacked Image: {attack_name}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(decrypted_att, cmap='gray')
        plt.title(f"Decrypted Watermark\nBER: {ber_att:.4f}, NCC: {ncc_att:.4f}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
