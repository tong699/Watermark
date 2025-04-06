import streamlit as st
import cv2
import numpy as np
import pywt
import qrcode
import base64
import hashlib
import io
from PIL import Image
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# Generate RSA key pair
def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key

# Sign watermark text using the private key
def sign_watermark_text(watermark_text, private_key):
    hasher = hashlib.sha256()
    hasher.update(watermark_text.encode())
    digest = hasher.digest()
    signature = private_key.sign(
        digest,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    signature_b64 = base64.b64encode(signature).decode()
    return signature_b64

# Generate a QR code that includes the watermark text and its signature
def generate_qr_with_signature(watermark_text, signature_b64):
    payload = f"WM:{watermark_text}\nSIG:{signature_b64}"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=3,
        border=2,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = np.array(img.convert('L'))
    return img

# Embed the watermark (QR image) into the cover image using DWT and DCT
def embed_watermark(cover_img, watermark_img, level, alpha):
    # Multi-level DWT on the cover image.
    coeffs = pywt.wavedec2(cover_img, 'haar', level=level)
    # Use detail coefficients from one of the levels (here level 1)
    LH, HL, HH = coeffs[1]
    
    # Resize watermark to match HL sub-band shape.
    wm_resized = cv2.resize(watermark_img, (HL.shape[1], HL.shape[0]))
    wm_norm = wm_resized.astype(np.float32) / 255.0

    # Apply DCT to the HL sub-band.
    HL = np.float32(HL)
    HL_dct = cv2.dct(HL)
    
    # Embed the watermark by adding a scaled version of it.
    HL_dct_embedded = HL_dct + alpha * wm_norm * HL_dct.max()
    
    # Inverse DCT to recover modified sub-band.
    HL_embedded = cv2.idct(HL_dct_embedded)
    coeffs[1] = (LH, HL_embedded, HH)
    
    # Reconstruct watermarked image using inverse DWT.
    watermarked_img = pywt.waverec2(coeffs, 'haar')
    watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    return watermarked_img

# Convert numpy image to bytes for download
def convert_image_to_bytes(img_array):
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im

st.title("Robust Medical Image Watermarking")

st.write("Upload a cover image and adjust parameters to embed a robust watermark that can survive attacks.")

# File uploader for cover image
cover_file = st.file_uploader("Upload Cover Image (preferably grayscale)", type=["jpg", "jpeg", "png"])

# Input for watermark text
watermark_text = st.text_input("Watermark Text", "Confidential watermark")

# Adjustable parameters
alpha = st.slider("Embedding Strength (alpha)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
dwt_level = st.slider("DWT Level", min_value=1, max_value=3, value=2)

if cover_file is not None:
    # Load cover image and convert to grayscale
    cover_image = Image.open(cover_file).convert("L")
    cover_img = np.array(cover_image)
    
    # Generate RSA keys and sign the watermark text
    private_key, public_key = generate_rsa_keys()
    signature_b64 = sign_watermark_text(watermark_text, private_key)
    
    # Generate QR watermark containing text and signature
    qr_watermark = generate_qr_with_signature(watermark_text, signature_b64)
    
    st.image(cover_image, caption="Cover Image", use_column_width=True)
    st.image(qr_watermark, caption="Generated QR Watermark", use_column_width=True)
    
    # Embed the QR watermark into the cover image
    watermarked_img = embed_watermark(cover_img, qr_watermark, dwt_level, alpha)
    
    st.image(watermarked_img, caption="Watermarked Image", use_column_width=True)
    
    # Provide download option for the watermarked image
    watermarked_bytes = convert_image_to_bytes(watermarked_img)
    st.download_button("Download Watermarked Image", watermarked_bytes, "watermarked.png", "image/png")
    
    # Display public key for later authentication verification
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    st.text_area("Public Key (for verification)", pem.decode(), height=200)
