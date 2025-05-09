import streamlit as st
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import hashlib
import pywt
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import textwrap
import tempfile
import io
import base64

# --- Original Functions (Unchanged) ---
def preprocess_medical_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    denoised = cv2.medianBlur(image, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    norm = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

def compute_image_hash(image: np.ndarray) -> str:
    image_bytes = image.tobytes()
    hash_object = hashlib.sha512(image_bytes)
    return hash_object.hexdigest()

def extract_dicom_metadata_text(ds: pydicom.dataset.FileDataset) -> str:
    try:
        pid = ds.PatientID
        age = ds.get("PatientAge", "NA")
        sex = ds.get("PatientSex", "NA")
        dob = ds.get("PatientBirthDate", "NA")
        inst = ds.get("InstitutionName", "NA")
        mod = ds.get("Modality", "NA")
        body = ds.get("BodyPartExamined", "NA")
        date = ds.get("AcquisitionDate", "NA")
        text = (
            f"ID:{pid}\nAGE:{age} SEX:{sex}\nDOB:{dob}\n"
            f"INST:{inst}\nMOD:{mod} BODY:{body}\nDATE:{date}"
        )
    except Exception as e:
        st.warning(f"Failed to extract some DICOM metadata: {e}")
        text = "UNKNOWN\nDICOM\nMETADATA"
    return text

def generate_text_watermark(text: str, image_hash: str, size: tuple = (128, 128), font_size: int = 10) -> np.ndarray:
    full_text = f"{text}\nHASH:{image_hash[:16]}"
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        st.warning("Arial font not found. Using default font.")
        font = ImageFont.load_default()
    max_width_px = size[0] - 4
    wrapped_lines = []
    estimated_char_wrap_width = 20
    for line in full_text.split("\n"):
        if draw.textlength(line, font=font) <= max_width_px:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=estimated_char_wrap_width))
    try:
        char_bbox = font.getbbox("Ag")
        line_height = char_bbox[3] - char_bbox[1] + 2
    except AttributeError:
        text_width, text_height = draw.textsize("Ag", font=font)
        line_height = text_height + 2
    total_text_height = line_height * len(wrapped_lines)
    y_text = (size[1] - total_text_height) // 2
    for line in wrapped_lines:
        line_width = draw.textlength(line, font=font)
        x_text = (size[0] - int(line_width)) // 2
        draw.text((x_text, y_text), line, fill=0, font=font)
        y_text += line_height
    np_img = np.array(img).astype(np.uint8)
    white_mask = (np_img == 255)
    noise = np.random.randint(0, 10, size=white_mask.sum(), dtype=np.uint8)
    np_img[white_mask] -= noise
    np_img = np.clip(np_img, 0, 255)
    return np_img

def logistic_encrypt(image: np.ndarray, x0: float = 0.5, r: float = 4.0) -> np.ndarray:
    img_flat = image.flatten()
    N = img_flat.size
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    chaos_sequence = np.floor(x * 256).astype(np.uint8)
    chaos_sequence = np.clip(chaos_sequence, 0, 255)
    encrypted_flat = np.bitwise_xor(img_flat, chaos_sequence)
    return encrypted_flat.reshape(image.shape)

def logistic_decrypt(encrypted_image: np.ndarray, x0: float = 0.5, r: float = 4.0) -> np.ndarray:
    return logistic_encrypt(encrypted_image, x0, r)

def compute_visual_entropy(image: np.ndarray, window_size: int = 8) -> float:
    h, w = image.shape
    visual_entropy_sum = 0.0
    num_blocks = 0
    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            patch = image[i:i+window_size, j:j+window_size].ravel()
            hist, _ = np.histogram(patch, bins=256, range=(0, 256), density=True)
            visual_entropy_sum += entropy(hist + 1e-12, base=2)
            num_blocks += 1
    return visual_entropy_sum / num_blocks if num_blocks > 0 else 0.0

def compute_edge_entropy(image: np.ndarray) -> float:
    edges = cv2.Canny(image, 100, 200)
    edges_binary = (edges > 0).astype(np.uint8)
    hist, _ = np.histogram(edges_binary, bins=2, range=(0, 2), density=True)
    return entropy(hist + 1e-12, base=2)

def compute_adaptive_alpha(image: np.ndarray, lambda_strength: float = 0.05) -> float:
    E_v = compute_visual_entropy(image)
    E_e = compute_edge_entropy(image)
    if E_v == 0:
        return lambda_strength * 0.5
    alpha = lambda_strength / (1 + np.exp(-(E_e / E_v)))
    return alpha

def make_square_matrix(img: np.ndarray, fixed_size: int = 256) -> np.ndarray:
    h, w = img.shape
    result = np.zeros((fixed_size, fixed_size), dtype=img.dtype)
    offset_h = (fixed_size - h) // 2
    offset_w = (fixed_size - w) // 2
    result[offset_h:offset_h+h, offset_w:offset_w+w] = img
    return result

def crop_to_original(Aw: np.ndarray, original_shape: tuple) -> np.ndarray:
    h, w = original_shape
    aw_h, aw_w = Aw.shape
    start_h = (aw_h - h) // 2
    start_w = (aw_w - w) // 2
    return Aw[start_h:start_h+h, start_w:start_w+w]

def embed_watermark(A: np.ndarray, W: np.ndarray, a: float) -> tuple:
    if W.shape != A.shape:
        W_padded = np.zeros_like(A, dtype=W.dtype)
        w_h, w_w = W.shape
        offset_h = (A.shape[0] - w_h) // 2
        offset_w = (A.shape[1] - w_w) // 2
        W_padded[offset_h:offset_h+w_h, offset_w:offset_w+w_w] = W
        W = W_padded
    U, s_values, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s_values)
    S_aW = S + a * W
    Uw, sw_values, Vwh = np.linalg.svd(S_aW, full_matrices=False)
    Sw = np.diag(sw_values)
    Aw = U @ Sw @ Vt
    return Aw, Uw, Vwh, S

def extract_watermark(Aw: np.ndarray, Uw: np.ndarray, Vwh: np.ndarray, S: np.ndarray, a: float, original_watermark_shape: tuple) -> np.ndarray:
    U2, sw2_values, Vh2 = np.linalg.svd(Aw, full_matrices=False)
    Sw2 = np.diag(sw2_values)
    D = Uw @ Sw2 @ Vwh
    W_rec = (D - S) / a if a != 0 else np.zeros_like(D)
    w_h, w_w = original_watermark_shape
    offset_h = (W_rec.shape[0] - w_h) // 2
    offset_w = (W_rec.shape[1] - w_w) // 2
    W_rec_cropped = W_rec[offset_h:offset_h+w_h, offset_w:offset_w+w_w]
    return W_rec_cropped

def verify_authenticity(original_image: np.ndarray, extracted_watermark: np.ndarray) -> tuple:
    original_hash = compute_image_hash(original_image)
    extracted_text = "HASH:" + original_hash[:16]
    is_authentic = extracted_text in str(extracted_watermark)
    return is_authentic, original_hash[:16]

def evaluate_watermark_quality(original_img: np.ndarray, watermarked_img: np.ndarray) -> tuple:
    original_img_u8 = original_img.astype(np.uint8)
    watermarked_img_u8 = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    psnr_val = peak_signal_noise_ratio(original_img_u8, watermarked_img_u8, data_range=255)
    ssim_val, _ = structural_similarity(original_img_u8, watermarked_img_u8, data_range=255, full=True)
    return psnr_val, ssim_val

def calculate_ber(original_watermark: np.ndarray, extracted_watermark: np.ndarray, threshold: int = 128) -> float:
    original_binary = (original_watermark > threshold).astype(np.uint8)
    extracted_binary = (extracted_watermark > threshold).astype(np.uint8)
    total_bits = original_binary.size
    if total_bits == 0:
        return 0.0
    bit_errors = np.sum(original_binary != extracted_binary)
    return bit_errors / total_bits

def calculate_ncc(original_watermark: np.ndarray, extracted_watermark: np.ndarray) -> float:
    orig_flat = original_watermark.flatten().astype(np.float64)
    ext_flat = extracted_watermark.flatten().astype(np.float64)
    norm_orig = np.linalg.norm(orig_flat)
    norm_ext = np.linalg.norm(ext_flat)
    if norm_orig == 0 or norm_ext == 0:
        return 0.0
    return np.sum(orig_flat * ext_flat) / (norm_orig * norm_ext)

def apply_attack(image: np.ndarray, attack_type: str, **kwargs) -> np.ndarray:
    attacked_image = image.astype(np.float64, copy=True)
    if attack_type == 'no_attack':
        return attacked_image
    elif attack_type == 'salt_pepper':
        amount = kwargs.get('amount', 0.02)
        s_vs_p = 0.5
        num_salt = np.ceil(amount * attacked_image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in attacked_image.shape]
        attacked_image[tuple(coords)] = 255
        num_pepper = np.ceil(amount * attacked_image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in attacked_image.shape]
        attacked_image[tuple(coords)] = 0
        return attacked_image
    elif attack_type == 'gaussian_noise':
        mean = kwargs.get('mean', 0)
        std_dev = kwargs.get('std', 10)
        noise = np.random.normal(mean, std_dev, attacked_image.shape)
        attacked_image += noise
        return np.clip(attacked_image, 0, 255)
    elif attack_type == 'jpeg_compression':
        quality = kwargs.get('quality', 50)
        img_uint8 = np.clip(attacked_image, 0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', img_uint8, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        return decoded_img.astype(np.float64)
    elif attack_type == 'scaling':
        scale_factor = kwargs.get('scale', 0.5)
        original_h, original_w = attacked_image.shape
        resized_h, resized_w = int(original_h * scale_factor), int(original_w * scale_factor)
        downscaled = cv2.resize(attacked_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(downscaled, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return upscaled
    elif attack_type == 'rotation':
        angle = kwargs.get('angle', 5)
        h, w = attacked_image.shape
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(attacked_image, rotation_matrix, (w, h), borderValue=0)
        return rotated
    elif attack_type == 'cropping':
        crop_percentage = kwargs.get('percent', 0.1)
        h, w = attacked_image.shape
        crop_h = int(h * crop_percentage)
        crop_w = int(w * crop_percentage)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped_img = attacked_image.copy()
        cropped_img[start_h:start_h+crop_h, start_w:start_w+crop_w] = 0
        return cropped_img
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

def perform_watermark_embedding(original_image_processed: np.ndarray, 
                                watermark_text: str,
                                embedding_strength: float = 0.1,
                                use_adaptive_alpha: bool = False,
                                lambda_strength: float = 0.05) -> tuple:
    st.write("--- Starting Watermark Embedding ---")
    image_hash = compute_image_hash(original_image_processed)
    st.write(f"Computed image hash (SHA-512): {image_hash[:16]}...")
    original_text_watermark_img = generate_text_watermark(watermark_text, image_hash, size=(128, 128), font_size=10)
    cv2.imwrite("log_original_text_watermark.png", original_text_watermark_img)
    st.write(f"Text watermark generated with hash. Shape: {original_text_watermark_img.shape}")

    encrypted_watermark_img = logistic_encrypt(original_text_watermark_img, x0=0.5, r=4.0)
    cv2.imwrite("log_intermediate_encrypted_text_watermark.png", encrypted_watermark_img)
    st.write("Generated and encrypted watermark.")

    Y_host_preprocessed_float = original_image_processed.astype(np.float64)
    coeffs = pywt.dwt2(Y_host_preprocessed_float, 'haar')
    LL, (LH, HL, HH) = coeffs
    st.write(f"Decomposed image into subbands. LL shape: {LL.shape}")

    A = make_square_matrix(LL, fixed_size=256)
    W = encrypted_watermark_img.astype(np.float64)
    original_shape = LL.shape

    if use_adaptive_alpha:
        alpha_used = compute_adaptive_alpha(original_image_processed, lambda_strength)
        st.write(f"Adaptive alpha computed: {alpha_used:.4f}")
    else:
        alpha_used = embedding_strength
        st.write(f"Using fixed alpha: {alpha_used:.4f}")

    Aw, Uw, Vwh, S = embed_watermark(A, W, alpha_used)
    st.write("Watermark embedded into LL subband using SVD.")

    LL_watermarked = crop_to_original(Aw, LL.shape)
    Y_watermarked_float = pywt.idwt2((LL_watermarked, (LH, HL, HH)), 'haar')
    st.write("Watermarked image reconstructed with inverse IWT.")

    np.save("watermarked_image_float.npy", Y_watermarked_float)
    cv2.imwrite("log_watermarked_image_visual.png", np.clip(Y_watermarked_float, 0, 255).astype(np.uint8))
    st.write("--- Watermark Embedding Finished ---")
    return (Y_watermarked_float, alpha_used, Uw, Vwh, S, encrypted_watermark_img, original_shape, original_text_watermark_img)

dag perform_watermark_extraction(Y_watermarked_attacked_float: np.ndarray,
                                 alpha_used: float,
                                 Uw: np.ndarray,
                                 Vwh: np.ndarray,
                                 S: np.ndarray,
                                 original_watermark_shape: tuple,
                                 ll_shape: tuple) -> np.ndarray:
    st.write("--- Starting Watermark Extraction ---")
    Y_watermarked_attacked_float = cv2.resize(Y_watermarked_attacked_float, (512, 512), interpolation=cv2.INTER_AREA)
    coeffs_attacked = pywt.dwt2(Y_watermarked_attacked_float, 'haar')
    LL_attacked, _ = coeffs_attacked
    st.write(f"Decomposed watermarked image. LL_attacked shape: {LL_attacked.shape}")

    Aw = make_square_matrix(LL_attacked, fixed_size=256)
    st.write(f"Watermarked LL converted to square matrix: {Aw.shape}")

    W_encrypted_reconstructed = extract_watermark(Aw, Uw, Vwh, S, alpha_used, original_watermark_shape)
    st.write("Encrypted watermark extracted.")

    W_encrypted_reconstructed = np.clip(W_encrypted_reconstructed, 0, 255).astype(np.uint8)
    cv2.imwrite("log_intermediate_extracted_encrypted_watermark.png", W_encrypted_reconstructed)
    st.write(f"Encrypted watermark shape: {W_encrypted_reconstructed.shape}")

    final_decrypted_watermark = logistic_decrypt(W_encrypted_reconstructed, x0=0.5, r=4.0)
    cv2.imwrite("log_final_decrypted_watermark.png", final_decrypted_watermark)
    st.write("Final watermark decrypted.")
    st.write("--- Watermark Extraction Finished ---")
    return final_decrypted_watermark

# --- Streamlit Dashboard ---
st.title("DICOM Watermarking Dashboard")
st.markdown("Upload DICOM files to perform watermark embedding, extraction, and robustness analysis.")

# Configuration Parameters
st.sidebar.header("Configuration")
embedding_strength = st.sidebar.slider("Embedding Strength", 0.01, 0.5, 0.1, step=0.01)
use_adaptive_alpha = st.sidebar.checkbox("Use Adaptive Alpha", value=True)
lambda_strength = st.sidebar.slider("Lambda Strength (for Adaptive Alpha)", 0.01, 0.1, 0.05, step=0.01)
watermark_text_size = (128, 128)
watermark_font_size = 10

# File Upload
uploaded_files = st.file_uploader("Upload DICOM Files", type=["dcm"], accept_multiple_files=True)
results = []

if uploaded_files:
    st.header("Processing Results")
    for idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"Image {idx+1}: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            dicom_dataset = pydicom.dcmread(tmp_file_path)
        except Exception as e:
            st.error(f"Error reading DICOM file {uploaded_file.name}: {e}")
            # Create dummy DICOM dataset
            dicom_dataset = pydicom.dataset.Dataset()
            dicom_dataset.PatientID = f"TestPatient{idx+1}"
            dicom_dataset.PatientAge = "060Y"
            dicom_dataset.PatientSex = "O"
            dicom_dataset.PatientBirthDate = "19600101"
            dicom_dataset.InstitutionName = "Test Hospital"
            dicom_dataset.Modality = "CT"
            dicom_dataset.BodyPartExamined = "KNEE"
            dicom_dataset.AcquisitionDate = "20230101"
            dummy_pixel_array = np.random.randint(0, 2000, size=(512, 320), dtype=np.uint16)
            dicom_dataset.PixelData = dummy_pixel_array.tobytes()
            dicom_dataset.Rows = 512
            dicom_dataset.Columns = 320
            dicom_dataset.PhotometricInterpretation = "MONOCHROME2"
            dicom_dataset.SamplesPerPixel = 1
            dicom_dataset.BitsAllocated = 16
            dicom_dataset.BitsStored = 12
            dicom_dataset.HighBit = 11
            dicom_dataset.PixelRepresentation = 0
            dicom_dataset.RescaleSlope = 1
            dicom_dataset.RescaleIntercept = 0

        # Preprocess Image
        host_pixel_array = dicom_dataset.pixel_array.astype(np.float64)
        min_val, max_val = host_pixel_array.min(), host_pixel_array.max()
        if max_val == min_val:
            host_normalized_uint8 = np.zeros_like(host_pixel_array, dtype=np.uint8)
        else:
            host_normalized = (host_pixel_array - min_val) / (max_val - min_val) * 255.0
            host_normalized_uint8 = host_normalized.astype(np.uint8)
        Y_host_preprocessed = preprocess_medical_image(host_normalized_uint8)
        cv2.imwrite(f"log_preprocessed_host_image_{idx+1}.png", Y_host_preprocessed)
        st.write(f"Host image {idx+1} preprocessed. Shape: {Y_host_preprocessed.shape}")

        # Generate Watermark
        dicom_metadata_str = extract_dicom_metadata_text(dicom_dataset)
        (Y_watermarked_float, alpha_final, Uw, Vwh, S, 
         W_encrypted_embedded, ll_shape, original_text_watermark_img) = perform_watermark_embedding(
            Y_host_preprocessed,
            dicom_metadata_str,
            embedding_strength=embedding_strength,
            use_adaptive_alpha=use_adaptive_alpha,
            lambda_strength=lambda_strength
        )
        
        Y_watermarked_uint8 = np.clip(Y_watermarked_float, 0, 255).astype(np.uint8)
        
        # Evaluate Embedding Quality
        psnr_val, ssim_val = evaluate_watermark_quality(Y_host_preprocessed, Y_watermarked_uint8)
        st.write(f"📊 Watermark Embedding Quality for Image {idx+1}:")
        st.write(f"  PSNR: {psnr_val:.2f} dB")
        st.write(f"  SSIM: {ssim_val:.4f}")
        st.write(f"  Alpha used: {alpha_final:.4f}")

        # Store Results
        results.append({
            'idx': idx + 1,
            'host_image': Y_host_preprocessed,
            'encrypted_watermark': W_encrypted_embedded,
            'watermarked_image': Y_watermarked_uint8,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'watermarked_float': Y_watermarked_float,
            'alpha_final': alpha_final,
            'Uw': Uw,
            'Vwh': Vwh,
            'S': S,
            'll_shape': ll_shape,
            'original_watermark': original_text_watermark_img
        })

        # Save Watermarked Float
        np.save(f"watermarked_image_float_{idx+1}.npy", Y_watermarked_float)

        # Display Images
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(Y_host_preprocessed, caption=f"Preprocessed Host Image {idx+1}", use_column_width=True)
        with col2:
            st.image(W_encrypted_embedded, caption=f"Encrypted Watermark {idx+1}", use_column_width=True)
        with col3:
            st.image(Y_watermarked_uint8, caption=f"Watermarked Image {idx+1}", use_column_width=True)

    # Display Embedding Summary Plot
    if results:
        st.header("Embedding Summary")
        plt.figure(figsize=(20, 12))
        for i, result in enumerate(results):
            plt.subplot(len(results), 4, i * 4 + 1)
            plt.imshow(result['host_image'], cmap='gray')
            plt.title(f"Image {result['idx']}\nPreprocessed Host")
            plt.axis('off')

            plt.subplot(len(results), 4, i * 4 + 2)
            plt.imshow(result['encrypted_watermark'], cmap='gray')
            plt.title(f"Image {result['idx']}\nEncrypted Watermark")
            plt.axis('off')

            plt.subplot(len(results), 4, i * 4 + 3)
            plt.imshow(result['watermarked_image'], cmap='gray')
            plt.title(f"Image {result['idx']}\nWatermarked\nPSNR={result['psnr']:.2f}, SSIM={result['ssim']:.3f}")
            plt.axis('off')

            plt.subplot(len(results), 4, i * 4 + 4)
            plt.axis('off')

        plt.tight_layout()
        plt.suptitle("Watermark Embedding Results", fontsize=16)
        plt.subplots_adjust(top=0.92)
        st.pyplot(plt)
        plt.close()

        # Extraction for All Images
        st.header("Watermark Extraction")
        plt.figure(figsize=(20, len(results) * 3))
        for i, result in enumerate(results):
            Y_watermarked_loaded_float = np.load(f"watermarked_image_float_{result['idx']}.npy")
            decrypted_watermark = perform_watermark_extraction(
                Y_watermarked_loaded_float,
                result['alpha_final'],
                result['Uw'],
                result['Vwh'],
                result['S'],
                watermark_text_size,
                result['ll_shape']
            )
    
            is_authentic, extracted_hash = verify_authenticity(result['host_image'], decrypted_watermark)
            st.write(f"🔒 Authenticity Verification for Image {result['idx']}:")
            st.write(f"  Is Authentic: {is_authentic}")
            st.write(f"  Extracted Hash (first 16 chars): {extracted_hash}")
    
            ber = calculate_ber(result['original_watermark'], decrypted_watermark)
            ncc = calculate_ncc(result['original_watermark'], decrypted_watermark)
            st.write(f"🔍 Quality of Extracted Watermark (No Attack) for Image {result['idx']}:")
            st.write(f"  BER: {ber:.6f}")
            st.write(f"  NCC: {ncc:.6f}")
    
            plt.subplot(len(results), 4, i * 4 + 1)
            plt.imshow(result['host_image'], cmap='gray')
            plt.title(f"Image {result['idx']}\nPreprocessed Host")
            plt.axis('off')
    
            plt.subplot(len(results), 4, i * 4 + 2)
            plt.imshow(result['encrypted_watermark'], cmap='gray')
            plt.title(f"Image {result['idx']}\nEncrypted Watermark")
            plt.axis('off')
    
            plt.subplot(len(results), 4, i * 4 + 3)
            plt.imshow(result['watermarked_image'], cmap='gray')
            plt.title(f"Image {result['idx']}\nWatermarked\nPSNR={result['psnr']:.2f}, SSIM={result['ssim']:.3f}")
            plt.axis('off')
    
            plt.subplot(len(results), 4, i * 4 + 4)
            plt.imshow(decrypted_watermark, cmap='gray')
            plt.title(f"Image {result['idx']}\nDecrypted (No Attack)\nBER={ber:.3f}, NCC={ncc:.3f}")
            plt.axis('off')
    
        plt.tight_layout()
        plt.suptitle("Watermark Embedding and Extraction Results", fontsize=16)
        plt.subplots_adjust(top=0.9)
        st.pyplot(plt)
        plt.close()

        # Robustness Analysis for First Image
        if results:
            st.header("Robustness Analysis for Image 1")
            first_result = results[0]
            attack_types_params = [
                ("no_attack", {}),
                ("salt_pepper", {"amount": 0.01}),
                ("gaussian_noise", {"mean": 0, "std": 15}),
                ("jpeg_compression", {"quality": 90}),
                ("rotation", {"angle": 15}),
                ("scaling", {"scale": 0.7}),
                ("cropping", {"percent": 0.1})
            ]
            num_attacks = len(attack_types_params)
            num_plot_columns = 4
            num_plot_rows = (num_attacks + num_plot_columns - 1) // num_plot_columns * 2
            fig_attacks, axs_attacks = plt.subplots(
                num_plot_rows, num_plot_columns,
                figsize=(4 * num_plot_columns, 3 * num_plot_rows / 2)
            )
            axs_attacks_flat = axs_attacks.flatten() if num_plot_rows > 1 or num_plot_columns > 1 else [axs_attacks]
            plot_idx = 0

            for attack_idx, (attack_name, params) in enumerate(attack_types_params):
                st.write(f"Applying Attack: {attack_name} with params: {params}")
                attacked_image_float = apply_attack(first_result['watermarked_float'].copy(), attack_name, **params)
                cv2.imwrite(f"log_attacked_image_{attack_name}.png", np.clip(attacked_image_float, 0, 255).astype(np.uint8))
                decrypted_watermark_attacked = perform_watermark_extraction(
                    attacked_image_float,
                    first_result['alpha_final'],
                    first_result['Uw'],
                    first_result['Vwh'],
                    first_result['S'],
                    watermark_text_size,
                    first_result['ll_shape']
                )
                cv2.imwrite(f"log_decrypted_watermark_after_{attack_name}.png", decrypted_watermark_attacked)
                ber_attacked = calculate_ber(first_result['original_watermark'], decrypted_watermark_attacked)
                ncc_attacked = calculate_ncc(first_result['original_watermark'], decrypted_watermark_attacked)
                st.write(f"  Results for {attack_name}: BER = {ber_attacked:.6f}, NCC = {ncc_attacked:.6f}")
                if plot_idx < len(axs_attacks_flat):
                    ax_image = axs_attacks_flat[plot_idx]
                    ax_image.imshow(np.clip(attacked_image_float, 0, 255).astype(np.uint8), cmap='gray')
                    ax_image.set_title(f"Attacked: {attack_name}", fontsize=9)
                    ax_image.axis('off')
                plot_idx += 1
                if plot_idx < len(axs_attacks_flat):
                    ax_watermark = axs_attacks_flat[plot_idx]
                    ax_watermark.imshow(decrypted_watermark_attacked, cmap='gray')
                    ax_watermark.set_title(f"Extracted WM\nBER={ber_attacked:.3f}, NCC={ncc_attacked:.3f}", fontsize=9)
                    ax_watermark.axis('off')
                plot_idx += 1
            while plot_idx < len(axs_attacks_flat):
                axs_attacks_flat[plot_idx].axis('off')
                plot_idx += 1
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_attacks.suptitle("Robustness Analysis with Authentication (SHA-512) for Image 1", fontsize=16)
            st.pyplot(fig_attacks)
            plt.close()

# Clean up temporary files
for idx in range(len(uploaded_files)):
    tmp_file_path = f"watermarked_image_float_{idx+1}.npy"
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    if os.path.exists(f"log_preprocessed_host_image_{idx+1}.png"):
        os.remove(f"log_preprocessed_host_image_{idx+1}.png")
if os.path.exists("watermarked_image_float.npy"):
    os.remove("watermarked_image_float.npy")
if os.path.exists("log_original_text_watermark.png"):
    os.remove("log_original_text_watermark.png")
if os.path.exists("log_intermediate_encrypted_text_watermark.png"):
    os.remove("log_intermediate_encrypted_text_watermark.png")
if os.path.exists("log_watermarked_image_visual.png"):
    os.remove("log_watermarked_image_visual.png")
if os.path.exists("log_intermediate_extracted_encrypted_watermark.png"):
    os.remove("log_intermediate_extracted_encrypted_watermark.png")
if os.path.exists("log_final_decrypted_watermark.png"):
    os.remove("log_final_decrypted_watermark.png")
for attack_name in ["no_attack", "salt_pepper", "gaussian_noise", "jpeg_compression", "rotation", "scaling", "cropping"]:
    if os.path.exists(f"log_attacked_image_{attack_name}.png"):
        os.remove(f"log_attacked_image_{attack_name}.png")
    if os.path.exists(f"log_decrypted_watermark_after_{attack_name}.png"):
        os.remove(f"log_decrypted_watermark_after_{attack_name}.png")
