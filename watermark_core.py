import pydicom
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import entropy
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pywt
import hashlib
import os

# --- Image Preprocessing ---
def preprocess_medical_image(image: np.ndarray) -> np.ndarray:
    # Resize to standard 512x512
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    denoised = cv2.medianBlur(image, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    norm = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

# --- Hash Computation ---
def compute_image_hash(image: np.ndarray) -> str:
    image_bytes = image.tobytes()
    hash_object = hashlib.sha512(image_bytes)
    return hash_object.hexdigest()

# --- Watermark Generation & Encryption ---
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
        print(f"Warning: Failed to extract some DICOM metadata: {e}")
        text = "UNKNOWN\nDICOM\nMETADATA"
    return text

def generate_text_watermark(text: str, size: tuple = (128, 128), font_size: int = 10) -> np.ndarray:
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Arial font not found. Using default font.")
        font = ImageFont.load_default()
    max_width_px = size[0] - 4
    wrapped_lines = []
    estimated_char_wrap_width = 20
    for line in text.split("\n"):
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

# --- Adaptive Alpha Calculation (Optional) ---
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

# --- Authentication Verification ---
def verify_authenticity(original_image: np.ndarray, extracted_watermark: np.ndarray) -> tuple:
    original_hash = compute_image_hash(original_image)
    extracted_text = "HASH:" + original_hash[:16]
    is_authentic = extracted_text in str(extracted_watermark)
    return is_authentic, original_hash[:16]

# --- Evaluation Metrics ---
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

# --- Attack Simulation ---
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
        quality = kwargs.get('quality', 150)
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
        angle = kwargs.get('angle', 30)
        h, w = attacked_image.shape
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(attacked_image, rotation_matrix, (w, h), borderValue=0)
        return rotated
    elif attack_type == 'cropping':
        crop_percentage = kwargs.get('percent', 0.3)
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

# --- Main Embedding and Extraction Workflows ---
def perform_watermark_embedding(original_image_processed: np.ndarray, 
                                watermark_text: str,
                                embedding_strength: float = 0.1,
                                use_adaptive_alpha: bool = False,
                                lambda_strength: float = 0.05) -> tuple:
    print("--- Starting Watermark Embedding ---")
    original_text_watermark_img = generate_text_watermark(watermark_text, size=(128, 128), font_size=10)
    cv2.imwrite("log_original_text_watermark.png", original_text_watermark_img)
    print(f"Text watermark generated. Shape: {original_text_watermark_img.shape}")

    encrypted_watermark_img = logistic_encrypt(original_text_watermark_img, x0=0.5, r=4.0)
    cv2.imwrite("log_intermediate_encrypted_text_watermark.png", encrypted_watermark_img)
    print("Generated and encrypted watermark.")

    Y_host_preprocessed_float = original_image_processed.astype(np.float64)
    coeffs = pywt.dwt2(Y_host_preprocessed_float, 'haar')
    LL, (LH, HL, HH) = coeffs
    print(f"Decomposed image into subbands. LL shape: {LL.shape}")

    A = make_square_matrix(LL, fixed_size=256)
    W = encrypted_watermark_img.astype(np.float64)
    original_shape = LL.shape  # Store LL shape for extraction

    if use_adaptive_alpha:
        alpha_used = compute_adaptive_alpha(original_image_processed, lambda_strength)
        print(f"Adaptive alpha computed: {alpha_used:.4f}")
    else:
        alpha_used = embedding_strength
        print(f"Using fixed alpha: {alpha_used:.4f}")

    Aw, Uw, Vwh, S = embed_watermark(A, W, alpha_used)
    print("Watermark embedded into LL subband using SVD.")

    LL_watermarked = crop_to_original(Aw, LL.shape)
    Y_watermarked_float = pywt.idwt2((LL_watermarked, (LH, HL, HH)), 'haar')
    print("Watermarked image reconstructed with inverse DWT.")

    np.save("watermarked_image_float.npy", Y_watermarked_float)
    cv2.imwrite("log_watermarked_image_visual.png", np.clip(Y_watermarked_float, 0, 255).astype(np.uint8))
    print("--- Watermark Embedding Finished ---")
    return (Y_watermarked_float, alpha_used, Uw, Vwh, S, encrypted_watermark_img, original_shape, original_text_watermark_img)
                                    
def perform_watermark_extraction(Y_watermarked_attacked_float: np.ndarray,
                                 alpha_used: float,
                                 Uw: np.ndarray,
                                 Vwh: np.ndarray,
                                 S: np.ndarray,
                                 original_watermark_shape: tuple,
                                 ll_shape: tuple) -> np.ndarray:
    print("--- Starting Watermark Extraction ---")
    Y_watermarked_attacked_float = cv2.resize(Y_watermarked_attacked_float, (512, 512), interpolation=cv2.INTER_AREA)
    coeffs_attacked = pywt.dwt2(Y_watermarked_attacked_float, 'haar')
    LL_attacked, _ = coeffs_attacked
    print(f"Decomposed watermarked image. LL_attacked shape: {LL_attacked.shape}")

    Aw = make_square_matrix(LL_attacked, fixed_size=256)
    print(f"Watermarked LL converted to square matrix: {Aw.shape}")

    W_encrypted_reconstructed = extract_watermark(Aw, Uw, Vwh, S, alpha_used, original_watermark_shape)
    print("Encrypted watermark extracted.")

    W_encrypted_reconstructed = np.clip(W_encrypted_reconstructed, 0, 255).astype(np.uint8)
    cv2.imwrite("log_intermediate_extracted_encrypted_watermark.png", W_encrypted_reconstructed)
    print(f"Encrypted watermark shape: {W_encrypted_reconstructed.shape}")

    final_decrypted_watermark = logistic_decrypt(W_encrypted_reconstructed, x0=0.5, r=4.0)
    cv2.imwrite("log_final_decrypted_watermark.png", final_decrypted_watermark)
    print("Final watermark decrypted.")
    print("--- Watermark Extraction Finished ---")
    return final_decrypted_watermark
