import pydicom
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import entropy
import cv2
import pywt
from scipy.linalg import hessenberg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import hashlib
import secrets
from skimage.util import random_noise
from skimage.transform import rotate, resize
from skimage.io import imsave
import tempfile
import matplotlib.pyplot as plt


def preprocess_medical_image(image):
    # Step 1: Median filter to reduce salt-and-pepper or speckle noise
    denoised = cv2.medianBlur(image, 3)

    # Step 2: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Step 3: Normalize to 0–255 (float image to uint8)
    norm = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def extract_dicom_metadata_text(ds):
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
            f"{pid}\nAGE:{age} SEX:{sex}\nDOB:{dob}\n"
            f"{inst}\n{mod} {body}\nDATE:{date}"
        )
    except Exception as e:
        print(" Failed to extract metadata:", e)
        text = "UNKNOWN\nDICOM\nMETADATA"

    return text

def generate_text_watermark(text, size=(128, 128), font_size=11):
    from textwrap import wrap

    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    max_width = size[0] - 4  # Padding
    wrapped_lines = []
    for line in text.split("\n"):
        while True:
            # Wrap long lines to fit width
            bbox = draw.textbbox((0, 0), line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                wrapped_lines.append(line)
                break
            else:
                # Try wrapping at the nearest space
                wrapped = wrap(line, width=25)
                wrapped_lines.extend(wrapped)
                break

    # Calculate total height
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 2
    total_text_height = line_height * len(wrapped_lines)

    # Start vertically centered
    y = (size[1] - total_text_height) // 2

    for line in wrapped_lines:
        line_width = draw.textlength(line, font=font)
        x = (size[0] - int(line_width)) // 2
        draw.text((x, y), line, fill=0, font=font)
        y += line_height

    # Add light noise to background (only on white)
    np_img = np.array(img).astype(np.uint8)
    white_mask = (np_img == 255)
    np_img[white_mask] -= np.random.randint(0, 10, size=white_mask.sum()).astype(np.uint8)

    return np_img

def logistic_encrypt(wm_img, x0=0.5, r=4):
    wm_flat = wm_img.flatten()
    N = wm_flat.size
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])

    chaos_sequence = np.floor(x * 256).astype(np.uint8)
    encrypted_flat = np.bitwise_xor(wm_flat, chaos_sequence)
    return encrypted_flat.reshape(wm_img.shape)

# Step 1: Visual entropy
def compute_visual_entropy(image, window_size=8):
    h, w = image.shape
    visual_entropy = 0.0
    blocks = 0

    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            patch = image[i:i+window_size, j:j+window_size].ravel()
            hist, _ = np.histogram(patch, bins=256, range=(0, 256), density=True)
            visual_entropy += entropy(hist + 1e-10, base=2)   # bits
            blocks += 1

    return visual_entropy / blocks if blocks else 0.0


def compute_edge_entropy(image):
    edges = cv2.Canny(image, 100, 200)
    edges01 = (edges > 0).astype(np.uint8)              # 0/1
    hist, _ = np.histogram(edges01, bins=2, range=(0, 2), density=True)
    return entropy(hist + 1e-10, base=2)

def compute_adaptive_alpha(image, lam=0.05):
    E_v = compute_visual_entropy(image)
    E_e = compute_edge_entropy(image)

    if E_v == 0:
        return lam * 0.5

    return lam / (1 + np.exp(-(E_e / E_v)))

def apply_dwt(image):
    # DWT works on float; copy=False avoids extra memory
    coeffs2 = pywt.dwt2(image.astype(np.float32, copy=False), 'haar')
    LL, (HL, LH, HH) = coeffs2          
    return LL, HL, LH, HH               

def apply_svd(matrix):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_matrix = np.diag(S)
    return U, S_matrix, Vt

def blend_singular_values(S_cover, S_watermark, alpha):
    S_new = S_cover.copy()
    wm_size = S_watermark.shape[0]
    S_new[:wm_size, :wm_size] = (
        alpha * S_watermark + (1 - alpha) * S_cover[:wm_size, :wm_size]
    )
    return S_new

def reconstruct_matrix(U, S, Vt):
    return U @ S @ Vt

def reconstruct_y_channel(LL_prime, HL, LH, HH):
    coeffs = (LL_prime, (HL, LH, HH))
    Y_prime = pywt.idwt2(coeffs, 'haar')
    return Y_prime

def evaluate_watermark_quality(original_img, watermarked_img):
    psnr_val = peak_signal_noise_ratio(original_img, watermarked_img, data_range=255)
    ssim_val, _ = structural_similarity(original_img, watermarked_img, data_range=255, full=True)
    return psnr_val, ssim_val
#--------------------------




def decompose_watermarked_image(Y_wm):
    coeffs2 = pywt.dwt2(Y_wm, 'haar')
    LL_WM, (HL_WM, LH_WM, HH_WM) = coeffs2
    return LL_WM, HL_WM, LH_WM, HH_WM

def apply_hessenberg_extraction(LL_WM):
    H_WM, P_WM = hessenberg(LL_WM, calc_q=True)
    return H_WM, P_WM

def apply_svd_extraction(H_WM):
    U_WM, S_WM, Vt_WM = np.linalg.svd(H_WM, full_matrices=False)
    S_WM_matrix = np.diag(S_WM)
    return U_WM, S_WM_matrix, Vt_WM

def singular_value_power_correction(U, S_matrix, Vt, beta=0.95):
    # Apply element-wise power to singular values
    S_corrected = np.diag(np.power(np.diag(S_matrix), beta))
    A = U @ S_corrected @ Vt
    return np.clip(A, 0, 255)

def extract_encrypted_watermark_singular(S_WM, S_H, alpha):
    # Ensure matrix shapes are aligned
    wm_size = S_WM.shape[0]
    S_H_cropped = S_H[:wm_size, :wm_size]
    
    S_W_prime = (S_WM - (1 - alpha) * S_H_cropped) / alpha
    return S_W_prime

def reconstruct_encrypted_watermark(U_w, S_W_prime, Vt_w):
    # Multiply U, S, V^T to get the encrypted watermark matrix
    W_enc = U_w @ S_W_prime @ Vt_w
    return np.clip(W_enc, 0, 255).astype(np.uint8)

def logistic_decrypt(encrypted_img, x0=0.5, r=4):
    wm_flat = encrypted_img.flatten()
    N = wm_flat.size

    # Recreate the same chaotic sequence used in embedding
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])

    chaos_seq = np.floor(x * 256).astype(np.uint8)

    # XOR decryption
    decrypted_flat = np.bitwise_xor(wm_flat.astype(np.uint8), chaos_seq)
    decrypted_img = decrypted_flat.reshape(encrypted_img.shape)
    return decrypted_img

def calculate_ber(original, extracted):
    # Binarize both images (e.g., threshold at 128)
    orig_bin = (original > 128).astype(np.uint8)
    ext_bin = (extracted > 128).astype(np.uint8)
    
    total_bits = orig_bin.size
    bit_errors = np.sum(orig_bin != ext_bin)
    ber = bit_errors / total_bits
    return ber

def calculate_ncc(original, extracted):
    orig_flat = original.flatten().astype(np.float64)
    ext_flat = extracted.flatten().astype(np.float64)

    numerator = np.sum(orig_flat * ext_flat)
    denominator = np.sqrt(np.sum(orig_flat ** 2)) * np.sqrt(np.sum(ext_flat ** 2))

    ncc = numerator / denominator if denominator != 0 else 0
    return ncc


def apply_attack(image, attack_type, **kwargs):
    image = image.astype(np.float64)

    if attack_type == 'no_attack':
        return image
    elif attack_type == 'salt_pepper':
        amount = kwargs.get('amount', 0.02)
        s_vs_p = 0.5  # Salt vs Pepper ratio
        out = image.copy()
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out
    
    elif attack_type == 'gaussian_noise':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 10)
        noise = np.random.normal(mean, std, image.shape)
        attacked = image + noise
        return np.clip(attacked, 0, 255)

    elif attack_type == 'jpeg_compression':
        tmp = np.clip(image, 0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), kwargs.get('quality', 50)]
        _, encimg = cv2.imencode('.jpg', tmp, encode_param)
        decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        return decimg.astype(np.float64)

    elif attack_type == 'scaling':
        scale = kwargs.get('scale', 0.5)
        h, w = image.shape
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)

    elif attack_type == 'rotation':
        angle = kwargs.get('angle', 5)
        h, w = image.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    elif attack_type == 'cropping':
        percent = kwargs.get('percent', 0.2)
        h, w = image.shape
        crop_h = int(h * percent)
        crop_w = int(w * percent)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = image.copy()
        cropped[start_h:start_h + crop_h, start_w:start_w + crop_w] = 0  # zero out center block

        return cropped

    
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
