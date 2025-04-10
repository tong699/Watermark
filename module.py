import cv2
import numpy as np
import pywt
import hashlib
from scipy.linalg import hessenberg, svd
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
import pydicom

# ================== Image Preprocessing ==================
def preprocess(img):
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img = clahe.apply(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

# ================== Metadata Extraction ==================
def extract_dicom_metadata(ds):
    fields = ["PatientName", "PatientID", "StudyDate", "Modality", "InstitutionName"]
    content = []
    for field in fields:
        if field in ds:
            content.append(f"{field}: {ds.get(field)}")
    return "\n".join(content)

def generate_watermark_image(text, size=(128, 128)):
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    y0, dy = 15, 15
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(img, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,), 1)
    return img

# ================== Chaotic Encryption ==================
def logistic_map(x0, mu, size):
    x = np.zeros(size)
    x[0] = x0
    for i in range(1, size):
        x[i] = mu * x[i - 1] * (1 - x[i - 1])
    return x

def encrypt_watermark(wm, x0=0.5, mu=4):
    flat = wm.size
    chaos = logistic_map(x0, mu, flat)
    chaos_img = (chaos * 255).astype(np.uint8).reshape(wm.shape)
    return cv2.bitwise_xor(wm, chaos_img)

# ================== RSA ==================
def generate_rsa_keys():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    return priv, priv.public_key()

def sign_hash(private_key, data):
    return private_key.sign(data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())

def verify_signature(public_key, signature, data):
    try:
        public_key.verify(signature, data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        return True
    except Exception:
        return False

# ================== Helpers ==================
def rgb_to_ycbcr(img): return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
def pad_to_square(mat): 
    h, w = mat.shape
    s = max(h, w)
    padded = np.zeros((s, s), dtype=mat.dtype)
    padded[:h, :w] = mat
    return padded, (h, w)
def crop_to_original(mat, shape): return mat[:shape[0], :shape[1]]

# ================== Embedding ==================
def embed_watermark_dual(cover_path, wm_path, alpha=0.02, k=50):
    cover = cv2.imread(cover_path)
    wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    ycbcr = rgb_to_ycbcr(cover)
    Y, Cb, Cr = cv2.split(ycbcr); Y = Y.astype(np.float32)
    LL, (HL, LH, HH) = pywt.dwt2(Y, 'haar')
    wm = cv2.resize(wm, (min(*LL.shape), min(*LL.shape)))
    enc_wm = encrypt_watermark(wm)
    Uw, Sw, Vw = svd(enc_wm, full_matrices=False)

    LL_pad, shape = pad_to_square(LL)
    H, P = hessenberg(LL_pad, calc_q=True)
    U, S, V = svd(H, full_matrices=False)
    Sw_pad = np.pad(Sw[:k], (0, len(S) - k), 'constant')
    S_new = alpha * Sw_pad + (1 - alpha) * S
    H_ = U @ np.diag(S_new) @ V
    LL_ = crop_to_original(P @ H_ @ P.T, shape)

    Y_temp = pywt.idwt2((LL_, (HL, LH, HH)), 'haar')
    temp_img = cv2.cvtColor(cv2.merge((Y_temp.astype(np.uint8), Cb, Cr)), cv2.COLOR_YCrCb2BGR)

    priv_key, pub_key = generate_rsa_keys()
    digest = hashlib.sha256(temp_img.tobytes()).digest()
    sig = sign_hash(priv_key, digest)
    sig_bits = np.unpackbits(np.frombuffer(sig, np.uint8))

    HL_pad, shape_hl = pad_to_square(HL)
    Hh, Ph = hessenberg(HL_pad, calc_q=True)
    Uh, Sh, Vh = svd(Hh, full_matrices=False)
    ks = min(len(sig_bits), len(Sh))
    Sh[-ks:] = alpha * sig_bits[:ks] + (1 - alpha) * Sh[-ks:]
    HL_ = crop_to_original(Ph @ (Uh @ np.diag(Sh) @ Vh) @ Ph.T, shape_hl)

    Y_final = pywt.idwt2((LL_, (HL_, LH, HH)), 'haar')
    final = cv2.merge((Y_final.astype(np.uint8), Cb, Cr))
    return cv2.cvtColor(final, cv2.COLOR_YCrCb2BGR), Uw, Vw, S, Sh, alpha, pub_key, wm

# ================== Extraction ==================
def extract_watermark_dual(path, Uw, Vw, SH, S_HL, alpha, pub, k=50, x0=0.5, mu=4):
    img = cv2.imread(path)
    Y = cv2.split(rgb_to_ycbcr(img))[0].astype(np.float32)
    LL, (HL, LH, HH) = pywt.dwt2(Y, 'haar')

    H_LL, _ = hessenberg(pad_to_square(LL)[0], calc_q=True)
    _, SHw, _ = svd(H_LL, full_matrices=False)
    H_HL, _ = hessenberg(pad_to_square(HL)[0], calc_q=True)
    _, S_HL_wm, _ = svd(H_HL, full_matrices=False)

    Sw1 = (SHw[:k] - (1 - alpha) * SH[:k]) / alpha
    Sw2 = (S_HL_wm[:k] - (1 - alpha) * S_HL[:k]) / alpha
    Sw_avg = (Sw1 + Sw2) / 2

    enc_wm = Uw[:, :k] @ np.diag(Sw_avg) @ Vw[:k, :]
    chaos = (logistic_map(x0, mu, enc_wm.size) * 255).astype(np.uint8).reshape(enc_wm.shape)
    wm = cv2.bitwise_xor(np.clip(enc_wm, 0, 255).astype(np.uint8), chaos)

    bits = ((S_HL_wm[-1024:] - (1 - alpha) * S_HL[-1024:]) / alpha).round().astype(np.uint8)
    sig = np.packbits(bits).tobytes()
    digest = hashlib.sha256(img.tobytes()).digest()
    auth = verify_signature(pub, sig, digest)
    return wm, auth

# ================== Attacks ==================
def apply_jpeg(img, quality=40): return cv2.imdecode(cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1], 1)
def apply_noise(img, std=10): return np.clip(img + np.random.normal(0, std, img.shape), 0, 255).astype(np.uint8)
def apply_crop(img, p=0.2): h, w = img.shape[:2]; ch, cw = int(h*p), int(w*p); y, x = (h-ch)//2, (w-cw)//2; img[y:y+ch, x:x+cw] = 0; return img
def apply_rotate(img, angle=10): h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1); return cv2.warpAffine(img, M, (w,h), borderValue=(0,0,0))
def apply_scale(img, s=0.9): h, w = img.shape[:2]; resized = cv2.resize(img, (int(w*s), int(h*s))); result = np.zeros_like(img); y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2; result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized; return result
def apply_shift(img, dx=10, dy=10): M = np.float32([[1,0,dx],[0,1,dy]]); return cv2.warpAffine(img, M, img.shape[:2][::-1], borderValue=(0,0,0))

# ================== Evaluation ==================
def evaluate_dual(img, Uw, Vw, SH, S_HL, alpha, pub, ref_wm):
    cv2.imwrite("temp_eval.png", img)
    wm, valid = extract_watermark_dual("temp_eval.png", Uw, Vw, SH, S_HL, alpha, pub)
    ref = cv2.resize(ref_wm, wm.shape[::-1])
    _, b1 = cv2.threshold(ref, 127, 1, 0)
    _, b2 = cv2.threshold(wm, 127, 1, 0)
    ncc = np.corrcoef(b1.flatten(), b2.flatten())[0, 1]
    ber = np.mean(b1 != b2)
    return ncc, ber, valid, wm
