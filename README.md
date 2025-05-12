# Digital Medical Image Watermarking (DWT + SVD + Logistic Encryption)

## Overview

Streamlit application for embedding and extracting encrypted text watermarks in medical images using integer wavelet transform and singular value decomposition with logistic chaotic encryption.

## Features

* Embed adaptive or fixed-strength watermark in DICOM or grayscale images
* Extract watermark using saved metadata
* Compute and display PSNR, SSIM, BER, NCC metrics
* Download watermarked image and extraction metadata

## Installation

* Create Virtual environment:
  
  ```bash
   python -m venv env
  ```
* Activate:
  ```bash
   env/Scripts/Activate
  ```
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

* Launch application:

  ```bash
  streamlit run app.py
  ```

## Project Structure

* **app.py**: Streamlit interface for watermark embedding and extraction
* **watermark\_core.py**: Core functions for preprocessing, watermark generation, logistic encryption, embedding, extraction, evaluation
* **requirements.txt**: List of Python dependencies

## Dependencies

* streamlit
* opencv-python-headless
* numpy
* PyWavelets
* Pillow
* cryptography
* pydicom
* scikit-image
* pylibjpeg
* pylibjpeg-libjpeg
* pylibjpeg-openjpeg
* matplotlib
* scipy

## License

MIT License
