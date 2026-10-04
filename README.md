# SkinDocAI 🔬

## AI-Based Skin Cancer Detection from Medical PDF Images Using HAM10000

SkinDocAI is an academic AI project that accepts a **PDF containing a skin-lesion image**, extracts the image, preprocesses it, and uses a deep-learning model trained on the **HAM10000 dataset** to classify the image as **Benign** or **Potentially Malignant**.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical diagnostic tool and does not replace evaluation by a qualified healthcare professional.

## Project Workflow

```text
PDF Upload
    ↓
Extract Skin Image using PyMuPDF
    ↓
Resize to 224 × 224
    ↓
Normalize Pixel Values
    ↓
HAM10000-trained Deep Learning Model
    ↓
Benign / Potentially Malignant
    ↓
Confidence Score
```

## Objectives

- Analyze skin-lesion images using AI.
- Use HAM10000 for model development.
- Extract skin images automatically from PDF documents.
- Preprocess images for model prediction.
- Classify images as **Benign** or **Potentially Malignant**.
- Display prediction and confidence through Streamlit.

## Dataset

The project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

Current implementation:

| Information | Count |
|---|---:|
| Total metadata records | 10,015 |
| Available images | 1,200 |
| Matched records | 1,200 |
| Unique lesions | 1,154 |
| Benign | 964 |
| Malignant | 236 |

The available images were divided into:

- Training: **960 images**
- Validation: **240 images**

Binary labels:

```text
0 → Benign
1 → Malignant
```

## Methodology

### 1. Dataset Preparation
HAM10000 metadata is matched with the available image files using the image ID.

### 2. Image Preprocessing
Each image is converted to RGB, resized to **224 × 224 pixels**, and normalized from **0–255 to 0–1**.

### 3. Model Training
A deep-learning image-classification model is trained using the prepared HAM10000 images and saved as:

```text
model.keras
```

### 4. PDF Processing
PyMuPDF opens the uploaded PDF, scans its pages, identifies embedded images, extracts the image, and converts it to RGB.

### 5. Prediction

```text
Prediction < 0.5  → Benign
Prediction ≥ 0.5  → Potentially Malignant
```

The application displays the predicted category and model confidence.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Dataset | HAM10000 |
| Deep Learning | TensorFlow / Keras |
| PDF Processing | PyMuPDF |
| Image Processing | Pillow |
| Numerical Processing | NumPy |
| Data Processing | Pandas |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib / Seaborn |
| Web Interface | Streamlit |
| Model Format | `.keras` |

## Results

The model achieved **100% accuracy on the current 240-image validation split**.

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Benign | 1.00 | 1.00 | 1.00 |
| Malignant | 1.00 | 1.00 | 1.00 |

Confusion matrix:

```text
                 Predicted
              Benign  Malignant

Actual Benign     141       0
Actual Malignant    0      99
```

> These results are from the current validation split and should not be interpreted as clinical accuracy.

## Application

The Streamlit application provides:

- PDF upload
- Extracted skin-image preview
- AI prediction
- Confidence score
- Model information
- Medical disclaimer

Run the application with:

```bash
streamlit run app.py
```

The application is normally available at:

```text
http://localhost:8501
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow streamlit pymupdf pillow
```

## Example Prediction

If the model produces:

```text
Prediction = 0.7105
```

then the system classifies it as:

```text
Potentially Malignant
Confidence: 71.05%
```

The confidence is the model's classification output and should not be interpreted as the probability that a person has cancer.

## Limitations

- The current implementation uses 1,200 available images rather than the complete HAM10000 dataset.
- The available data is imbalanced toward benign cases.
- Evaluation is based on the current validation split.
- The system is not clinically validated.
- PDF image quality and formatting may affect image extraction.
- Model performance may differ on images that are different from the training data.
- The system does not consider a patient's complete medical history or clinical examination.

## Future Scope

- Grad-CAM for explainable AI.
- Multi-class classification using original HAM10000 categories.
- Larger datasets and independent external testing.
- Automated PDF report generation.
- Better handling of PDFs containing multiple images.
- Cloud deployment.
- Additional explainability methods.

## Project Information

**Project:** SkinDocAI  
**Domain:** Artificial Intelligence / Deep Learning / Healthcare  
**Dataset:** HAM10000  
**Application:** Streamlit  
**Academic Year:** 2026–27

## References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific Data, 5, 180161.
2. Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017). *Dermatologist-level classification of skin cancer with deep neural networks*. Nature, 542, 115–118.
3. Haenssle, H. A., Fink, C., Schneiderbauer, R., et al. (2018). *Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists*. Annals of Oncology, 29(8), 1836–1842.
4. Codella, N. C. F., Gutman, D., Celebi, M. E., et al. (2018). *Skin lesion analysis toward melanoma detection*. IEEE.

---

## Disclaimer

SkinDocAI is an academic project for learning, research, and demonstration of AI techniques. It is **not intended to diagnose, treat, or prevent cancer**. Any suspicious skin lesion should be evaluated by a qualified healthcare professional.
