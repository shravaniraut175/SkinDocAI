import os

# Reduce TensorFlow terminal messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import pymupdf
import numpy as np
from PIL import Image
import io


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SkinDocAI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Remove unnecessary top spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    /* Result card */
    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        background-color: #f7f7f7;
        border: 1px solid #dddddd;
        margin-top: 0.5rem;
    }

    .result-title {
        font-size: 1rem;
        color: #666;
        margin-bottom: 0.3rem;
    }

    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .confidence-value {
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Compact warning */
    .disclaimer {
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.8rem;
        line-height: 1.3;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Reduce header space */
    header {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🔬 SkinDocAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Skin Lesion Analysis using HAM10000</div>',
    unsafe_allow_html=True
)


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload Medical Report (PDF)",
    type=["pdf"],
    label_visibility="visible"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.read()

    pdf_document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    extracted_image = None

    # Search for image
    for page in pdf_document:

        images = page.get_images(full=True)

        if images:

            image_info = images[0]

            xref = image_info[0]

            base_image = pdf_document.extract_image(xref)

            image_bytes = base_image["image"]

            extracted_image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            break


    # =====================================================
    # NO IMAGE
    # =====================================================

    if extracted_image is None:

        st.warning(
            "⚠️ No skin-lesion image was found in this PDF. "
            "Please upload a PDF containing a lesion photograph."
        )


    # =====================================================
    # IMAGE FOUND
    # =====================================================

    else:

        st.success("✓ PDF analyzed successfully")

        # -------------------------------------------------
        # TWO COLUMN LAYOUT
        # -------------------------------------------------

        left_col, right_col = st.columns(
            [1.15, 1],
            gap="large"
        )


        # =================================================
        # LEFT SIDE — IMAGE
        # =================================================

        with left_col:

            st.markdown(
                "### 🖼️ Extracted Skin Image"
            )

            st.image(
                extracted_image,
                caption="Image extracted from uploaded PDF",
                width="stretch"
            )


        # =================================================
        # RIGHT SIDE — ANALYSIS
        # =================================================

        with right_col:

            st.markdown(
                "### 🔍 AI Analysis"
            )

            # ---------------------------------------------
            # PREPROCESS IMAGE
            # ---------------------------------------------

            image = extracted_image.resize(
                (224, 224)
            )

            image_array = np.array(image) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # ---------------------------------------------
            # PREDICTION
            # ---------------------------------------------

            prediction = model.predict(
                image_array,
                verbose=0
            )[0][0]


            # ---------------------------------------------
            # RESULT
            # ---------------------------------------------

            if prediction >= 0.5:

                result = "Potentially Malignant"

                confidence = prediction * 100

                result_icon = "⚠️"

            else:

                result = "Benign"

                confidence = (1 - prediction) * 100

                result_icon = "✅"


           # ---------------------------------------------
            # RESULT
            # ---------------------------------------------

            st.markdown("### Prediction")

            if result == "Benign":

                st.success(
                    f"✅ **Benign**\n\n"
                    f"Model confidence: **{confidence:.2f}%**"
                )

            else:

                st.error(
                    f"⚠️ **Potentially Malignant**\n\n"
                    f"Model confidence: **{confidence:.2f}%**"
                )


            # ---------------------------------------------
            # PREDICTION SCORE
            # ---------------------------------------------

            st.markdown("#### Prediction Score")

            st.progress(
                min(int(confidence), 100)
            )


            # ---------------------------------------------
            # MODEL INFORMATION
            # ---------------------------------------------

            st.markdown(
                """
                **Model:** HAM10000-trained CNN  
                **Dataset:** HAM10000  
                **Task:** Binary skin-cancer     classification
                """
            )

            

        # =================================================
        # DISCLAIMER
        # =================================================

        st.markdown(
            """
            <div class="disclaimer">
            ⚠️ <b>Disclaimer:</b> This system is developed for
            educational and research purposes only. It is not a
            medical diagnostic tool. The prediction should not be
            used for clinical decisions. Please consult a qualified
            healthcare professional for medical evaluation.
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# BEFORE PDF UPLOAD
# =========================================================

else:

    st.info(
        "👆 Upload a PDF containing a skin-lesion image to begin analysis."
    )