"""
👁️ OptiCare AI - Advanced Medical Eye Analysis Platform
"""

import sys
import os
from pathlib import Path

# -------------------------------------------------
# FIX IMPORT PATHS (CRITICAL)
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st
from PIL import Image
import pandas as pd
import gdown
import nltk

from config.settings import Config
from src.dataloader import DataLoader
from src.embeddings import EmbeddingManager, SemanticSearch
from src.chatbot import MedicalChatbot
from src.safety import SafetyProtocol

from download_model import MODEL_PATH, download_model

# -------------------------------------------------
# NLTK DOWNLOAD (FIX FOR STREAMLIT CLOUD)
# -------------------------------------------------
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("punkt", quiet=True)

# -------------------------------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE (ONLY IF NOT FOUND)
# -------------------------------------------------
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
    with st.spinner("📥 Downloading AI model..."):
        download_model()

# -------------------------------------------------
# IMAGE CLASSIFIER (SHOW REAL ERRORS)
# -------------------------------------------------
EyeDiseaseClassifier = None
ImageQualityChecker = None

try:
    from src.image_classifier import EyeDiseaseClassifier, ImageQualityChecker
except Exception as e:
    st.error(f"❌ Image classifier failed to load:\n\n{e}")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="OptiCare AI - Eye Disease Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"# {Config.APP_NAME}\n\nVersion {Config.VERSION}\n\nAI-powered eye disease analysis platform."
    }
)

# -------------------------------------------------
# SYSTEM INITIALIZATION (CACHED)
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def initialize_system():
    # Data
    data_loader = DataLoader(str(Config.DATA_DIR))
    texts, diseases, metadata = data_loader.load_documents()

    # If data is missing, fallback
    if not texts:
        texts = ["Welcome to OptiCare AI. Dataset missing on Streamlit Cloud."]
        diseases = ["No disease data"]
        metadata = [{}]

    stats = data_loader.get_statistics()

    # Embeddings
    embedding_manager = EmbeddingManager(Config.EMBEDDING_MODEL)
    embedding_manager.create_index(texts)

    semantic_search = SemanticSearch(embedding_manager, texts, diseases)
    chatbot = MedicalChatbot(semantic_search)
    safety = SafetyProtocol()

    classifier = EyeDiseaseClassifier() if EyeDiseaseClassifier else None
    quality_checker = ImageQualityChecker() if ImageQualityChecker else None

    return chatbot, safety, classifier, quality_checker, stats

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "initialized" not in st.session_state:
    (
        st.session_state.chatbot,
        st.session_state.safety,
        st.session_state.classifier,
        st.session_state.quality_checker,
        st.session_state.stats
    ) = initialize_system()

    st.session_state.initialized = True
    st.session_state.chat_history = []
    st.session_state.analysis_count = 0

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">👁️ OptiCare AI</h1>
    <p style="text-align:center;">
    Advanced Medical Eye Analysis & Disease Detection Platform
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")

    st.metric("Knowledge Sentences", st.session_state.stats.get("total_sentences", 0))
    st.metric("Diseases", st.session_state.stats.get("total_diseases", 0))
    st.metric("Analyses Performed", st.session_state.analysis_count)

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)
    show_probabilities = st.checkbox("Show All Probabilities", True)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history.clear()
        st.session_state.chatbot.clear_conversation()
        st.success("Chat history cleared")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 AI Chatbot",
    "🖼️ Image Analysis",
    "📊 Analytics",
    "📚 Medical Info"
])

# ================= TAB 1: CHATBOT =================
with tab1:
    st.warning(Config.MEDICAL_DISCLAIMER)

    question = st.text_input(
        "Ask a question about eye diseases:",
        placeholder="What are symptoms of glaucoma?"
    )

    if st.button("🔍 Analyze") and question:
        safety_result = st.session_state.safety.evaluate(question)

        if safety_result["emergency"]["is_emergency"]:
            st.error(safety_result["emergency"]["message"])
        else:
            response = st.session_state.chatbot.get_answer(
                question,
                k=Config.TOP_K_RESULTS
            )

            disease_info = Config.get_disease_info(response["disease"])

            st.success(
                f"{disease_info['emoji']} {disease_info['name']} "
                f"({response['confidence']:.1%} confidence)"
            )

            st.write(response["answer"])
            st.session_state.analysis_count += 1

# ================= TAB 2: IMAGE ANALYSIS =================
with tab2:
    uploaded_file = st.file_uploader(
        "Upload an eye image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.session_state.quality_checker:
            quality = st.session_state.quality_checker.check_quality(image)
            st.info(quality["recommendation"])

        if st.session_state.classifier:
            if st.button("🔬 Analyze Image"):
                result = st.session_state.classifier.analyze_with_confidence(
                    image,
                    threshold=confidence_threshold
                )

                st.success(f"Prediction: {result['predicted_class']}")
                st.metric("Confidence", f"{result['confidence']:.1%}")

                if show_probabilities:
                    st.table(
                        pd.DataFrame(
                            result["all_probabilities"].items(),
                            columns=["Disease", "Probability"]
                        ).sort_values("Probability", ascending=False)
                    )

                st.session_state.analysis_count += 1
        else:
            st.error("❌ Image classifier not available")

# ================= TAB 3: ANALYTICS =================
with tab3:
    st.metric("Total Questions", len(st.session_state.chat_history))
    st.metric("Total Analyses", st.session_state.analysis_count)

# ================= TAB 4: MEDICAL INFO =================
with tab4:
    st.markdown(st.session_state.safety.get_when_to_see_doctor())

    for key, disease in Config.DISEASES.items():
        with st.expander(f"{disease['emoji']} {disease['name']}"):
            st.write(f"Severity: **{disease['severity'].title()}**")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    f"<center>👁️ OptiCare AI v{Config.VERSION} — Educational Use Only</center>",
    unsafe_allow_html=True
)
