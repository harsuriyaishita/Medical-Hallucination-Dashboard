import streamlit as st
import json
import torch

from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer as NLI_Tokenizer, AutoModelForSequenceClassification

from model import MedicalHallucinationDetector


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Medical Hallucination Dashboard",
    layout="wide"
)


# =========================
# 🎨 DARK SaaS UI (PROFESSIONAL + 12% SMALLER CARDS + FIXED ALIGNMENT)
# =========================
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 20% 30%, rgba(59,130,246,0.25), transparent 40%),
        radial-gradient(circle at 80% 20%, rgba(168,85,247,0.25), transparent 40%),
        radial-gradient(circle at 50% 80%, rgba(34,211,238,0.20), transparent 40%),
        linear-gradient(135deg, #0b1220, #111827);
    color: #e5e7eb;
    font-size: 16.5px;
}

/* GLOBAL TEXT */
p, div, span, label {
    font-size: 17px !important;
    color: #e2e8f0 !important;
}

/* MAIN TITLE CENTERED */
h1 {
    font-size: 3.2rem !important;
    font-weight: 900;
    color: #fbbf24 !important;
    text-align: center;
    margin-bottom: 0px;
}

h2, h3 {
    color: white !important;
    text-align: center;
}

/* TEXT AREA */
textarea {
    background: #0f172a !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    border: 1px solid #334155 !important;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    font-size: 16px;
    font-weight: 700;
    border-radius: 10px;
}

/* =========================
   📉 REDUCED CARD SIZE (~12%)
========================= */
.card {
    padding: 13px;               /* ↓ reduced */
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 8px 16px rgba(0,0,0,0.35);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 120px;
}

/* PERFECT CENTER ALIGN INSIDE CARD */
.card h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    text-align: center;
}

.metric {
    font-size: 32px;             /* ↓ reduced */
    font-weight: 900;
    color: #fbbf24;
    margin-top: 6px;
    text-align: center;
    line-height: 1.2;
}

/* COLORS */
.consistency-card { background: linear-gradient(135deg,#3b82f6,#1d4ed8); }
.risk-card { background: linear-gradient(135deg,#ef4444,#b91c1c); }
.status-card { background: linear-gradient(135deg,#10b981,#047857); }
.medcpt-card { background: linear-gradient(135deg,#8b5cf6,#6d28d9); }
.entailment-card { background: linear-gradient(135deg,#22c55e,#15803d); }
.contra-card { background: linear-gradient(135deg,#f97316,#c2410c); }

/* =========================
   🚨 HALLUCINATION CARD (slightly bigger than others but reduced)
========================= */
.hallucination-card {
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    border: 2px solid rgba(255,255,255,0.15);
    padding: 14px;
    transform: scale(1.03);
}

.hallucination-card .metric {
    font-size: 30px !important;
    font-weight: 900;
    color: #ffffff;
}

/* HIGHLIGHT */
.highlight {
    background: rgba(239, 68, 68, 0.35);
    padding: 2px 5px;
    border-radius: 5px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================
# TITLE
# =========================
#st.title("🧠 Medical Hallucination Dashboard")
st.markdown("""
<div style="
    display:flex;
    align-items:center;
    justify-content:center;
    gap:12px;
    font-size:3.1rem;
    font-weight:900;
    color:#fbbf24;
    background: linear-gradient(90deg, #0b1220, #111827);
    padding: 18px;
    border-radius: 14px;
">
🧠 Medical Hallucination Dashboard
</div>
""", unsafe_allow_html=True)
st.caption("MedCPT + NLI + Hallucination Detection System")


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    med_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    med_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device).eval()

    nli_tokenizer = NLI_Tokenizer.from_pretrained("facebook/bart-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to(device).eval()

    return med_tokenizer, med_model, nli_tokenizer, nli_model


med_tokenizer, med_model, nli_tokenizer, nli_model = load_models()

detector = MedicalHallucinationDetector(
    med_tokenizer, med_model, nli_tokenizer, nli_model
)


# =========================
# INPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    ai_text = st.text_area("📝 AI Output", height=240)

with col2:
    final_text = st.text_area("📄 Reference Text", height=240)


# =========================
# HELPERS (UNCHANGED)
# =========================
def classify(consistency, risk):
    if consistency < 0.4 or risk > 0.6:
        return "❌ Not Reliable", "risk-card"
    elif consistency < 0.65 or risk > 0.4:
        return "⚠️ Medium Reliable", "status-card"
    else:
        return "✅ Reliable", "status-card"


def hallucination_score(consistency, risk):
    return risk * (1 - consistency)


def hall_label(score):
    if score > 0.6:
        return "🚨 Hallucination Detected", "hallucination-card"
    elif score > 0.3:
        return "⚠️ Partial Hallucination", "risk-card"
    else:
        return "✅ No Hallucination", "status-card"


def highlight(ai, ref, entail):
    ref_words = set(ref.lower().split())
    return " ".join([
        f"<span class='highlight'>{w}</span>" if w.lower() not in ref_words and entail < 0.6 else w
        for w in ai.split()
    ])


def explain(sim, entail, contra, risk):
    reasons = []
    if sim < 0.5:
        reasons.append("Low MedCPT similarity")
    if entail < 0.5:
        reasons.append("Weak entailment")
    if contra > 0.4:
        reasons.append("Contradiction detected")
    if risk > 0.6:
        reasons.append("High risk generation")
    return " | ".join(reasons) if reasons else "Well supported by reference"


# =========================
# ANALYZE
# =========================
if st.button("🚀 Run Analysis"):

    if not ai_text or not final_text:
        st.warning("Enter both texts")
    else:
        with st.spinner("Analyzing..."):
            result = detector.predict(ai_text, final_text)

        consistency = result["consistency_score"]
        risk = result["risk_score"]
        sim = result["similarity"]
        entail = result["entailment"]
        contra = result["contradiction"]

        cons_pct = round(consistency * 100, 2)
        risk_pct = round(risk * 100, 2)

        hall_score = hallucination_score(consistency, risk)
        hall_pct = round(hall_score * 100, 2)

        hall_text, hall_color = hall_label(hall_score)
        label, color = classify(consistency, risk)

        # =========================
        # MAIN CARDS
        # =========================
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class='card consistency-card'>
                <h3>Consistency</h3>
                <div class='metric'>{cons_pct}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='card risk-card'>
                <h3>Risk</h3>
                <div class='metric'>{risk_pct}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='card {color}'>
                <h3>Status</h3>
                <div class='metric'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # =========================
        # HALLUCINATION CARD
        # =========================
        st.markdown("### 🚨 Hallucination Detection")

        st.markdown(f"""
        <div class='card {hall_color}'>
            <h3>Hallucination Score</h3>
            <div class='metric'>{hall_pct}%</div>
            <h3>{hall_text}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # =========================
        # MODEL SIGNALS
        # =========================
        st.subheader("🧠 Model Signals")

        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown(f"<div class='card medcpt-card'><h3>MedCPT</h3><div class='metric'>{round(sim*100,2)}%</div></div>", unsafe_allow_html=True)

        with s2:
            st.markdown(f"<div class='card entailment-card'><h3>Entailment</h3><div class='metric'>{round(entail*100,2)}%</div></div>", unsafe_allow_html=True)

        with s3:
            st.markdown(f"<div class='card contra-card'><h3>Contradiction</h3><div class='metric'>{round(contra*100,2)}%</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("🧠 Explanation Engine")
        st.info(explain(sim, entail, contra, risk))

        st.subheader("🔥 Hallucinated Content Highlight")
        st.markdown(highlight(ai_text, final_text, entail), unsafe_allow_html=True)


