import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🇪🇹 Amharic-Oromo Translator",
    page_icon="🇪🇹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# ETHIOPIAN FLAG COLORS
# ============================================================
GREEN = "#078930"
YELLOW = "#FCDD09"
RED = "#DA121A"
BLUE = "#0F47AF"
DARK = "#1a1a2e"
LIGHT = "#f8f9fa"

# ============================================================
# CUSTOM CSS — PROFESSIONAL & MODERN
# ============================================================
st.markdown(f"""
<style>
    /* ── Global ── */
    .main {{
        background: linear-gradient(145deg, #ffffff 0%, #f2f4f8 100%);
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }}

    /* ── Title ── */
    .title-container {{
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }}
    .flag-icon {{
        font-size: 3.2rem;
        display: block;
        margin-bottom: -0.2rem;
    }}
    .main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, {GREEN}, {YELLOW}, {RED});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }}
    .sub-title {{
        font-size: 1.05rem;
        font-weight: 400;
        color: #2c3e50;
        margin-top: -0.2rem;
        letter-spacing: 0.3px;
        border-bottom: 2px solid {YELLOW};
        display: inline-block;
        padding-bottom: 0.3rem;
    }}
    .sub-title-2 {{
        font-size: 0.95rem;
        color: #34495e;
        margin-top: 0.2rem;
        font-weight: 300;
    }}

    /* ── Cards ── */
    .card {{
        background: white;
        padding: 1.8rem 2rem;
        border-radius: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid rgba(255,255,255,0.6);
        backdrop-filter: blur(2px);
        margin-bottom: 1.2rem;
        transition: box-shadow 0.25s ease;
    }}
    .card:hover {{
        box-shadow: 0 20px 50px rgba(0,0,0,0.08);
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, {GREEN}, {BLUE}) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 40px !important;
        padding: 0.6rem 2.2rem !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.4px;
        box-shadow: 0 6px 18px rgba(7, 137, 48, 0.25);
        transition: all 0.25s ease !important;
        width: 100%;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 28px rgba(7, 137, 48, 0.35);
        background: linear-gradient(135deg, #0a9e3a, {BLUE}) !important;
    }}

    /* ── Swap button ── */
    .swap-btn > button {{
        background: {DARK} !important;
        color: white !important;
        font-weight: 500 !important;
        border-radius: 40px !important;
        padding: 0.3rem 1.2rem !important;
        font-size: 0.85rem !important;
        box-shadow: none !important;
        border: 1px solid #444 !important;
        background: #2c3e50 !important;
    }}
    .swap-btn > button:hover {{
        background: #1a2a3a !important;
        transform: scale(1.02);
    }}

    /* ── Text input ── */
    .stTextArea textarea {{
        border-radius: 18px !important;
        border: 1.5px solid #e0e4e8 !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem !important;
        background: #fafbfc !important;
        transition: border 0.2s ease;
    }}
    .stTextArea textarea:focus {{
        border-color: {GREEN} !important;
        box-shadow: 0 0 0 3px rgba(7, 137, 48, 0.12);
    }}

    /* ── Radio buttons ── */
    .stRadio > div {{
        gap: 0.8rem;
    }}
    .stRadio label {{
        font-weight: 500;
        color: #1e293b;
        background: #f1f4f8;
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        border: 1px solid transparent;
        transition: all 0.15s ease;
    }}
    .stRadio label:hover {{
        background: #e2e8f0;
    }}
    .stRadio [data-baseweb="radio"]:checked + label {{
        background: {GREEN};
        color: white;
        border-color: {GREEN};
    }}

    /* ── Success box ── */
    .stSuccess {{
        background: #eafaf1 !important;
        border-radius: 18px !important;
        padding: 1.2rem 1.6rem !important;
        border-left: 5px solid {GREEN} !important;
        box-shadow: 0 2px 8px rgba(7, 137, 48, 0.06);
    }}

    /* ── Example buttons ── */
    .ex-btn > button {{
        background: white !important;
        color: #1e293b !important;
        border: 1px solid #dce0e5 !important;
        border-radius: 30px !important;
        padding: 0.25rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }}
    .ex-btn > button:hover {{
        background: {GREEN} !important;
        color: white !important;
        border-color: {GREEN} !important;
        transform: scale(1.02);
    }}

    /* ── Footer ── */
    .footer {{
        text-align: center;
        padding: 1.5rem 1rem;
        margin-top: 2.5rem;
        border-radius: 30px;
        background: linear-gradient(135deg, {GREEN}, {YELLOW}, {RED});
        color: white;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }}
    .footer small {{
        font-weight: 300;
        opacity: 0.9;
    }}

    /* ── Divider ── */
    hr {{
        margin: 1.8rem 0;
        border: 0;
        height: 2px;
        background: linear-gradient(to right, {GREEN}, {YELLOW}, {RED});
        opacity: 0.3;
        border-radius: 10px;
    }}

    /* ── Responsive tweaks ── */
    @media (max-width: 600px) {{
        .main-title {{
            font-size: 2rem;
        }}
        .card {{
            padding: 1.2rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
AMH = "amh_Ethi"
ORM = "orm_Latn"

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    base = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model_amh_orm = PeftModel.from_pretrained(base, "fetle/amh-orm-nmt")
    base2 = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model_orm_amh = PeftModel.from_pretrained(base2, "fetle/orm-amh-nmt")
    return tokenizer, model_amh_orm, model_orm_amh

tokenizer, model_amh_orm, model_orm_amh = load_models()

# ============================================================
# TRANSLATION FUNCTION
# ============================================================
def translate_text(text, direction):
    if not text or text.strip() == "":
        return "⚠️ Please enter some text to translate."
    
    if direction == "Amharic → Oromo":
        model = model_amh_orm
        src_lang, tgt_lang = AMH, ORM
    else:
        model = model_orm_amh
        src_lang, tgt_lang = ORM, AMH
    
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================
# UI — PROFESSIONAL LAYOUT
# ============================================================
st.markdown("""
<div class="title-container">
    <span class="flag-icon">🇪🇹</span>
    <div class="main-title">Amharic ↔ Oromo Translator</div>
    <div class="sub-title">ለኢትዮጵያውያን የተዘጋጀ የትርጉም አገልግሎት</div>
    <div class="sub-title-2">Tajaajila hiikuu kan Etiyophiyaaf qophaaʼe</div>
</div>
""", unsafe_allow_html=True)

# ── Card: Direction + Swap ──
with st.container():
    col_dir, col_swap = st.columns([5, 1])
    with col_dir:
        direction = st.radio(
            "🔄 Select Direction",
            ["Amharic → Oromo", "Oromo → Amharic"],
            horizontal=True,
            key="direction"
        )
    with col_swap:
        st.write("")
        st.write("")
        if st.button("⇄ Swap", key="swap_btn", help="Swap translation direction"):
            if direction == "Amharic → Oromo":
                direction = "Oromo → Amharic"
            else:
                direction = "Amharic → Oromo"

# ── Card: Input ──
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    text = st.text_area(
        "📝 Enter Your Text",
        height=120,
        placeholder="Type your sentence here... ዓረፍተ ነገር ይጻፉ... Himoota galchi...",
        key="input_text"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Translate Button ──
col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
with col_btn2:
    translate_clicked = st.button("🔮 Translate", type="primary", use_container_width=True)

# ── Result ──
if translate_clicked:
    if text:
        with st.spinner("Translating... እየተረጎመ... Hiikuu..."):
            result = translate_text(text, direction)
        st.success(f"✅ **Translation / ትርጉም / Hiikuu:**\n\n{result}")
    else:
        st.warning("⚠️ Please enter some text to translate.")

# ── Examples ──
st.markdown("---")
st.markdown("### 📌 Try These Examples")

cols = st.columns(3)

examples = [
    ("ሰላም እንዴት ነህ?", "Amharic → Oromo"),
    ("እናመሰግናለን", "Amharic → Oromo"),
    ("አዲስ አበባ", "Amharic → Oromo"),
    ("nagaa dha, akkam?", "Oromo → Amharic"),
    ("galatoomaa", "Oromo → Amharic"),
    ("Finfinnee", "Oromo → Amharic"),
    ("በቃ እንሂድ", "Amharic → Oromo"),
    ("haa deemnu", "Oromo → Amharic"),
    ("ሰላም", "Amharic → Oromo"),
]

for i, (example_text, example_dir) in enumerate(examples):
    with cols[i % 3]:
        st.markdown('<div class="ex-btn">', unsafe_allow_html=True)
        if st.button(example_text, key=f"ex_{i}"):
            text = example_text
            direction = example_dir
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown(f"""
<div class="footer">
    🇪🇹 <b>ኢትዮጵያ ትበልጭ! · Etiyophiyaa Injifattu! · Ethiopia Triumphs!</b> 🇪🇹<br>
    <small>Model: NLLB-200 (600M) + LoRA &nbsp;|&nbsp; Training: 5,000 sentences &nbsp;|&nbsp; Best chrF: 27.39</small>
</div>
""", unsafe_allow_html=True)
