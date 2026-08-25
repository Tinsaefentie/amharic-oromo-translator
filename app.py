"""
የኢትዮጵያ ቋንቋዎች ትርጉም መድረክ | Tajaajila Hiika Afaanii
Bidirectional Amharic <-> Afaan Oromo Neural Machine Translation Platform

Stack (unchanged): facebook/nllb-200-distilled-600M base model,
PEFT/LoRA adapters (fetle/amh-orm-nmt, fetle/orm-amh-nmt),
PyTorch + Hugging Face Transformers, auto CUDA/CPU device selection.

Run with:  streamlit run app.py
"""

import json
import html as html_lib

import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# 1. CONFIGURATION — technical stack kept fully intact
# ============================================================================

BASE_MODEL = "facebook/nllb-200-distilled-600M"
LORA_AMH_TO_ORM = "fetle/amh-orm-nmt"   # Amharic -> Afaan Oromo adapter
LORA_ORM_TO_AMH = "fetle/orm-amh-nmt"   # Afaan Oromo -> Amharic adapter

AMH = "amh_Ethi"
ORM = "orm_Latn"

DIR_AM_TO_OR = "Amharic ➔ Afaan Oromo"
DIR_OR_TO_AM = "Afaan Oromo ➔ Amharic"

DIRECTIONS = {
    DIR_AM_TO_OR: {"src": AMH, "tgt": ORM, "adapter": LORA_AMH_TO_ORM,
                   "src_label": "አማርኛ / Amharic", "tgt_label": "አፋን ኦሮሞ / Afaan Oromo"},
    DIR_OR_TO_AM: {"src": ORM, "tgt": AMH, "adapter": LORA_ORM_TO_AMH,
                   "src_label": "አፋን ኦሮሞ / Afaan Oromo", "tgt_label": "አማርኛ / Amharic"},
}

EXAMPLES = {
    DIR_AM_TO_OR: [
        "ሰላም! እንደምን አለህ?",
        "ኢትዮጵያ በብዙ ብሔር ብሔረሰቦች የበለጸገች ሀገር ናት።",
        "ትምህርት የስኬት መሠረት ነው።",
    ],
    DIR_OR_TO_AM: [
        "Akkam jirta? Nagaa?",
        "Itoophiyaan biyya sabaa fi sablammoota baayʼeen badhaadhe dha.",
        "Barnoonni hundee milkaaʼinaa ti.",
    ],
}

st.set_page_config(
    page_title="የኢትዮጵያ ቋንቋዎች ትርጉም መድረክ | Tajaajila Hiika Afaanii",
    page_icon="🇪🇹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 2. MODEL LOADING (cached) — PyTorch + Transformers + PEFT
# ============================================================================


@st.cache_resource(show_spinner=False)
def load_pipeline(adapter_id: str):
    """Load base NLLB model + tokenizer + LoRA adapter, cached per adapter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _lang_token_id(tokenizer, lang_code: str) -> int:
    """Resolve an NLLB language-code token id across tokenizer versions."""
    lang_map = getattr(tokenizer, "lang_code_to_id", None)
    if lang_map and lang_code in lang_map:
        return lang_map[lang_code]
    return tokenizer.convert_tokens_to_ids(lang_code)


def run_translation(text: str, src_lang: str, tgt_lang: str, adapter_id: str,
                     max_length: int = 256, num_beams: int = 4) -> str:
    tokenizer, model, device = load_pipeline(adapter_id)
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    forced_bos_token_id = _lang_token_id(tokenizer, tgt_lang)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()


# ============================================================================
# 3. SESSION STATE
# ============================================================================

def init_state():
    defaults = {
        "direction": DIR_AM_TO_OR,
        "input_text": "",
        "output_text": "",
        "last_error": "",
        "is_translating": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def swap_direction():
    st.session_state.direction = (
        DIR_OR_TO_AM if st.session_state.direction == DIR_AM_TO_OR else DIR_AM_TO_OR
    )
    st.session_state.input_text, st.session_state.output_text = (
        st.session_state.output_text,
        st.session_state.input_text,
    )
    st.session_state.last_error = ""


def set_example(direction: str, text: str):
    st.session_state.direction = direction
    st.session_state.input_text = text
    st.session_state.output_text = ""
    st.session_state.last_error = ""


def clear_input():
    st.session_state.input_text = ""
    st.session_state.output_text = ""
    st.session_state.last_error = ""


def do_translate():
    text = st.session_state.input_text.strip()
    if not text:
        st.session_state.output_text = ""
        st.session_state.last_error = ""
        return
    config = DIRECTIONS[st.session_state.direction]
    st.session_state.last_error = ""
    try:
        st.session_state.output_text = run_translation(
            text, config["src"], config["tgt"], config["adapter"]
        )
    except Exception as exc:  # surfaced in the UI, not swallowed
        st.session_state.output_text = ""
        st.session_state.last_error = f"{type(exc).__name__}: {exc}"


# ============================================================================
# 4. CUSTOM CSS — Ethiopian premium theme, glassmorphism, gradient accents
# ============================================================================

CUSTOM_CSS = """
<style>
:root {
    --emerald: #078930;
    --gold: #FCDD09;
    --crimson: #DA121A;
    --navy: #1a1a2e;
    --navy-light: #23233f;
    --glass-border: rgba(252, 221, 9, 0.25);
}

.stApp {
    background: radial-gradient(circle at 15% 10%, #20203a 0%, var(--navy) 55%, #12121f 100%);
    color: #f4f4f8;
}

/* Flag-gradient top bar */
.flag-bar {
    height: 6px;
    width: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--emerald) 0%, var(--gold) 50%, var(--crimson) 100%);
    margin-bottom: 1.1rem;
    box-shadow: 0 0 18px rgba(252, 221, 9, 0.35);
}

.hero-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.4rem;
}

.hero-title {
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, var(--gold), #fff6c9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #c9c9d8;
    margin-top: 0.35rem;
}

.badge-row { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }

.badge {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
}
.badge-emerald { background: rgba(7,137,48,0.18); color: #6be396; border-color: rgba(7,137,48,0.5); }
.badge-gold { background: rgba(252,221,9,0.14); color: var(--gold); border-color: rgba(252,221,9,0.45); }
.badge-crimson { background: rgba(218,18,26,0.16); color: #ff7b7f; border-color: rgba(218,18,26,0.45); }

/* Glassmorphism translation cards */
.glass-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.2rem 1.3rem 0.9rem 1.3rem;
    backdrop-filter: blur(8px);
    position: relative;
}

.card-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--gold);
    margin-bottom: 0.55rem;
}

.metrics-row {
    display: flex;
    gap: 1.1rem;
    margin-top: 0.5rem;
    font-size: 0.78rem;
    color: #9d9dc0;
}
.metrics-row span b { color: #e6e6f0; }

.output-box {
    min-height: 160px;
    border-radius: 12px;
    border: 1px dashed rgba(252,221,9,0.3);
    background: rgba(0,0,0,0.18);
    padding: 0.9rem 1rem;
    font-size: 1.02rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.output-placeholder { color: #7a7a95; font-style: italic; }

/* Buttons */
.stButton>button {
    border-radius: 10px !important;
    border: 1px solid var(--glass-border) !important;
    background: linear-gradient(135deg, rgba(7,137,48,0.85), rgba(7,137,48,0.55)) !important;
    color: #fff !important;
    font-weight: 600 !important;
    transition: all 0.15s ease-in-out;
}
.stButton>button:hover {
    border-color: var(--gold) !important;
    box-shadow: 0 0 14px rgba(252,221,9,0.35);
    transform: translateY(-1px);
}

.swap-btn button {
    background: linear-gradient(135deg, rgba(252,221,9,0.85), rgba(218,18,26,0.55)) !important;
}

.clear-btn button {
    background: linear-gradient(135deg, rgba(218,18,26,0.75), rgba(26,26,46,0.4)) !important;
}

.copy-btn {
    margin-top: 0.7rem;
    background: linear-gradient(135deg, rgba(7,137,48,0.9), rgba(252,221,9,0.35));
    border: 1px solid var(--glass-border);
    color: #fff;
    font-weight: 600;
    padding: 0.45rem 1rem;
    border-radius: 10px;
    cursor: pointer;
    font-size: 0.85rem;
}
.copy-btn:hover { box-shadow: 0 0 12px rgba(252,221,9,0.35); }

.example-chip {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    margin: 0.15rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--glass-border);
    font-size: 0.78rem;
    color: #d8d8e8;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14142a 0%, #1a1a2e 100%);
    border-right: 1px solid var(--glass-border);
}

.sidebar-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.sidebar-metric { display: flex; justify-content: space-between; font-size: 0.85rem; margin: 0.3rem 0; }
.sidebar-metric span:last-child { color: var(--gold); font-weight: 700; }

.footer-note {
    text-align: center;
    color: #7a7a95;
    font-size: 0.75rem;
    margin-top: 2.2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

textarea, .stTextArea textarea {
    background: rgba(0,0,0,0.22) !important;
    color: #f4f4f8 !important;
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
}

div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--glass-border);
    padding: 0.4rem 0.9rem;
    border-radius: 10px;
    margin-right: 0.4rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 5. HEADER
# ============================================================================

st.markdown('<div class="flag-bar"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-card">
        <p class="hero-title">የኢትዮጵያ ቋንቋዎች ትርጉም መድረክ</p>
        <p class="hero-title" style="font-size:1.15rem; opacity:0.85;">Tajaajila Hiika Afaanii</p>
        <p class="hero-subtitle">
            A neural machine translation platform bridging Amharic and Afaan Oromo —
            built on NLLB-200 with adapter-based fine-tuning for low-resource Ethiopian languages.
        </p>
        <div class="badge-row">
            <span class="badge badge-emerald">NLLB-200 Distilled 600M</span>
            <span class="badge badge-gold">LoRA Fine-Tuned</span>
            <span class="badge badge-crimson">chrF 27.39</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# 6. DIRECTION CONTROLS
# ============================================================================

ctrl_left, ctrl_mid, ctrl_right = st.columns([5, 1, 2])

with ctrl_left:
    st.radio(
        "Translation direction / Kallattii Hiikkaa",
        options=[DIR_AM_TO_OR, DIR_OR_TO_AM],
        key="direction",
        horizontal=True,
        label_visibility="collapsed",
    )

with ctrl_mid:
    st.markdown('<div class="swap-btn">', unsafe_allow_html=True)
    st.button("⇄ Swap", on_click=swap_direction, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ctrl_right:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    st.button("🧹 አጽዳ / Clear", on_click=clear_input, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

current_config = DIRECTIONS[st.session_state.direction]

# Example chips
st.markdown(
    f'<span style="font-size:0.8rem; color:#9d9dc0;">Try an example / Fakkeenya:</span>',
    unsafe_allow_html=True,
)
example_cols = st.columns(len(EXAMPLES[st.session_state.direction]))
for i, example_text in enumerate(EXAMPLES[st.session_state.direction]):
    with example_cols[i]:
        st.button(
            example_text if len(example_text) <= 28 else example_text[:26] + "…",
            key=f"example_{st.session_state.direction}_{i}",
            on_click=set_example,
            args=(st.session_state.direction, example_text),
            use_container_width=True,
        )

st.write("")

# ============================================================================
# 7. TRANSLATION WORKSPACE
# ============================================================================

src_col, tgt_col = st.columns(2, gap="large")

with src_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-label">{current_config["src_label"]}</div>', unsafe_allow_html=True)
    st.text_area(
        "source_input",
        key="input_text",
        height=180,
        placeholder="ጽሑፍዎን እዚህ ይጻፉ... / Barreeffama kee asitti barreessi...",
        label_visibility="collapsed",
    )
    char_count = len(st.session_state.input_text)
    word_count = len(st.session_state.input_text.split()) if st.session_state.input_text.strip() else 0
    st.markdown(
        f"""
        <div class="metrics-row">
            <span>ፊደላት / Characters: <b>{char_count}</b></span>
            <span>ቃላት / Words: <b>{word_count}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.button(
        "🌐 ተርጉም / Hiiki (Translate)",
        on_click=do_translate,
        use_container_width=True,
        type="primary",
    )

with tgt_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-label">{current_config["tgt_label"]}</div>', unsafe_allow_html=True)

    if st.session_state.last_error:
        st.error(f"Translation failed: {st.session_state.last_error}")
    elif st.session_state.output_text:
        safe_output = html_lib.escape(st.session_state.output_text)
        st.markdown(f'<div class="output-box">{safe_output}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="output-box output-placeholder">ትርጉምዎ እዚህ ይታያል... / Hiikni kee asitti ni argama...</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.output_text:
        copy_payload = json.dumps(st.session_state.output_text)
        components.html(
            f"""
            <button class="copy-btn" onclick="copyText()">📋 Copy to clipboard</button>
            <script>
            function copyText() {{
                const text = {copy_payload};
                navigator.clipboard.writeText(text).then(() => {{
                    const btn = document.querySelector('.copy-btn');
                    const original = btn.innerText;
                    btn.innerText = '✅ Copied!';
                    setTimeout(() => {{ btn.innerText = original; }}, 1500);
                }});
            }}
            </script>
            <style>
                .copy-btn {{
                    background: linear-gradient(135deg, rgba(7,137,48,0.9), rgba(252,221,9,0.35));
                    border: 1px solid rgba(252,221,9,0.25);
                    color: #fff;
                    font-weight: 600;
                    padding: 0.45rem 1rem;
                    border-radius: 10px;
                    cursor: pointer;
                    font-size: 0.85rem;
                    font-family: sans-serif;
                }}
                .copy-btn:hover {{ box-shadow: 0 0 12px rgba(252,221,9,0.35); }}
                body {{ margin: 0; background: transparent; }}
            </style>
            """,
            height=50,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# 8. SIDEBAR — Institutional metadata / project impact
# ============================================================================

with st.sidebar:
    st.markdown("### 🇪🇹 Project Overview")
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-metric"><span>Base Model</span><span>NLLB-200 (600M)</span></div>
        <div class="sidebar-metric"><span>Fine-Tuning</span><span>LoRA (PEFT)</span></div>
        <div class="sidebar-metric"><span>Dataset Size</span><span>5,000 pairs</span></div>
        <div class="sidebar-metric"><span>Best chrF</span><span>27.39</span></div>
        <div class="sidebar-metric"><span>Directions</span><span>AMH ⇄ ORM</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📖 Why this matters / Bu'aa Hawaasaaf"):
        st.markdown(
            """
            Amharic and Afaan Oromo are among Ethiopia's most widely spoken
            languages, yet remain low-resource in NLP research. This platform
            was developed as part of a research internship to explore
            adapter-based fine-tuning of multilingual translation models —
            supporting cross-lingual communication and national language
            integration at scale, without retraining the full base model.
            """
        )

    with st.expander("⚙️ Technical Details"):
        st.markdown(
            f"""
            - **Base model:** `{BASE_MODEL}`
            - **Amharic → Oromo adapter:** `{LORA_AMH_TO_ORM}`
            - **Oromo → Amharic adapter:** `{LORA_ORM_TO_AMH}`
            - **Language codes:** `{AMH}`, `{ORM}`
            - **Device:** Auto-detected (CUDA if available, else CPU)
            - **Decoding:** Beam search (beams=4), max length 256
            """
        )

    device_label = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟡 CPU"
    st.caption(f"Runtime device: {device_label}")

# ============================================================================
# 9. FOOTER
# ============================================================================

st.markdown(
    """
    <div class="footer-note">
        የኢትዮጵያ ቋንቋዎች ትርጉም መድረክ · Tajaajila Hiika Afaanii<br>
        Built on open multilingual research to support Ethiopia's linguistic diversity.
    </div>
    """,
    unsafe_allow_html=True,
)
