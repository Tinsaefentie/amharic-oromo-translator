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

DIR_AM_TO_OR = "Amharic → Afaan Oromo"
DIR_OR_TO_AM = "Afaan Oromo → Amharic"

DIRECTIONS = {
    DIR_AM_TO_OR: {"src": AMH, "tgt": ORM, "adapter": LORA_AMH_TO_ORM,
                   "src_label": "Amharic", "src_native": "አማርኛ",
                   "tgt_label": "Afaan Oromo", "tgt_native": "Afaan Oromoo"},
    DIR_OR_TO_AM: {"src": ORM, "tgt": AMH, "adapter": LORA_ORM_TO_AMH,
                   "src_label": "Afaan Oromo", "src_native": "Afaan Oromoo",
                   "tgt_label": "Amharic", "tgt_native": "አማርኛ"},
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
    page_icon="🌐",
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
# 4. DESIGN SYSTEM
# ----------------------------------------------------------------------------
# Ink-charcoal surface, a single working accent (deep emerald) for action and
# state, and an antique-gold hairline reserved for one signature moment
# (the top rule + the language-pair seam). Crimson is kept out of decoration
# entirely and used only as the semantic error color. Newsreader (serif)
# carries the institutional register for the Latin display type; Noto Serif
# Ethiopic mirrors that weight for Ge'ez. Inter runs the interface; JetBrains
# Mono is reserved for anything that reads as data — language codes, metrics,
# model identifiers — to signal "this is a technical system," not a poster.
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Noto+Serif+Ethiopic:wght@400;500;600&family=Noto+Sans+Ethiopic:wght@400;500;600&display=swap');

:root {
    --ink: #0B0D11;
    --surface: #12151B;
    --surface-raised: #171A21;
    --hairline: rgba(255,255,255,0.08);
    --hairline-strong: rgba(255,255,255,0.16);
    --text: #E7E9EE;
    --text-muted: #8B93A3;
    --text-faint: #575F6E;
    --emerald: #14804A;
    --emerald-strong: #1C9A5A;
    --emerald-wash: rgba(20,128,74,0.12);
    --gold: #C8A44D;
    --gold-wash: rgba(200,164,77,0.10);
    --crimson: #B4453F;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.stApp {
    background: var(--ink);
    color: var(--text);
}

.block-container { padding-top: 1.6rem; max-width: 1180px; }

.ethiopic { font-family: 'Noto Serif Ethiopic', 'Noto Sans Ethiopic', serif; }
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }

/* ---------- Header ---------- */

.top-rule {
    height: 2px;
    width: 100%;
    background: var(--gold);
    opacity: 0.55;
    margin-bottom: 1.4rem;
}

.masthead-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
}

.eyebrow-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    text-align: right;
}

.masthead-title {
    font-family: 'Noto Serif Ethiopic', serif;
    font-size: 2.15rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.25;
    margin: 0.1rem 0 0 0;
}

.masthead-subtitle {
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-size: 1.25rem;
    font-weight: 400;
    color: var(--text-muted);
    margin: 0.1rem 0 0.7rem 0;
}

.masthead-desc {
    font-size: 0.92rem;
    color: var(--text-muted);
    max-width: 620px;
    line-height: 1.55;
    margin-bottom: 1.1rem;
}

.header-divider {
    border-bottom: 1px solid var(--hairline);
    margin-bottom: 1.3rem;
}

/* ---------- Segmented direction control ---------- */

div[role="radiogroup"] {
    display: inline-flex;
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
}

div[role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.45rem 1rem !important;
    margin: 0 !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    transition: all 0.12s ease;
}

div[role="radiogroup"] label:has(input:checked) {
    background: var(--emerald-wash) !important;
    color: var(--text) !important;
    box-shadow: inset 0 0 0 1px rgba(20,128,74,0.45);
}

div[role="radiogroup"] label div p { font-size: 0.85rem !important; font-weight: 500; }

/* ---------- Buttons ---------- */

.stButton>button {
    border-radius: 8px !important;
    border: 1px solid var(--hairline-strong) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    box-shadow: none !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}
.stButton>button:hover {
    border-color: var(--text-muted) !important;
    background: var(--surface-raised) !important;
    color: var(--text) !important;
}
.stButton>button:active { transform: none !important; }

.stButton>button[kind="primary"] {
    background: var(--emerald) !important;
    border: 1px solid var(--emerald) !important;
    color: #F3FBF6 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.62rem 1.2rem !important;
}
.stButton>button[kind="primary"]:hover {
    background: var(--emerald-strong) !important;
    border-color: var(--emerald-strong) !important;
}

.example-btn button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    text-align: left !important;
    justify-content: flex-start !important;
}

/* ---------- Pipeline seam (signature element) ---------- */

.seam {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin: 0.9rem 0 1.1rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--text-faint);
    letter-spacing: 0.02em;
}
.seam .code { color: var(--gold); }
.seam .dot { width: 3px; height: 3px; border-radius: 50%; background: var(--text-faint); }
.seam::before, .seam::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--hairline);
}

/* ---------- Panels ---------- */

.panel {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    padding: 1rem 1.15rem 0.9rem 1.15rem;
}

.panel-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.7rem;
}

.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-faint);
}

.panel-lang {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
}
.panel-lang .native { color: var(--text-muted); font-weight: 400; margin-left: 0.35rem; }

.metrics-row {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    margin-top: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-faint);
}
.metrics-row b { color: var(--text-muted); font-weight: 600; }

.output-body {
    min-height: 168px;
    border-radius: 6px;
    background: var(--ink);
    border: 1px solid var(--hairline);
    padding: 0.85rem 0.95rem;
    font-size: 1rem;
    line-height: 1.65;
    color: var(--text);
    white-space: pre-wrap;
    word-wrap: break-word;
}
.output-empty { color: var(--text-faint); font-style: italic; font-size: 0.88rem; }

textarea, .stTextArea textarea {
    background: var(--ink) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    border: 1px solid var(--hairline) !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--emerald-strong) !important;
    box-shadow: 0 0 0 1px var(--emerald-strong) !important;
}

.copy-btn {
    background: transparent;
    border: 1px solid var(--hairline-strong);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    padding: 0.32rem 0.7rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.7rem;
    letter-spacing: 0.03em;
}
.copy-btn:hover { border-color: var(--gold); color: var(--gold); }

/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] {
    background: var(--surface-raised);
    border-right: 1px solid var(--hairline);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

.side-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-bottom: 0.7rem;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    padding: 0.42rem 0;
    border-bottom: 1px solid var(--hairline);
    font-size: 0.82rem;
}
.spec-row:last-child { border-bottom: none; }
.spec-row .k { color: var(--text-muted); }
.spec-row .v { font-family: 'JetBrains Mono', monospace; color: var(--gold); font-size: 0.78rem; }

.status-line {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.9rem;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--emerald-strong); }
.status-dot.cpu { background: var(--gold); }

/* ---------- Footer ---------- */

.footer-note {
    text-align: center;
    color: var(--text-faint);
    font-size: 0.72rem;
    letter-spacing: 0.03em;
    margin-top: 2.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--hairline);
}

/* Streamlit chrome cleanup */
[data-testid="stExpander"] {
    border: 1px solid var(--hairline) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}
#MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 5. HEADER
# ============================================================================

st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="masthead-row">
        <span class="eyebrow">National Language Technology · Research Preview</span>
        <span class="eyebrow-right">NLLB&#8209;200 + LoRA &nbsp;·&nbsp; chrF 27.39</span>
    </div>
    <p class="masthead-title">የኢትዮጵያ ቋንቋዎች ትርጉም መድረክ</p>
    <p class="masthead-subtitle">Tajaajila Hiika Afaanii</p>
    <p class="masthead-desc">
        A bidirectional neural machine translation system for Amharic and
        Afaan Oromo, built on NLLB-200 and adapted with low-rank fine-tuning
        for two of Ethiopia's most widely spoken, low-resource languages.
    </p>
    <div class="header-divider"></div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# 6. DIRECTION CONTROLS
# ============================================================================

ctrl_left, ctrl_right = st.columns([3, 1])

with ctrl_left:
    st.radio(
        "Translation direction",
        options=[DIR_AM_TO_OR, DIR_OR_TO_AM],
        key="direction",
        horizontal=True,
        label_visibility="collapsed",
    )

with ctrl_right:
    swap_col, clear_col = st.columns(2)
    with swap_col:
        st.button("⇄ Swap", on_click=swap_direction, use_container_width=True)
    with clear_col:
        st.button("Clear", on_click=clear_input, use_container_width=True)

current_config = DIRECTIONS[st.session_state.direction]

st.markdown(
    f"""
    <div class="seam">
        <span class="code">{current_config['src']}</span>
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        <span class="code">{current_config['tgt']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<span style="font-family:JetBrains Mono, monospace; font-size:0.7rem; '
    'letter-spacing:0.08em; text-transform:uppercase; color:#575F6E;">Examples</span>',
    unsafe_allow_html=True,
)
example_cols = st.columns(len(EXAMPLES[st.session_state.direction]))
for i, example_text in enumerate(EXAMPLES[st.session_state.direction]):
    with example_cols[i]:
        st.markdown('<div class="example-btn">', unsafe_allow_html=True)
        label = example_text if len(example_text) <= 34 else example_text[:32] + "…"
        st.button(
            label,
            key=f"example_{st.session_state.direction}_{i}",
            on_click=set_example,
            args=(st.session_state.direction, example_text),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ============================================================================
# 7. TRANSLATION WORKSPACE
# ============================================================================

src_col, tgt_col = st.columns(2, gap="medium")

with src_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel-head">
            <div>
                <div class="panel-label">Source</div>
                <div class="panel-lang">{current_config['src_label']}
                    <span class="native ethiopic">{current_config['src_native']}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_area(
        "source_input",
        key="input_text",
        height=170,
        placeholder="Enter text to translate…",
        label_visibility="collapsed",
    )
    char_count = len(st.session_state.input_text)
    word_count = len(st.session_state.input_text.split()) if st.session_state.input_text.strip() else 0
    st.markdown(
        f"""
        <div class="metrics-row">
            <span>Characters&nbsp;<b>{char_count}</b></span>
            <span>Words&nbsp;<b>{word_count}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.button(
        "Translate",
        on_click=do_translate,
        use_container_width=True,
        type="primary",
    )

with tgt_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel-head">
            <div>
                <div class="panel-label">Output</div>
                <div class="panel-lang">{current_config['tgt_label']}
                    <span class="native ethiopic">{current_config['tgt_native']}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_error:
        st.error(f"Translation failed — {st.session_state.last_error}")
    elif st.session_state.output_text:
        safe_output = html_lib.escape(st.session_state.output_text)
        st.markdown(f'<div class="output-body">{safe_output}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="output-body output-empty">Translation will appear here.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.output_text:
        copy_payload = json.dumps(st.session_state.output_text)
        components.html(
            f"""
            <button class="copy-btn" id="copyBtn" onclick="copyText()">COPY</button>
            <script>
            function copyText() {{
                const text = {copy_payload};
                navigator.clipboard.writeText(text).then(() => {{
                    const btn = document.getElementById('copyBtn');
                    const original = btn.innerText;
                    btn.innerText = 'COPIED';
                    setTimeout(() => {{ btn.innerText = original; }}, 1400);
                }});
            }}
            </script>
            <style>
                body {{ margin: 0; background: transparent; }}
                .copy-btn {{
                    background: transparent;
                    border: 1px solid rgba(255,255,255,0.16);
                    color: #8B93A3;
                    font-family: 'JetBrains Mono', monospace;
                    font-weight: 500;
                    padding: 0.32rem 0.7rem;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 0.7rem;
                    letter-spacing: 0.05em;
                    margin-top: 8px;
                }}
                .copy-btn:hover {{ border-color: #C8A44D; color: #C8A44D; }}
            </style>
            """,
            height=44,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# 8. SIDEBAR — Institutional metadata / project impact
# ============================================================================

with st.sidebar:
    st.markdown('<div class="side-eyebrow">Model Card</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spec-row"><span class="k">Base model</span><span class="v">NLLB-200 · 600M</span></div>
        <div class="spec-row"><span class="k">Fine-tuning</span><span class="v">LoRA (PEFT)</span></div>
        <div class="spec-row"><span class="k">Dataset</span><span class="v">5,000 pairs</span></div>
        <div class="spec-row"><span class="k">Best chrF</span><span class="v">27.39</span></div>
        <div class="spec-row"><span class="k">Directions</span><span class="v">AMH ⇄ ORM</span></div>
        """,
        unsafe_allow_html=True,
    )

    device_is_cuda = torch.cuda.is_available()
    st.markdown(
        f"""
        <div class="status-line">
            <span class="status-dot {'cuda' if device_is_cuda else 'cpu'}"></span>
            <span>{'GPU · CUDA' if device_is_cuda else 'CPU inference'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander("About this project"):
        st.markdown(
            """
            Amharic and Afaan Oromo are among Ethiopia's most widely spoken
            languages, yet remain low-resource in NLP research. This platform
            explores adapter-based fine-tuning of a multilingual translation
            model to support cross-lingual communication between the two
            languages, without retraining the full base model.
            """
        )

    with st.expander("Technical specification"):
        st.markdown(
            f"""
            - **Base model:** `{BASE_MODEL}`
            - **Amharic → Oromo adapter:** `{LORA_AMH_TO_ORM}`
            - **Oromo → Amharic adapter:** `{LORA_ORM_TO_AMH}`
            - **Language codes:** `{AMH}`, `{ORM}`
            - **Device:** auto-detected (CUDA if available, else CPU)
            - **Decoding:** beam search, beams = 4, max length 256
            """
        )

# ============================================================================
# 9. FOOTER
# ============================================================================

st.markdown(
    """
    <div class="footer-note">
        ETHIOPIAN NATIONAL LANGUAGE TECHNOLOGY INITIATIVE &nbsp;—&nbsp; RESEARCH PREVIEW
    </div>
    """,
    unsafe_allow_html=True,
)
