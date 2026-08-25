import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Page configuration
st.set_page_config(
    page_title="🇪🇹 Amharic-Oromo Translator",
    page_icon="🇪🇹",
    layout="centered"
)

# Language codes
AMH = "amh_Ethi"
ORM = "orm_Latn"

@st.cache_resource
def load_models():
    """Load tokenizer and both translation models."""
    print("📥 Loading tokenizer and models...")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    
    # Load Amharic → Oromo model
    base = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model_amh_orm = PeftModel.from_pretrained(base, "fetle/amh-orm-nmt")
    
    # Load Oromo → Amharic model
    base2 = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model_orm_amh = PeftModel.from_pretrained(base2, "fetle/orm-amh-nmt")
    
    print("✅ Models loaded successfully!")
    return tokenizer, model_amh_orm, model_orm_amh

# Load models
tokenizer, model_amh_orm, model_orm_amh = load_models()

def translate_text(text, direction):
    """Translate text between Amharic and Oromo."""
    if not text or text.strip() == "":
        return "⚠️ Please enter some text to translate."
    
    if direction == "Amharic → Oromo":
        model = model_amh_orm
        src_lang = AMH
        tgt_lang = ORM
    else:
        model = model_orm_amh
        src_lang = ORM
        tgt_lang = AMH
    
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )
    
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
# UI
# ============================================================

st.markdown("""
# 🇪🇹 Amharic ↔ Oromo Translator
### ለኢትዮጵያውያን የተዘጋጀ የትርጉም አገልግሎት · Tajaajila hiikuu kan Etiyophiyaaf qophaa'e
""")

# Direction selection
direction = st.radio(
    "🔄 Select Translation Direction",
    ["Amharic → Oromo", "Oromo → Amharic"],
    horizontal=True
)

# Input text
text = st.text_area(
    "📝 Enter Text",
    height=150,
    placeholder="Type your sentence here... ዓረፍተ ነገር ይጻፉ... Himoota galchi..."
)

# Translate button
if st.button("🔮 Translate", type="primary", use_container_width=True):
    if text:
        with st.spinner("Translating... እየተረጎመ... Hiikuu..."):
            result = translate_text(text, direction)
        st.success(f"✅ **Translation / ትርጉም / Hiikuu:**\n\n{result}")
    else:
        st.warning("⚠️ Please enter some text to translate.")

# Examples
st.markdown("---")
st.markdown("### 📝 Try These Examples")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Amharic → Oromo**")
    if st.button("ሰላም እንዴት ነህ?"):
        st.session_state.text = "ሰላም እንዴት ነህ?"
        st.session_state.direction = "Amharic → Oromo"
    if st.button("እናመሰግናለን"):
        st.session_state.text = "እናመሰግናለን"
        st.session_state.direction = "Amharic → Oromo"
    if st.button("አዲስ አበባ"):
        st.session_state.text = "አዲስ አበባ"
        st.session_state.direction = "Amharic → Oromo"

with col2:
    st.markdown("**Oromo → Amharic**")
    if st.button("nagaa dha, akkam?"):
        st.session_state.text = "nagaa dha, akkam?"
        st.session_state.direction = "Oromo → Amharic"
    if st.button("galatoomaa"):
        st.session_state.text = "galatoomaa"
        st.session_state.direction = "Oromo → Amharic"
    if st.button("Finfinnee"):
        st.session_state.text = "Finfinnee"
        st.session_state.direction = "Oromo → Amharic"

# Preserve session state
if "text" in st.session_state:
    st.session_state.text = st.session_state.text
if "direction" in st.session_state:
    st.session_state.direction = st.session_state.direction

# Footer
st.markdown("---")
st.caption("""
**Model:** NLLB-200 (600M) + LoRA | **Training:** 5,000 sentences | **Best chrF:** 27.39  
🇪🇹 ኢትዮጵያ ትበልጭ! · Etiyophiyaa Injifattu! · Ethiopia Triumphs!
""")
