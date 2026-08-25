import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="Amharic-Oromo Translator", layout="centered")

AMH = "amh_Ethi"
ORM = "orm_Latn"

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    base = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model_amh_orm = PeftModel.from_pretrained(base, "Tinsaefentie/amh-orm-nmt")
    base2 = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model_orm_amh = PeftModel.from_pretrained(base2, "Tinsaefentie/orm-amh-nmt")
    return tokenizer, model_amh_orm, model_orm_amh

tokenizer, model_amh_orm, model_orm_amh = load_models()

def translate_text(text, direction):
    if not text or text.strip() == "":
        return "Please enter some text."
    if direction == "Amharic → Oromo":
        model, src, tgt = model_amh_orm, AMH, ORM
    else:
        model, src, tgt = model_orm_amh, ORM, AMH
    tokenizer.src_lang = src
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt), max_length=64, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🇪🇹 Amharic ↔ Oromo Translator")
st.markdown("ለኢትዮጵያውያን የተዘጋጀ የትርጉም አገልግሎት · Tajaajila hiikuu kan Etiyophiyaaf qophaa'e")

direction = st.radio("🔄 Direction", ["Amharic → Oromo", "Oromo → Amharic"])
text = st.text_area("📝 Enter Text", height=150, placeholder="Type your sentence here...")

if st.button("🔮 Translate", type="primary"):
    if text:
        result = translate_text(text, direction)
        st.success(f"✅ Translation: {result}")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.caption("Model: NLLB-200 (600M) + LoRA | Training: 5K sentences | Best chrF: 27.39")
