import os
import tempfile
from PIL import Image
import pytesseract
from gtts import gTTS
from langdetect import detect
import streamlit as st
from functools import lru_cache

# Check if Hindi is installed
def is_language_installed(lang_code='hin'):
    tessdata_dir = os.environ.get('TESSDATA_PREFIX', '/usr/share/tesseract-ocr/4.00/tessdata')
    return os.path.exists(os.path.join(tessdata_dir, f'{lang_code}.traineddata'))

hindi_installed = is_language_installed('hin')

# Braille character map (same as yours, truncated for brevity here)
braille_map = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
    'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
    'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽',
    'z': '⠵',
    ' ': ' ', '\n': '\n', ',': '⠂', '.': '⠲', '?': '⠦', '!': '⠖',

    # Hindi vowels, consonants, and other signs
    "अ": "⠁", "आ": "⠡", "इ": "⠊", "ई": "⠒", "उ": "⠥",
    "ऊ": "⠳", "ए": "⠑", "ऐ": "⠣", "ओ": "⠕", "औ": "⠷",
    "ऋ": "⠗", "क": "⠅", "ख": "⠩", "ग": "⠛", "घ": "⠣",
    "ङ": "⠻", "च": "⠉", "छ": "⠡", "ज": "⠚", "झ": "⠒",
    "ञ": "⠱", "ट": "⠞", "ठ": "⠾", "ड": "⠙", "ढ": "⠹",
    "ण": "⠻", "त": "⠞", "थ": "⠮", "द": "⠙", "ध": "⠹",
    "न": "⠝", "प": "⠏", "फ": "⠟", "ब": "⠃", "भ": "⠫",
    "म": "⠍", "य": "⠽", "र": "⠗", "ल": "⠇", "व": "⠧",
    "श": "⠱", "ष": "⠳", "स": "⠎", "ह": "⠓", "क्ष": "⠟",
    "ज्ञ": "⠻", "ड़": "⠚", "ढ़": "⠚", "फ़": "⠋", "ज़": "⠵",
    "ग्य": "⠛⠽", "त्र": "⠞⠗", "श्र": "⠱⠗",

    "ा": "⠡", "ि": "⠊", "ी": "⠒", "ु": "⠥", "ू": "⠳",
    "े": "⠑", "ै": "⠣", "ो": "⠕", "ौ": "⠷", "ृ": "⠗",

    "्": "⠄", "ं": "⠈", "ः": "⠘", "ँ": "⠨",

    "०": "⠚", "१": "⠁", "२": "⠃", "३": "⠉", "४": "⠙",
    "५": "⠑", "६": "⠋", "७": "⠛", "८": "⠓", "९": "⠊",

    "।": "⠲", ",": "⠂", "?": "⠦", "!": "⠖", "\"": "⠶",
    "'": "⠄", ";": "⠆", ":": "⠒", ".": "⠲", "-": "⠤",
    "(": "⠶", ")": "⠶", "/": "⠌",

    "A": "⠁", "B": "⠃", "C": "⠉", "D": "⠙", "E": "⠑",
    "F": "⠋", "G": "⠛", "H": "⠓", "I": "⠊", "J": "⠚",
    "K": "⠅", "L": "⠇", "M": "⠍", "N": "⠝", "O": "⠕",
    "P": "⠏", "Q": "⠟", "R": "⠗", "S": "⠎", "T": "⠞",
    "U": "⠥", "V": "⠧", "W": "⠺", "X": "⠭", "Y": "⠽", "Z": "⠵",

}

@lru_cache(maxsize=128)
def text_to_braille(text):
    return ''.join(braille_map.get(ch, ' ') for ch in text)

def resize_image(image, max_size=(1024, 1024)):
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def main():
    st.title("🖼️ Image Text Assistive - Hindi & English OCR")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img = resize_image(img)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Set OCR languages based on availability
        lang_config = 'eng+hin' if hindi_installed else 'eng'

        extracted_text = pytesseract.image_to_string(img, lang=lang_config)

        if not hindi_installed:
            st.warning("⚠️ Hindi OCR is not enabled. Only English text will be extracted.\n\n"
                       "To enable Hindi, install it using:\n\n```sudo apt install tesseract-ocr-hin```")

        # Detect language
        try:
            detected_lang = detect(extracted_text)
        except:
            detected_lang = 'en'

        gtts_lang = 'hi' if detected_lang == 'hi' else 'en'

        # Braille conversion
        braille_body = text_to_braille(extracted_text)
        braille_prefix = '⠰⠓ ' if gtts_lang == 'hi' else '⠰⠑ '
        braille_text = braille_prefix + braille_body

        st.subheader("📝 Extracted Text")
        st.write(extracted_text)

        st.subheader("⠃ Braille Translation")
        st.text(braille_text)

        # Text to Speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tts = gTTS(text=extracted_text, lang=gtts_lang)
            tts.save(tmp_audio.name)
            st.audio(tmp_audio.name, format="audio/mp3")

if __name__ == "__main__":
    main()
