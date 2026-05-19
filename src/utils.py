# Helper functions for Arabic NLP project

import re

# Normalize Arabic characters

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text


# Remove Arabic diacritics

def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                             ّ    | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ
                         """, re.VERBOSE)
    return re.sub(arabic_diacritics, '', text)

# Basic text cleaning

def clean_text(text):
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    return text