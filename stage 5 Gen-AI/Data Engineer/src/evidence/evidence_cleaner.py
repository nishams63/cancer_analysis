"""Evidence text cleaning and whitespace normalization."""
import re

class EvidenceCleaner:
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize newlines
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        # Remove redundant spaces
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        # Remove page break markers
        cleaned = re.sub(r"--- Page \d+ ---", "", cleaned)
        return cleaned.strip()
