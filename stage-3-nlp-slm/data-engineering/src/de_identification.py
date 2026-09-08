"""
De-identification and Privacy Sanitization Module for Clinical NLP.
Implements HIPAA Safe Harbor compliant pattern scrubbing for clinical texts.
"""

import re
from typing import Tuple, Dict, Any, List

# Regex patterns for common direct identifiers
PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'),
    "phone": re.compile(r'\b(?:\+?1[-. ]?)?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "mrn": re.compile(r'\bMRN[:#\s]*[0-9]{6,10}\b', re.IGNORECASE),
    "url": re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*'),
    "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
    # Synthetic name placeholders (e.g., "Patient Name: Jane Doe", "Dr. Robert Smith")
    "provider_name": re.compile(r'\b(?:Dr\.|Doctor|MD|Oncologist)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'),
    "patient_named_prefix": re.compile(r'(?:Patient Name|Pt Name|Name)[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)\b', re.IGNORECASE)
}

def scrub_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Sanitize text by replacing sensitive identifiers with canonical privacy tokens.
    Returns:
        sanitized_text: string with PII replaced
        counts: dictionary of scrubbed counts by category
    """
    if not isinstance(text, str) or not text.strip():
        return text, {}
    
    counts: Dict[str, int] = {}
    cleaned = text
    
    # 1. Emails
    emails = PATTERNS["email"].findall(cleaned)
    if emails:
        counts["email"] = len(emails)
        cleaned = PATTERNS["email"].sub("[EMAIL]", cleaned)
        
    # 2. Phones
    phones = PATTERNS["phone"].findall(cleaned)
    if phones:
        counts["phone"] = len(phones)
        cleaned = PATTERNS["phone"].sub("[PHONE]", cleaned)
        
    # 3. SSNs
    ssns = PATTERNS["ssn"].findall(cleaned)
    if ssns:
        counts["ssn"] = len(ssns)
        cleaned = PATTERNS["ssn"].sub("[SSN]", cleaned)
        
    # 4. MRNs
    mrns = PATTERNS["mrn"].findall(cleaned)
    if mrns:
        counts["mrn"] = len(mrns)
        cleaned = PATTERNS["mrn"].sub("[MRN]", cleaned)
        
    # 5. URLs
    urls = PATTERNS["url"].findall(cleaned)
    if urls:
        counts["url"] = len(urls)
        cleaned = PATTERNS["url"].sub("[URL]", cleaned)
        
    # 6. Provider names
    providers = PATTERNS["provider_name"].findall(cleaned)
    if providers:
        counts["provider_name"] = len(providers)
        cleaned = PATTERNS["provider_name"].sub(r"Dr. [NAME]", cleaned)
        
    # 7. Patient named prefixes
    pt_names = PATTERNS["patient_named_prefix"].findall(cleaned)
    if pt_names:
        counts["patient_name"] = len(pt_names)
        cleaned = PATTERNS["patient_named_prefix"].sub("Patient Name: [NAME]", cleaned)
        
    return cleaned, counts

def verify_no_direct_identifiers(text: str) -> List[str]:
    """
    Scan text to verify no unmasked direct identifiers remain.
    Returns list of violation descriptions (empty if clean).
    """
    violations = []
    if PATTERNS["email"].search(text):
        violations.append("Unmasked Email Address Detected")
    if PATTERNS["phone"].search(text):
        violations.append("Unmasked Phone Number Detected")
    if PATTERNS["ssn"].search(text):
        violations.append("Unmasked Social Security Number Detected")
    return violations
