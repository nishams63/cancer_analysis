"""
Prompt management and schema parsing module for Stage 5 SLM Engineering.
Enforces the standardized clinical decision-support contract:
Risk: <Low | Moderate | High>
Key Finding: <summary of genomic and therapeutic tolerance>
Action: <clinically aligned therapeutic action>
"""

import re
from typing import Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizerFast

DEFAULT_SYSTEM_PROMPT = (
    "You are an oncology clinical decision-support assistant. "
    "Summarize the provided clinical progress note into structured clinical risk, "
    "key finding, and recommended action. "
    "Preserve all explicit negations. Do not fabricate unprescribed antineoplastics or toxicities. "
    "Use only facts grounded in the note."
)


def build_clinical_prompt(
    tokenizer: PreTrainedTokenizerFast,
    clinical_note: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    target_response: Optional[str] = None
) -> str:
    """Builds chat-formatted prompt using official Qwen tokenizer template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Clinical Note:\n{clinical_note.strip()}\n\nProvide:\nRisk:\nKey Finding:\nAction:"}
    ]
    if target_response:
        messages.append({"role": "assistant", "content": target_response.strip()})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_clinical_output(raw_text: str) -> Dict[str, Any]:
    """
    Parses the generated assistant completion into structured clinical fields.
    Does NOT use regex rules to override the model's prediction.
    If fields are missing or invalid, records parsing errors honestly.
    """
    # Isolate assistant response if full prompt is included
    if "<|im_start|>assistant" in raw_text:
        assistant_part = raw_text.split("<|im_start|>assistant")[-1]
    else:
        assistant_part = raw_text

    # Remove end-of-turn tokens
    clean_text = assistant_part.replace("<|im_end|>", "").strip()

    risk_match = re.search(r"Risk:\s*([^\n\r]+)", clean_text, re.IGNORECASE)
    key_finding_match = re.search(r"Key Finding:\s*([^\n\r]+(?:\n(?!(?:Action:|$))[^\n\r]+)*)", clean_text, re.IGNORECASE)
    action_match = re.search(r"Action:\s*([^\n\r]+(?:\n(?!(?:Risk:|$))[^\n\r]+)*)", clean_text, re.IGNORECASE)

    parsed = {
        "raw_text": clean_text,
        "risk": risk_match.group(1).strip() if risk_match else None,
        "key_finding": key_finding_match.group(1).strip() if key_finding_match else None,
        "action": action_match.group(1).strip() if action_match else None,
        "is_valid_format": False,
        "missing_fields": []
    }

    for field in ["risk", "key_finding", "action"]:
        if parsed[field] is None:
            parsed["missing_fields"].append(field)

    parsed["is_valid_format"] = (len(parsed["missing_fields"]) == 0)

    # Validate risk category
    if parsed["risk"] is not None:
        clean_risk = parsed["risk"].strip().capitalize()
        if clean_risk in {"Low", "Moderate", "High"}:
            parsed["normalized_risk"] = clean_risk
        else:
            parsed["normalized_risk"] = "Invalid"
            parsed["is_valid_format"] = False
    else:
        parsed["normalized_risk"] = "Missing"

    return parsed
