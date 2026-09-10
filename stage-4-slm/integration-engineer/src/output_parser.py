"""
Output Parsing and Structured Response Builder Module.
Extracts Risk, Key Finding, Action and formats concise spoken summary.
"""

import re
from typing import Dict, Any, Tuple


class OutputParser:
    """Parses raw text generation into structured clinical fields."""

    @staticmethod
    def parse_generation(raw_text: str) -> Tuple[Dict[str, str], str]:
        """
        Parses raw model output into fields and generates the 3-line spoken summary.
        """
        fields = {"Risk": "", "Key Finding": "", "Action": ""}
        text = str(raw_text).strip()

        risk_m = re.search(r"Risk:\s*([^\n\r]+)", text, re.IGNORECASE)
        kf_m = re.search(r"Key Finding:\s*([^\n\r]+)", text, re.IGNORECASE)
        act_m = re.search(r"Action:\s*([^\n\r]+)", text, re.IGNORECASE)

        if risk_m:
            fields["Risk"] = risk_m.group(1).strip()
        if kf_m:
            fields["Key Finding"] = kf_m.group(1).strip()
        if act_m:
            fields["Action"] = act_m.group(1).strip()

        # Build concise 3-line spoken summary (voice-ready)
        r_line = f"Risk: {fields['Risk']}." if fields["Risk"] else "Risk: Unspecified."
        kf_line = f"Key finding: {fields['Key Finding']}." if fields["Key Finding"] else "Key finding: Review clinical documentation."
        a_line = f"Action: {fields['Action']}." if fields["Action"] else "Action: Clinical evaluation recommended."

        spoken_summary = f"{r_line}\n{kf_line}\n{a_line}"

        return fields, spoken_summary
