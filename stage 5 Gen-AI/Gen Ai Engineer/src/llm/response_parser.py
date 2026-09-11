"""Response parser extracting clean narrative sections from raw LLM output."""
import re
from typing import Dict, Any


class ResponseParser:
    def parse_narrative(self, raw_text: str) -> Dict[str, str]:
        # Strip potential markdown code fences
        cleaned = re.sub(r"^```(?:markdown|text)?", "", raw_text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

        sections = {"full_narrative": cleaned}

        # Check for standard SOAP headings
        for sec in ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]:
            m = re.search(rf"(?:###|##|\*\*|[0-9]\.)?\s*{sec}[:\s](.+?)(?=(?:###|##|\*\*|[0-9]\.)?\s*(?:SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN)|$)", cleaned, re.IGNORECASE | re.DOTALL)
            if m:
                sections[sec.lower()] = m.group(1).strip()

        return sections