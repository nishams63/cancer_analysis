"""Treatment and oncology drug indication validator."""
from typing import Dict, Any, List, Tuple


class TreatmentValidator:
    # Allowed drug indications
    APPROVED_REGIMENS = {
        "NSCLC": [
            "Osimertinib", "Cisplatin", "Docetaxel", "Savolitinib", "Tepotinib",
            "Carboplatin", "Standard Cytotoxic Chemotherapy", "Third-generation EGFR TKI",
            "EGFR TKI", "TKI", "Carboplatin AUC", "Pemetrexed"
        ],
        "Colorectal": ["FOLFOX", "FOLFIRI", "Capecitabine", "Cetuximab", "Bevacizumab", "Standard Cytotoxic Chemotherapy"],
        "Breast": ["Docetaxel", "Doxorubicin", "Cyclophosphamide", "Trastuzumab", "Pertuzumab", "Standard Cytotoxic Chemotherapy"],
        "Melanoma": [
            "Pembrolizumab", "Nivolumab", "Ipilimumab", "Dabrafenib", "Trametinib",
            "Standard Cytotoxic Chemotherapy", "anti-PD-1", "anti-PD-1 re-challenge", "PD-1 inhibitor"
        ]
    }

    def validate(self, patient_dict: Dict[str, Any], scenario: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        violations = []
        cancer_type = patient_dict.get("demographics", {}).get("cancer_type", "")
        treatments = patient_dict.get("treatments", [])

        approved_list = list(self.APPROVED_REGIMENS.get(cancer_type, []))
        if scenario:
            for req in scenario.get("required_entities", []):
                if req.get("entity_type") == "treatment":
                    allowed_vals = req.get("allowed_values") or []
                    for av in allowed_vals:
                        if av and "toxicity" not in av.lower() and "myocarditis" not in av.lower():
                            approved_list.append(av)

        for t in treatments:
            tname = t.get("treatment_name", "")
            if approved_list and not any(app.lower() in tname.lower() or tname.lower() in app.lower() for app in approved_list):
                violations.append(f"Unapproved or unrecognized drug '{tname}' for {cancer_type}")

        return (len(violations) == 0, violations)