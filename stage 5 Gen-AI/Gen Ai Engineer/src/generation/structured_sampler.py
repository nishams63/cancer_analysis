"""Master Structured Synthetic Patient Sampler.

Enforces:
1. Medical facts originate HERE, not from the LLM.
2. Controlled deterministic generation seeded for reproducibility.
3. Incorporates reference distributions and scenario specifications.
"""
import random
from typing import Dict, Any, Optional
from datetime import datetime
from src.utils.seeds import SeedContext
from .patient_builder import StructuredSyntheticPatient, Demographics, Resistance
from .rarity_sampler import RaritySampler
from .temporal_sampler import TemporalSampler
from .missingness_sampler import MissingnessSampler


class StructuredSampler:
    def __init__(self, rare_space_config: Dict[str, Any] = None):
        self.rarity_sampler = RaritySampler(rare_space_config)
        self.temporal_sampler = TemporalSampler()
        self.missingness_sampler = MissingnessSampler()
        self._patient_counter = 0

    def sample_patient(self, scenario: Dict[str, Any], seed: Optional[int] = None) -> StructuredSyntheticPatient:
        """Sample a complete, factual synthetic oncology patient scenario."""
        seed = seed if seed is not None else random.randint(1000, 999999)
        rng = random.Random(seed)

        self._patient_counter += 1
        patient_id = f"SYN-P{self._patient_counter:06d}"
        scenario_id = scenario.get("scenario_id", "SYN-S001")

        # 1. Demographics
        sk = scenario.get("patient_skeleton", {})
        age = sk.get("age", rng.randint(45, 75))
        sex = sk.get("sex", rng.choice(["Male", "Female"]))
        cancer_type = sk.get("cancer_type", "NSCLC")
        cancer_stage = "Stage IV"

        # Check required entities for demographic overrides
        for req in scenario.get("required_entities", []):
            if req.get("name") == "age":
                min_v = int(req.get("min_value", 80))
                max_v = int(req.get("max_value", 90))
                age = rng.randint(min_v, max_v)

        demographics = Demographics(
            age=age,
            sex=sex,
            cancer_type=cancer_type,
            cancer_stage=cancer_stage
        )

        # 2. Mutations
        mutations = self.rarity_sampler.sample_mutations_for_scenario(scenario, rng)

        # 3. Biomarkers & Labs
        biomarkers = {}
        for req in scenario.get("required_entities", []):
            etype = req.get("entity_type")
            if etype in {"biomarker", "lab", "vital"}:
                name = req.get("name")
                min_v = req.get("min_value")
                max_v = req.get("max_value")
                if min_v is not None and max_v is not None:
                    val = round(rng.uniform(min_v, max_v), 2)
                    biomarkers[name] = val
                elif min_v is not None:
                    biomarkers[name] = round(min_v + rng.uniform(0.1, 1.0), 2)
                elif req.get("allowed_values"):
                    # Categorical biomarker or lab
                    biomarkers[name] = rng.choice(req["allowed_values"])

        # Default biomarkers if not present
        if "tumor_size_cm" not in biomarkers:
            biomarkers["tumor_size_cm"] = round(rng.uniform(1.2, 4.5), 1)
        if "creatinine_level" not in biomarkers and "serum_creatinine" not in biomarkers:
            biomarkers["creatinine_level"] = round(rng.uniform(0.7, 1.1), 2)

        # 4. Treatments & Dosages
        treatments = []
        dosages = {}
        adverse_events = []

        for req in scenario.get("required_entities", []):
            if req.get("entity_type") == "treatment":
                tname = req.get("name", "")
                allowed = req.get("allowed_values", [])
                tval = rng.choice(allowed) if allowed else tname

                # Check if this treatment entity is actually a prior toxicity/adverse event
                if "toxicity" in tname.lower() or "myocarditis" in str(allowed).lower() or "adverse" in tname.lower():
                    adverse_events.append({
                        "ae_name": tval,
                        "grade": 3,
                        "ctcae_term": "Immune-mediated myocarditis",
                        "attribution": "Prior immune checkpoint inhibitor"
                    })
                else:
                    treatments.append({"treatment_name": tval, "line": sk.get("prior_lines_therapy", 1)})
                    if "Osimertinib" in tval or "EGFR TKI" in tval:
                        dosages["Osimertinib"] = 80.0
                    elif "Cisplatin" in tval:
                        dosages["Cisplatin"] = 75.0
                    elif "Capecitabine" in tval:
                        dosages["Capecitabine"] = 1250.0

        if not treatments:
            treatments.append({"treatment_name": "Standard Cytotoxic Chemotherapy", "line": 1})

        # 5. Adverse Events
        if "serum_creatinine" in biomarkers and biomarkers["serum_creatinine"] >= 3.0:
            adverse_events.append({
                "ae_name": "Acute Kidney Injury",
                "grade": 3,
                "ctcae_term": "Acute kidney injury",
                "attribution": "Probable Cisplatin-induced nephrotoxicity"
            })
        if "INVERSE_WEIGHT_HAZARD" in scenario.get("category", ""):
            adverse_events.append({
                "ae_name": "Mild Xerosis",
                "grade": 1,
                "ctcae_term": "Dry skin",
                "attribution": "EGFR-TKI cutaneous effect"
            })

        # 6. Resistance
        resistance_status = True if (len(mutations) >= 2 or "resistance" in scenario.get("title", "").lower()) else False
        resistance = Resistance(
            status=resistance_status,
            mechanism="Secondary bypass or acquired gatekeeper mutation" if resistance_status else None
        )

        # 7. Timeline
        timeline = self.temporal_sampler.build_timeline_for_scenario(scenario, rng)

        # 8. Missing Fields
        missing_fields = self.missingness_sampler.sample_missing_fields(scenario, rng)
        for mf in missing_fields:
            if mf in biomarkers:
                del biomarkers[mf]

        # 9. Provenance Metadata
        provenance = {
            "generator": "StructuredSampler_v1.0",
            "seed": seed,
            "scenario_version": scenario.get("version", "1.0.0"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "distribution_version": "1.0.0",
            "constraint_version": "1.0.0"
        }

        return StructuredSyntheticPatient(
            scenario_id=scenario_id,
            patient_id=patient_id,
            synthetic=True,
            demographics=demographics,
            mutations=mutations,
            biomarkers=biomarkers,
            treatments=treatments,
            dosages=dosages,
            adverse_events=adverse_events,
            resistance=resistance,
            timeline=timeline,
            missing_fields=missing_fields,
            scenario_requirements={
                "prompt_id": scenario.get("scenario_id"),
                "category": scenario.get("category"),
                "target_stages": [str(s) for s in scenario.get("target_stages", [])]
            },
            provenance=provenance
        )