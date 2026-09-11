"""Sampler for longitudinal oncology timelines and acute decompensation."""
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
from .patient_builder import TimelineEvent


class TemporalSampler:
    def __init__(self):
        pass

    def build_timeline_for_scenario(self, scenario: Dict[str, Any], rng: random.Random) -> List[TimelineEvent]:
        events = []
        base_date = datetime(2026, 1, 15)
        sid = scenario.get("scenario_id", "")

        # 1. Baseline Diagnosis
        events.append(TimelineEvent(
            event_id="EVT-01",
            event_type="diagnosis",
            timestamp=base_date.strftime("%Y-%m-%d"),
            delta_days=0,
            description="Initial clinical staging and histological confirmation",
            details={"stage": "Stage IV", "ecog": 1}
        ))

        # 2. Line 1 Treatment Start
        line1_date = base_date + timedelta(days=rng.randint(7, 14))
        events.append(TimelineEvent(
            event_id="EVT-02",
            event_type="treatment_initiation",
            timestamp=line1_date.strftime("%Y-%m-%d"),
            delta_days=(line1_date - base_date).days,
            description="Initiation of systemic antineoplastic therapy",
            details={"regimen": "Standard First-Line Protocol"}
        ))

        # 3. Progression or Toxicity Event
        if "RAPID_SEPSIS" in scenario.get("category", ""):
            # Hyper-acute shock scenario
            chemo_date = line1_date + timedelta(days=21)
            shock_date = chemo_date + timedelta(days=8)
            events.append(TimelineEvent(
                event_id="EVT-03",
                event_type="chemo_administration",
                timestamp=chemo_date.strftime("%Y-%m-%d"),
                delta_days=(chemo_date - base_date).days,
                description="Chemotherapy cycle 2 administration",
                details={"regimen": "Docetaxel 75 mg/m2"}
            ))
            events.append(TimelineEvent(
                event_id="EVT-04",
                event_type="acute_toxicity",
                timestamp=shock_date.strftime("%Y-%m-%d") + "T14:00:00",
                delta_days=(shock_date - base_date).days,
                description="Hyper-acute decompensation: febrile neutropenia progressing to septic shock within 6 hours",
                details={"temp_c": 39.4, "map_mmhg": 59, "lactate": 4.2}
            ))
        elif "TIMELINE_DISCORDANCE" in scenario.get("category", ""):
            # RECIST progression on 10/14, stale clinic note on 10/16
            ct_date = base_date + timedelta(days=90)
            note_date = ct_date + timedelta(days=2)
            events.append(TimelineEvent(
                event_id="EVT-03",
                event_type="radiology_scan",
                timestamp=ct_date.strftime("%Y-%m-%d"),
                delta_days=(ct_date - base_date).days,
                description="CT Chest/Abdomen: 52% increase in target hepatic metastases (RECIST Progressive Disease)",
                details={"recist_status": "Progressive Disease", "size_increase_pct": 52}
            ))
            events.append(TimelineEvent(
                event_id="EVT-04",
                event_type="outpatient_clinic_note",
                timestamp=note_date.strftime("%Y-%m-%d"),
                delta_days=(note_date - base_date).days,
                description="Enc note: Stale copy-forward documentation erroneously recording 'Stable Disease'",
                details={"text_claim": "Stable Disease", "contradiction": True}
            ))
        else:
            prog_date = line1_date + timedelta(days=rng.randint(120, 240))
            events.append(TimelineEvent(
                event_id="EVT-03",
                event_type="progression",
                timestamp=prog_date.strftime("%Y-%m-%d"),
                delta_days=(prog_date - base_date).days,
                description="Confirmed radiological and clinical progression on therapy",
                details={"resistance_detected": True}
            ))

        return events