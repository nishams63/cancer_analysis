"""Pipeline to build evidence store and chunk documents."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evidence.evidence_loader import EvidenceLoader
from src.utils.logging import get_logger

logger = get_logger("build_evidence_store")

def run_build_evidence_store() -> dict:
    logger.info("Building approved evidence store...")
    loader = EvidenceLoader()
    docs = [
        {
            "file": Path("stage5/data/evidence/documents/oncology_system_architecture_spec.md"),
            "doc_id": "DOC-SPEC-001",
            "title": "Oncology Decision Engine System Architecture Specification",
            "source": "ONCOLOGY_SPEC_001",
            "source_type": "clinical_guideline",
            "version": "2026.v2",
            "category": "clinical_terminology",
            "section": "System Architecture & 6-Stage Roadmap"
        },
        {
            "file": Path("stage5/data/evidence/documents/nccn_nsclc_targeted_therapy_guideline.md"),
            "doc_id": "DOC-NCCN-002",
            "title": "NCCN NSCLC Biomarker Concordance & Targeted Therapy",
            "source": "NCCN_NSCLC_REF_002",
            "source_type": "external_reference",
            "version": "2026.v1",
            "category": "biomarker",
            "section": "Molecular Testing & Resistance Pathways"
        },
        {
            "file": Path("stage5/data/evidence/documents/fda_oncology_drug_safety_bulletins.md"),
            "doc_id": "DOC-FDA-003",
            "title": "FDA Oncology Drug Labeling & Dosage Specifications",
            "source": "FDA_DRUG_LABEL_003",
            "source_type": "external_reference",
            "version": "2026.v1",
            "category": "treatment",
            "section": "Approved Dosages & Safety Bulletins"
        },
        {
            "file": Path("stage5/data/evidence/documents/ctcae_v5_organ_toxicity_grading.md"),
            "doc_id": "DOC-CTCAE-004",
            "title": "CTCAE v5.0 Toxicity & Organ Grading Definitions",
            "source": "CTCAE_TOXICITY_004",
            "source_type": "external_reference",
            "version": "v5.0",
            "category": "toxicity",
            "section": "Organ Toxicity Grading Criteria"
        }
    ]
    results = {}
    for d in docs:
        if d["file"].exists():
            res = loader.process_document(
                filepath=d["file"],
                document_id=d["doc_id"],
                title=d["title"],
                source=d["source"],
                source_type=d["source_type"],
                version=d["version"],
                evidence_category=d["category"],
                section=d["section"]
            )
            results[d["doc_id"]] = res
    logger.info(f"Evidence store processed {len(results)} documents.")
    return results

if __name__ == "__main__":
    run_build_evidence_store()
