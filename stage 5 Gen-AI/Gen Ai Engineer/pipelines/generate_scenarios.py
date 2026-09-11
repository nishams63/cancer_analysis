"""Structured scenario batch generator."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
from src.generation import ScenarioLoader, StructuredSampler
from src.validation import ConstraintValidator


def generate_structured_batch(n: int = 15, seed: int = 42, output_path: str = None):
    output_path = output_path or os.path.join("stage5", "genai", "generation", "structured", "synthetic_scenarios.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    loader = ScenarioLoader()
    scenarios = loader.get_all_scenarios()
    sampler = StructuredSampler()
    validator = ConstraintValidator()

    generated = []
    for idx in range(n):
        sc = scenarios[idx % len(scenarios)]
        cur_seed = seed + idx
        pt = sampler.sample_patient(sc, seed=cur_seed)
        val_res = validator.validate_patient(pt.to_dict(), sc)
        if val_res["valid"]:
            generated.append(pt.to_dict())

    with open(output_path, "w", encoding="utf-8") as f:
        for pt_dict in generated:
            f.write(json.dumps(pt_dict) + "\n")

    print(f"Generated {len(generated)} structured scenarios -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_structured_batch()