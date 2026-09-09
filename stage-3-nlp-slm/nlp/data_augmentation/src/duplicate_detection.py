"""
Duplicate and Near-Duplicate Detection Module.
Calculates exact cryptographic hashes, canonical whitespace hashes,
and n-gram Jaccard similarities to prevent trivial duplicates or near-verbatim clones.
"""

from typing import Set, Tuple, List
import hashlib
import re


def compute_sha256(text: str) -> str:
    """Compute standard SHA256 hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_canonical_hash(text: str) -> str:
    """Compute hash after whitespace canonicalization and lowercasing."""
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def compute_token_set(text: str, n: int = 2) -> Set[str]:
    """Extract word n-grams for Jaccard similarity estimation."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < n:
        return set(tokens)
    return set(" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_jaccard_similarity(text1: str, text2: str, n: int = 2) -> float:
    """Calculate word n-gram Jaccard similarity between two texts."""
    set1 = compute_token_set(text1, n=n)
    set2 = compute_token_set(text2, n=n)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union) if union > 0 else 0.0


class DuplicateRegistry:
    """In-memory registry to detect duplicates across original and augmented pools."""

    def __init__(self):
        self.exact_hashes: Set[str] = set()
        self.canonical_hashes: Set[str] = set()

    def register(self, text: str) -> None:
        """Add text to registry."""
        self.exact_hashes.add(compute_sha256(text))
        self.canonical_hashes.add(compute_canonical_hash(text))

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        """Check if text is an exact or canonical duplicate."""
        exact_h = compute_sha256(text)
        if exact_h in self.exact_hashes:
            return True, "exact_duplicate"
        canonical_h = compute_canonical_hash(text)
        if canonical_h in self.canonical_hashes:
            return True, "canonical_duplicate"
        return False, "unique"
