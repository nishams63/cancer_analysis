"""
Contextual Clinical Concept Model Module for Stage 3 Clinical NLP.
Implements a lightweight Transformer Token Classifier (all-MiniLM-L6-v2) for contextual
extraction of GENE_MUTATION, DRUG_NAME, DOSAGE, and ADVERSE_EVENT mentions,
integrated with clinical negation scoping and contextual embeddings.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

import sys
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from token_alignment import (
    LABEL_LIST,
    ID2LABEL,
    LABEL2ID,
    canonicalize_text,
    align_spans_to_bio_tokens,
    reconstruct_spans_from_bio_predictions
)
from negation_detection import resolve_concept_polarity

MODEL_PRETRAINED_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH = 256


class ClinicalNERDataset(Dataset):
    """PyTorch Dataset for Clinical BIO Token Classification."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for _, row in df.iterrows():
            text = canonicalize_text(row["text"])
            raw_ents = row["ner_entities"]
            entities = json.loads(raw_ents) if isinstance(raw_ents, str) else raw_ents
            aligned = align_spans_to_bio_tokens(text, entities, self.tokenizer, max_length=self.max_length)
            self.samples.append({
                "input_ids": torch.tensor(aligned["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(aligned["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(aligned["labels"], dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ContextualClinicalConceptModel:
    """
    Contextual Clinical Concept Model wrapping Transformer Token Classification
    and contextual representation extraction.
    """

    def __init__(self, model_name_or_path: str = MODEL_PRETRAINED_NAME, device: Optional[str] = None):
        self.model_name = model_name_or_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        self.model.to(self.device)
        self.is_trained = False

    def freeze_lower_layers(self, num_layers_to_freeze: int = 4):
        """Freeze token embeddings and lower encoder layers for parameter-efficient adaptation."""
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.model.bert.encoder.layer):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def fit(
        self,
        df_train: pd.DataFrame,
        epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 4e-5,
        weight_decay: float = 0.01,
        verbose: bool = True
    ) -> "ContextualClinicalConceptModel":
        """Fine-tune token classification head strictly on TRAIN data."""
        if verbose:
            print(f"[ConceptModel] Preparing training dataset ({len(df_train)} records)...")

        dataset = ClinicalNERDataset(df_train, self.tokenizer, max_length=DEFAULT_MAX_LENGTH)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.freeze_lower_layers(num_layers_to_freeze=4)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        self.model.train()

        start_time = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            total_steps = 0

            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

            avg_loss = total_loss / max(1, total_steps)
            if verbose:
                print(f"[ConceptModel] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"[ConceptModel] Training complete in {elapsed:.2f}s ({len(df_train)/elapsed:.1f} docs/sec).")

        self.is_trained = True
        return self

    def predict_entities(self, text: str, assign_polarity: bool = True) -> List[Dict[str, Any]]:
        """
        Extract clinical concepts from narrative text using contextual transformer predictions,
        reconstruct exact character spans, and attribute contextual polarity.
        """
        canon_text = canonicalize_text(text)
        if not canon_text:
            return []

        self.model.eval()
        enc = self.tokenizer(
            canon_text,
            max_length=DEFAULT_MAX_LENGTH,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].cpu().numpy().tolist()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # (seq_len, num_labels)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy().tolist()

        raw_entities = reconstruct_spans_from_bio_predictions(
            canon_text,
            pred_ids,
            token_probs=confidences,
            offsets=offsets
        )

        # Map character spans back if necessary and attach clinical polarity
        attributed_entities = []
        for ent in raw_entities:
            ent_copy = dict(ent)
            if assign_polarity:
                polarity = resolve_concept_polarity(canon_text, ent["start"], ent["end"])
                ent_copy["polarity"] = polarity
            attributed_entities.append(ent_copy)

        return attributed_entities

    def extract_contextual_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Extract 384-dimensional mean-pooled contextual embeddings for downstream classification.
        """
        self.model.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [canonicalize_text(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(
                batch_texts,
                max_length=DEFAULT_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                # Extract hidden states from base bert encoder
                bert_out = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = bert_out.last_hidden_state  # (B, L, 384)

                # Mean pooling with attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled = (sum_embeddings / sum_mask).cpu().numpy()
                all_embeddings.append(pooled)

        return np.vstack(all_embeddings)

    def save(self, save_dir: Path) -> None:
        """Save fine-tuned transformer weights, tokenizer, and label mappings."""
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        with open(save_dir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump({
                "label_list": LABEL_LIST,
                "id2label": ID2LABEL,
                "label2id": LABEL2ID
            }, f, indent=2)

    @classmethod
    def load(cls, load_dir: Path, device: Optional[str] = None) -> "ContextualClinicalConceptModel":
        """Load fine-tuned model and tokenizer from local directory."""
        instance = cls(model_name_or_path=str(load_dir), device=device)
        instance.is_trained = True
        return instance
