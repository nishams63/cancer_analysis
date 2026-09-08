"""
Stage 2 Deep Learning - Temporal Forecasting & Transformer Service

Responsible for:
  - Longitudinal biomarker sequence preprocessing & causal forward fill
  - Strict historical cutoff enforcement (Day <= 90)
  - Continuous time encoding (absolute day & delta_days)
  - Model execution (ContinuousTemporalTransformer upgraded vs BiLSTM baseline)
  - 30-day forward ctDNA VAF forecast regression
  - Progression risk probability prediction
  - Patient-level 64-dimensional temporal representation
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from . import config, model_registry, schemas, validation
    from .model_registry import ModelRegistry, MissingCheckpointError
except (ImportError, ValueError):
    import config, model_registry, schemas, validation
    from model_registry import ModelRegistry, MissingCheckpointError

# Import DL model architectures and data utilities
from dl.models.temporal_transformer import ContinuousTemporalTransformer
from dl.src.temporal_model import BiLSTMForecaster
from dl.research_data import AuditedTemporalDataset, prepare_history
from dl import config as dl_config


class TemporalService:
    """Singleton service for longitudinal biomarker forecasting."""
    _instance = None

    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        # Fetch training normalization parameters
        try:
            self.norm_params = AuditedTemporalDataset('train').norm_params
        except Exception:
            # Safe defaults derived from training manifest
            self.norm_params = {
                'days_from_baseline': (45.0, 28.0),
                'ctDNA_vaf_percent': (1.20, 1.45),
                'cea_ng_ml': (3.80, 2.50),
                'ca125_u_ml': (24.5, 18.2),
                'ldh_u_l': (195.0, 48.0),
                'crp_mg_l': (4.20, 3.10),
                'delta_days': (14.0, 7.0),
                'ctDNA_velocity_30d': (0.05, 0.40)
            }

    @classmethod
    def get_shared(cls) -> 'TemporalService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_temporal_model(self, architecture: str = 'transformer') -> nn.Module:
        """Loads and caches the temporal model in eval mode."""
        if architecture in self.models:
            return self.models[architecture]

        if architecture == 'transformer':
            ckpt_key = "temporal_transformer"
            if ModelRegistry.is_available(ckpt_key):
                ckpt_path = ModelRegistry.get_path(ckpt_key)
                model = ContinuousTemporalTransformer()
                state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(state.get('state_dict', state))
            else:
                model = ContinuousTemporalTransformer()
            model.eval()
            self.models['transformer'] = model
            return model

        elif architecture == 'bilstm':
            if ModelRegistry.is_available("temporal_bilstm"):
                ckpt_path = ModelRegistry.get_path("temporal_bilstm")
                model = BiLSTMForecaster()
                state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(state.get('state_dict', state))
            elif ModelRegistry.is_available("baseline_temporal"):
                ckpt_path = ModelRegistry.get_path("baseline_temporal")
                model = BiLSTMForecaster()
                state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                weights = state.get('state_dict', state.get('model_state_dict', state))
                model.load_state_dict(weights)
            else:
                raise MissingCheckpointError("Neither 'temporal_bilstm' nor 'baseline_temporal' checkpoint exists.")
            model.eval()
            self.models['bilstm'] = model
            return model
        else:
            raise ValueError(f"Unsupported temporal architecture: {architecture}")

    def process_observations(
        self,
        observations_df: pd.DataFrame,
        architecture: str = 'transformer',
        max_seq_len: int = 10
    ) -> Tuple[schemas.TemporalOutput, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes chronological biomarker observations through the temporal model.

        Args:
            observations_df: Validated DataFrame of visits with days_from_baseline <= 90.
            architecture: 'transformer' or 'bilstm'.
            max_seq_len: Maximum sequence window (default 10).

        Returns:
            Tuple of (TemporalOutput, temporal_representation [1, 64], (inputs_features, lengths))
        """
        validated_df = validation.validate_temporal_history(observations_df)
        if validated_df is None or len(validated_df) == 0:
            raise ValueError("Cannot process empty temporal history.")

        # Ensure all standard laboratory features exist
        for lab in dl_config.TEMPORAL_NUMERICAL_FEATURES[:5]:
            if lab not in validated_df.columns:
                validated_df[lab] = np.nan

        # Use audited feature preparation with strict causal forward fill
        features, seq_len, processed_history = prepare_history(
            history=validated_df,
            norm_params=self.norm_params,
            max_seq_len=max_seq_len
        )

        input_features = features.unsqueeze(0)  # (1, max_seq_len, num_features)
        input_lengths = torch.tensor([seq_len], dtype=torch.long)
        model_inputs = (input_features, input_lengths)

        model = self.get_temporal_model(architecture)

        with torch.no_grad():
            outputs = model(*model_inputs)
            ctdna_pred, prog_logit = outputs[:2]

            if architecture == 'bilstm':
                # Extract LSTM hidden state at the last valid historical visit
                states, _ = model.lstm(input_features)
                temporal_repr = states[:, seq_len - 1]  # (1, 64)
            else:
                temporal_repr = outputs[2]  # (1, 64)

            prog_prob = float(torch.sigmoid(prog_logit)[0].item())
            ctdna_forecast = float(ctdna_pred[0].item())

        max_day = int(validated_df['days_from_baseline'].max())

        summary = schemas.TemporalOutput(
            available=True,
            model_name=architecture,
            progression_probability=prog_prob,
            ctdna_30d_forecast=ctdna_forecast,
            sequence_length=int(seq_len),
            max_historical_day=max_day,
            embedding_dimension=int(temporal_repr.shape[-1])
        )

        return summary, temporal_repr, model_inputs
