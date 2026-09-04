"""
Stage 2 Deep Learning (DL) - Longitudinal BiLSTM Training Script
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from . import config, utils, datasets, temporal_model
except (ImportError, ValueError):
    import config, utils, datasets, temporal_model

def collate_temporal_batch(batch):
    features = torch.stack([item['features'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    target_ctdna = torch.stack([item['target_ctdna'] for item in batch])
    target_prog = torch.stack([item['target_progression'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    return features, lengths, target_ctdna, target_prog, patient_ids

def train_one_epoch(model, loader, reg_criterion, cls_criterion, optimizer, device, lambda_cls=1.0):
    model.train()
    total_loss, total_reg_loss, total_cls_loss = 0.0, 0.0, 0.0
    all_y_ctdna, all_p_ctdna = [], []
    all_y_prog, all_p_prog = [], []

    for features, lengths, y_ctdna, y_prog, _ in loader:
        features = features.to(device)
        lengths = lengths.to(device)
        y_ctdna = y_ctdna.to(device)
        y_prog = y_prog.to(device)

        optimizer.zero_grad()
        pred_ctdna, pred_prog_logits = model(features, lengths)

        l_reg = reg_criterion(pred_ctdna, y_ctdna)
        l_cls = cls_criterion(pred_prog_logits, y_prog)
        loss = l_reg + lambda_cls * l_cls

        loss.backward()
        optimizer.step()

        batch_sz = features.size(0)
        total_loss += loss.item() * batch_sz
        total_reg_loss += l_reg.item() * batch_sz
        total_cls_loss += l_cls.item() * batch_sz

        all_y_ctdna.extend(y_ctdna.cpu().numpy())
        all_p_ctdna.extend(pred_ctdna.detach().cpu().numpy())
        all_y_prog.extend(y_prog.cpu().numpy())
        all_p_prog.extend(torch.sigmoid(pred_prog_logits).detach().cpu().numpy())

    n = len(loader.dataset)
    reg_metrics = utils.compute_regression_metrics(np.array(all_y_ctdna), np.array(all_p_ctdna))
    cls_metrics = utils.compute_binary_metrics(np.array(all_y_prog), np.array(all_p_prog))

    return total_loss / n, total_reg_loss / n, total_cls_loss / n, reg_metrics, cls_metrics

def evaluate(model, loader, reg_criterion, cls_criterion, device, lambda_cls=1.0):
    model.eval()
    total_loss, total_reg_loss, total_cls_loss = 0.0, 0.0, 0.0
    all_y_ctdna, all_p_ctdna = [], []
    all_y_prog, all_p_prog = [], []

    with torch.no_grad():
        for features, lengths, y_ctdna, y_prog, _ in loader:
            features = features.to(device)
            lengths = lengths.to(device)
            y_ctdna = y_ctdna.to(device)
            y_prog = y_prog.to(device)

            pred_ctdna, pred_prog_logits = model(features, lengths)

            l_reg = reg_criterion(pred_ctdna, y_ctdna)
            l_cls = cls_criterion(pred_prog_logits, y_prog)
            loss = l_reg + lambda_cls * l_cls

            batch_sz = features.size(0)
            total_loss += loss.item() * batch_sz
            total_reg_loss += l_reg.item() * batch_sz
            total_cls_loss += l_cls.item() * batch_sz

            all_y_ctdna.extend(y_ctdna.cpu().numpy())
            all_p_ctdna.extend(pred_ctdna.cpu().numpy())
            all_y_prog.extend(y_prog.cpu().numpy())
            all_p_prog.extend(torch.sigmoid(pred_prog_logits).cpu().numpy())

    n = len(loader.dataset)
    reg_metrics = utils.compute_regression_metrics(np.array(all_y_ctdna), np.array(all_p_ctdna))
    cls_metrics = utils.compute_binary_metrics(np.array(all_y_prog), np.array(all_p_prog))

    return total_loss / n, total_reg_loss / n, total_cls_loss / n, reg_metrics, cls_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Longitudinal BiLSTM Forecaster")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimensions")
    parser.add_argument("--arch", type=str, default="lstm", choices=["lstm", "transformer"], help="Architecture")
    parser.add_argument("--lambda-cls", type=float, default=1.0, help="Weight for classification loss")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 epochs on small subset for verification")
    args = parser.parse_args()

    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cpu")
    print(f"Executing Temporal Training on: {device} | Architecture: {args.arch} | Epochs: {args.epochs}")

    # 1. Datasets & Loaders
    train_dataset = datasets.TemporalSequenceDataset(split='train')
    # Use training normalization parameters for validation split (Anti-Leakage)
    val_dataset = datasets.TemporalSequenceDataset(split='validation', norm_params=train_dataset.norm_params)

    if args.smoke_test:
        print(">>> SMOKE TEST MODE: Sampling 32 train and 16 validation patients.")
        train_dataset = Subset(train_dataset, range(min(32, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(16, len(val_dataset))))
        args.epochs = 2

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_temporal_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_temporal_batch)

    # 2. Model & Optimization
    model = temporal_model.build_temporal_model(
        arch=args.arch,
        input_dim=config.NUM_TEMPORAL_FEATURES,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.2
    ).to(device)

    reg_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_f1': []}
    best_val_score = -999.0
    best_checkpoint = None

    print("\nStarting Longitudinal BiLSTM Multi-Task Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_reg, tr_cls, tr_reg_m, tr_cls_m = train_one_epoch(
            model, train_loader, reg_criterion, cls_criterion, optimizer, device, lambda_cls=args.lambda_cls
        )
        val_loss, val_reg, val_cls, val_reg_m, val_cls_m = evaluate(
            model, val_loader, reg_criterion, cls_criterion, device, lambda_cls=args.lambda_cls
        )
        scheduler.step(val_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_reg_m['mae'])
        history['val_f1'].append(val_cls_m['f1'])

        # Composite score: higher classification F1 and lower regression MAE
        composite_score = val_cls_m['f1'] - (val_reg_m['mae'] * 0.1)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {tr_loss:.4f} (Reg: {tr_reg:.3f}, Cls: {tr_cls:.3f}) | "
              f"Val Loss: {val_loss:.4f} (MAE: {val_reg_m['mae']:.3f}, R2: {val_reg_m['r2']:.3f} | F1: {val_cls_m['f1']:.3f}, AUC: {val_cls_m['roc_auc']:.3f})")

        if composite_score > best_val_score:
            best_val_score = composite_score
            norm_p_serializable = {k: [float(v[0]), float(v[1])] for k, v in (train_dataset.norm_params if hasattr(train_dataset, 'norm_params') else train_dataset.dataset.norm_params).items()}
            best_checkpoint = {
                'model_architecture': args.arch,
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'hidden_dim': args.hidden_dim,
                'input_dim': config.NUM_TEMPORAL_FEATURES,
                'best_val_reg_metrics': val_reg_m,
                'best_val_cls_metrics': val_cls_m,
                'composite_score': float(composite_score),
                'norm_params': norm_p_serializable,
                'temporal_features': config.ALL_TEMPORAL_FEATURES,
                'random_seed': config.RANDOM_SEED,
                'disclaimer': config.MANDATORY_DISCLAIMER
            }
            utils.save_checkpoint(best_checkpoint, str(config.BEST_TEMPORAL_CHECKPOINT))

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.1f}s. Best Validation Composite Score: {best_val_score:.4f}")

    if not args.smoke_test and best_checkpoint is not None:
        utils.plot_temporal_training_curves(history, str(config.FIGURES_DIR / 'temporal_training_curves.png'))
        print(f"Saved best checkpoint to: {config.BEST_TEMPORAL_CHECKPOINT}")
        print(f"Saved curves to: {config.FIGURES_DIR}")

if __name__ == '__main__':
    main()
