"""
Stage 2 Deep Learning (DL) - Pathology Image CNN Training Script
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
    from . import config, utils, datasets, image_model
except (ImportError, ValueError):
    import config, utils, datasets, image_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = utils.compute_classification_metrics(np.array(all_targets), np.array(all_preds))
    return epoch_loss, metrics

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    eval_loss = total_loss / len(loader.dataset)
    metrics = utils.compute_classification_metrics(np.array(all_targets), np.array(all_preds))
    return eval_loss, metrics, np.array(all_probs)

def main():
    parser = argparse.ArgumentParser(description="Train Pathology Tile CNN Classifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "custom_cnn"], help="Architecture")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained backbone")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze backbone for CPU efficiency")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch on small subset for verification")
    parser.add_argument("--use-dataset-norm", action="store_true", help="Use empirical dataset normalization instead of ImageNet")
    args = parser.parse_args()

    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cpu")
    print(f"Executing on device: {device} | Architecture: {args.arch} | Epochs: {args.epochs}")

    # 1. Datasets & Loaders
    train_dataset = datasets.PathologyTileDataset(split='train', use_imagenet_norm=(not args.use_dataset_norm))
    val_dataset = datasets.PathologyTileDataset(split='validation', use_imagenet_norm=(not args.use_dataset_norm))

    if args.smoke_test:
        print(">>> SMOKE TEST MODE: Sampling 128 train and 64 validation tiles.")
        train_dataset = Subset(train_dataset, range(min(128, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(64, len(val_dataset))))
        args.epochs = 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model & Optimization
    model = image_model.build_image_model(
        arch=args.arch,
        pretrained=args.pretrained,
        num_classes=config.NUM_CLASSES,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # Balanced class weights (4000:4000:4000) -> uniform cross-entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = -1.0
    best_checkpoint = None

    print("\nStarting Pathology Tile CNN Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, val_probs = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['macro_f1'])
        history['val_f1'].append(val_metrics['macro_f1'])

        ep_duration = time.time() - ep_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, F1: {train_metrics['macro_f1']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_loss:.4f}, F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Save best model based on validation Macro F1
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_checkpoint = {
                'model_architecture': args.arch,
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'best_val_metrics': val_metrics,
                'class_to_idx': config.CLASS_TO_IDX,
                'idx_to_class': config.IDX_TO_CLASS,
                'normalization': 'dataset' if args.use_dataset_norm else 'imagenet',
                'norm_mean': config.DATASET_MEAN if args.use_dataset_norm else config.IMAGENET_MEAN,
                'norm_std': config.DATASET_STD if args.use_dataset_norm else config.IMAGENET_STD,
                'random_seed': config.RANDOM_SEED,
                'disclaimer': config.MANDATORY_DISCLAIMER
            }
            utils.save_checkpoint(best_checkpoint, str(config.BEST_IMAGE_CHECKPOINT))

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.1f}s. Best Validation Macro F1: {best_val_f1:.4f}")

    if not args.smoke_test and best_checkpoint is not None:
        # Plot and save curves
        utils.plot_image_training_curves(history, str(config.FIGURES_DIR / 'image_training_curves.png'))
        cm = np.array(best_checkpoint['best_val_metrics']['confusion_matrix'])
        utils.plot_confusion_matrix_heatmap(cm, config.CLASSES, str(config.FIGURES_DIR / 'validation_confusion_matrix.png'))
        print(f"Saved best checkpoint to: {config.BEST_IMAGE_CHECKPOINT}")
        print(f"Saved curves and confusion matrix to: {config.FIGURES_DIR}")

if __name__ == '__main__':
    main()
