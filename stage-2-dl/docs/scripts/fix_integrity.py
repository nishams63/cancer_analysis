from pathlib import Path

root = Path(__file__).parent
p = root / 'dl/training/run_benchmarks.py'
s = p.read_text(encoding='utf-8')
start = s.index('    roc_auc, pr_auc = 0.5, 0.5')
end = s.index('\n    return {', start)
s = s[:start] + '''    roc_auc, pr_auc = float('nan'), float('nan')
    if y_prob is not None:
        if not np.isfinite(y_prob).all():
            raise ValueError("Non-finite predictions must not produce benchmark metrics")
        if is_multiclass:
            from sklearn.preprocessing import label_binarize
            labels = np.arange(y_prob.shape[1])
            if len(np.unique(y_true)) == len(labels):
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', labels=labels))
                binary = label_binarize(y_true, classes=labels)
                areas = []
                for k in labels:
                    precision, recall, _ = precision_recall_curve(binary[:, k], y_prob[:, k])
                    areas.append(auc(recall, precision))
                pr_auc = float(np.mean(areas))
        elif len(np.unique(y_true)) == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall, precision))
''' + s[end:]
# Remove the invalid orchestrator rather than leaving an executable fake-report path.
s = s[:s.index('def run_all_benchmarks(')] + '''def run_all_benchmarks(smoke_test=False):
    from dl.training.validated_experiments import main
    return main(smoke_test=smoke_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    run_all_benchmarks(smoke_test=args.smoke_test)
'''
# Unverified old checkpoints/embeddings are retained but cannot be silently reused.
s = s.replace('if chk_path.exists():', 'if False:  # Legacy caches have no reproducible experiment fingerprint.')
s = s.replace("if pooling_mode == 'attention' and chk_path.exists():", 'if False:  # Do not accept unverified legacy checkpoints.')
s = s.replace('if cache_file.exists():', 'if False:  # Do not accept unverified legacy embeddings.')
s = s.replace('freeze_backbone=True\n    ).to(device)', 'freeze_backbone=use_pretrained\n    ).to(device)')
s = s.replace('am.PatientPathologyMIL(pooling_mode=pooling_mode, feature_dim=128, pretrained=True)', "am.PatientPathologyMIL(backbone_name='resnet50', pooling_mode=pooling_mode, feature_dim=128, pretrained=True)")
p.write_text(s, encoding='utf-8')

p = root / 'dl/models/pathology_benchmarks.py'
s = p.read_text(encoding='utf-8')
for name in ('resnet18', 'resnet50', 'efficientnet_b0'):
    s = s.replace(f'''        try:
            self.backbone = models.{name}(weights=weights)
        except Exception:
            self.backbone = models.{name}(weights=None)''', f'''        # A failed pretrained download is an error, never a random-weight fallback.
        self.backbone = models.{name}(weights=weights)''')
s = s.replace('if freeze_backbone:', 'if freeze_backbone and pretrained:')
# Random models remain fully trainable for offline forward-pass tests.
p.write_text(s, encoding='utf-8')

p = root / 'dl/models/attention_mil.py'
s = p.read_text(encoding='utf-8').replace('        sample_in = torch.zeros(1, 3, 224, 224)', '        self.backbone.eval()\n        sample_in = torch.zeros(1, 3, 224, 224)')
p.write_text(s, encoding='utf-8')

p = root / 'dl/models/multimodal_fusion.py'
s = p.read_text(encoding='utf-8').replace('risk_score = self.risk_head(h).squeeze(-1)', 'risk_score = torch.sigmoid(prog_logit)  # Only expose the supervised risk head.')
p.write_text(s, encoding='utf-8')

p = root / 'dl/uncertainty_calibration.py'
s = p.read_text(encoding='utf-8').replace('        self.enable_dropout_only(model)', '        original_modes = {m: m.training for m in model.modules()}\n        self.enable_dropout_only(model)')
s = s.replace('                    probs = torch.softmax(out, dim=-1)', '                    probs = torch.sigmoid(out) if out.ndim == 1 or out.shape[-1] == 1 else torch.softmax(out, dim=-1)')
s = s.replace('        samples = np.stack(', '        for module, mode in original_modes.items():\n            module.training = mode\n        samples = np.stack(')
s = s.replace('uncertainty = std_prob * 2.0  # ~95% confidence interval half-width', 'uncertainty = std_prob * 2.0  # Scaled MC spread, not a confidence interval.')
s = s.replace('        self.temperature.data.fill_(1.5)', '        val_logits = val_logits.detach()\n        self.temperature.data.fill_(1.5)')
s = s.replace('        learned_temp = float(self.temperature.item())', '        with torch.no_grad():\n            self.temperature.clamp_(0.05, 10.0)\n        learned_temp = float(self.temperature.item())')
p.write_text(s, encoding='utf-8')

p = root / 'dl/explainability.py'
s = p.read_text(encoding='utf-8').replace('output = self.model(input_tensor)', 'output = self.model(input_tensor.detach().clone().requires_grad_(True))')
s = s.replace('f1_drop = max(0.0, base_f1 - perm_f1)', 'f1_drop = base_f1 - perm_f1')
p.write_text(s, encoding='utf-8')

p = root / 'dl/robustness_suite.py'
s = p.read_text(encoding='utf-8').replace('auc = 0.5', "auc = float('nan')")
p.write_text(s, encoding='utf-8')
