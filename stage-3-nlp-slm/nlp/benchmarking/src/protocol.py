"""Development-only data access, immutable-source auditing and shared metrics."""
from pathlib import Path
import hashlib
import json
import re
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'nlp' / 'benchmarking'
sys.path.insert(0, str(ROOT / 'nlp' / 'src'))
LABELS = ['GENE_MUTATION', 'DRUG_NAME', 'DOSAGE', 'ADVERSE_EVENT']

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def inventory():
    """Hash protected files without interpreting locked-test content."""
    paths = []
    for folder in ['data-engineering', 'eda', 'evaluation', 'nlp']:
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and 'benchmarking' not in p.parts and '__pycache__' not in p.parts
                     and '.pytest_cache' not in p.parts)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def load_development(split):
    if split not in ('TRAIN', 'VALIDATION'):
        raise ValueError('Locked test is forbidden in development')
    from config import TRAIN_PARQUET_PATH, VAL_PARQUET_PATH
    path = TRAIN_PARQUET_PATH if split == 'TRAIN' else VAL_PARQUET_PATH
    df = pd.read_parquet(path)
    if set(df.data_split) != {split}:
        raise ValueError('Manifest partition mismatch')
    if (pd.to_datetime(df.document_date) > pd.to_datetime(df.index_date)).any():
        raise ValueError('Future observation beyond index date')
    for row in df.itertuples():
        for ent in json.loads(row.ner_entities):
            if not 0 <= ent['start'] < ent['end'] <= len(row.text):
                raise ValueError('Invalid annotation offsets')
    return df

def split_audit(train, val):
    result = {c: len(set(train[c]) & set(val[c]))
              for c in ['patient_id', 'encounter_id', 'document_id']}
    normalize = lambda t: re.sub(r'\s+', ' ', t).strip().casefold()
    result['canonical_text'] = len(set(map(normalize, train.text)) & set(map(normalize, val.text)))
    if any(result.values()):
        raise ValueError(f'Development split leakage: {result}')
    return result

def classification(y, pred, labels):
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    return {'accuracy': accuracy_score(y, pred), 'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'critical_recall': report['CRITICAL']['recall'] if report.get('CRITICAL', {}).get('support',0)>0 else None,
            'per_class': {k: report[k] for k in labels},
            'confusion_matrix': confusion_matrix(y, pred, labels=labels).tolist(), 'labels': labels}

def span_metrics(golds, predictions, relaxed=False):
    """Maximum one-to-one bipartite matching prevents double counting."""
    counts = {label: [0, 0, 0] for label in LABELS}
    for gold, preds in zip(golds, predictions, strict=True):
        for label in LABELS:
            gs = [g for g in gold if g['label'] == label]
            ps = [p for p in preds if p['label'] == label]
            edges = [[i for i,g in enumerate(gs) if
                      (max(p['start'],g['start']) < min(p['end'],g['end']) if relaxed else
                       (p['start'],p['end']) == (g['start'],g['end']))] for p in ps]
            matched = {}
            def visit(j, seen):
                for i in edges[j]:
                    if i in seen: continue
                    seen.add(i)
                    if i not in matched or visit(matched[i], seen):
                        matched[i] = j
                        return True
                return False
            tp = sum(visit(j,set()) for j in range(len(ps)))
            counts[label] = [a+b for a,b in zip(counts[label],[tp,len(ps),len(gs)])]
    def summarize(c):
        tp,npred,ngold = c
        precision = tp/npred if npred else 0.
        recall = tp/ngold if ngold else 0.
        return {'precision':precision, 'recall':recall,
                'f1':2*tp/(npred+ngold) if npred+ngold else 0.,
                'tp':tp,'predicted':npred,'gold':ngold}
    return {'micro':summarize([sum(c[i] for c in counts.values()) for i in range(3)]),
            'per_type':{k:summarize(v) for k,v in counts.items()}}
