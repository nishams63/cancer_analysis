"""Rerun existing controlled context cases without touching old evaluation outputs."""
import ast
from protocol import *
from negation_detection import resolve_concept_polarity

def run():
    source=ROOT/'evaluation'/'src'/'negation_metrics.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cases=None
    for node in tree.body:
        if isinstance(node,ast.AnnAssign) and getattr(node.target,'id',None)=='DIAGNOSTIC_CASES':
            cases=ast.literal_eval(node.value)
    if cases is None: raise RuntimeError('Diagnostic case definition unavailable')
    records=[]
    for case in cases:
        text=case['sentence']; start=text.casefold().find(case['concept'].casefold())
        pred=resolve_concept_polarity(text,start,start+len(case['concept']))
        records.append({'category':case['category'],'expected':case['expected_polarity'],
            'predicted':pred,'correct':pred==case['expected_polarity']})
    result={'diagnostic_cases':len(records),'correct':sum(r['correct'] for r in records),
        'accuracy':sum(r['correct'] for r in records)/len(records),'cases':records,
        'scope':'Controlled diagnostic cases only, not corpus-wide context accuracy'}
    write_json(BENCH/'results'/'metrics'/'context_diagnostics.json',result)
    print(result['correct'],'/',result['diagnostic_cases'])

if __name__=='__main__':run()
