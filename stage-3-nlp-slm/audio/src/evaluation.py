"""Paired reference-based evaluation with no reference-assisted transcription."""
import re
from difflib import SequenceMatcher

def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def canonical(text): return ' '.join(re.findall(r'\w+(?:\.\d+)?',text.casefold()))

def error_counts(reference,hypothesis):
    r,h=canonical(reference),canonical(hypothesis)
    return {'word_errors':edit_distance(r.split(),h.split()),'reference_words':len(r.split()),
            'character_errors':edit_distance(r,h),'reference_characters':len(r)}

def preservation(reference,hypothesis,entities):
    h=canonical(hypothesis)
    result={}
    for label in ['DRUG_NAME','DOSAGE','GENE_MUTATION','ADVERSE_EVENT']:
        terms=[canonical(reference[e['start']:e['end']]) for e in entities if e['label']==label]
        result[label]={'preserved':sum((' '+t+' ') in (' '+h+' ') for t in terms),'total':len(terms)}
    for label,pattern in [('NUMERIC',r'\b\d+(?:\.\d+)?\b'),('NEGATION',r'\b(?:no|not|denies|without|negative)\b')]:
        terms=re.findall(pattern,reference,re.I)
        result[label]={'preserved':sum(len(re.findall(r'\b'+re.escape(t)+r'\b',hypothesis,re.I))>=
                                      terms[:i+1].count(t) for i,t in enumerate(terms)),'total':len(terms)}
    return result

def project_entities(reference,hypothesis,entities):
    """Map unchanged hypothesis spans to original coordinates; unmatched spans stay FP.

    Only reference-aligned exact character runs qualify for exact NER. This strict
    metric intentionally penalizes STT substitutions and deletions. No gold entities
    are removed from the denominator.
    """
    mapping={j+k:i+k for i,j,n in SequenceMatcher(None,reference,hypothesis,autojunk=False).get_matching_blocks() for k in range(n)}
    projected=[]
    for e in entities:
        positions=[mapping.get(i) for i in range(e['start'],e['end'])]
        valid=bool(positions) and all(p is not None for p in positions)
        valid=valid and positions==list(range(positions[0],positions[0]+len(positions)))
        projected.append({**e,'start':positions[0] if valid else -1,'end':positions[-1]+1 if valid else -1})
    return projected
