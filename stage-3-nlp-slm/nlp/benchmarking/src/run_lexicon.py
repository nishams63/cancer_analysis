"""Additional transparent train-only span lexicon, keeping frozen triage heads.

No manually invented annotations or validation vocabulary. Longest non-overlapping
train-annotated phrase matches replace hand-written extraction dictionaries.
"""
import collections
import re
import time
from protocol import *

class TrainSpanLexicon:
    def __init__(self,terms):
        self.terms=terms
        self.lookup={t['phrase']:t['label'] for t in terms}
        patterns=[r'\s+'.join(map(re.escape,t['phrase'].split())) for t in sorted(terms,key=lambda t:-len(t['phrase']))]
        self.pattern=re.compile(r'(?<!\w)(?:'+ '|'.join(patterns)+r')(?!\w)',re.I)

    def extract(self,text):
        from negation_detection import resolve_concept_polarity
        output=[]
        for match in self.pattern.finditer(text):
            key=' '.join(match.group().lower().split())
            output.append({'start':match.start(),'end':match.end(),'label':self.lookup[key],
                'text':match.group(),'polarity':resolve_concept_polarity(text,match.start(),match.end())})
        return output

def run():
    train,val=load_development('TRAIN'),load_development('VALIDATION');split_audit(train,val)
    start=time.perf_counter(); counts=collections.defaultdict(collections.Counter)
    for row in train.itertuples():
        for e in json.loads(row.ner_entities):
            phrase=' '.join(row.text[e['start']:e['end']].lower().split())
            if phrase: counts[phrase][e['label']]+=1
    terms=[{'phrase':p,'label':c.most_common(1)[0][0],'train_support':sum(c.values())} for p,c in counts.items()]
    model=TrainSpanLexicon(terms); training=time.perf_counter()-start
    out=BENCH/'models'/'train_span_lexicon';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'terms.json',terms)
    gold=[json.loads(e) for e in val.ner_entities]
    start=time.perf_counter();pred=[model.extract(t) for t in val.text];elapsed=time.perf_counter()-start
    result={'name':'train_span_lexicon','status':'EVALUATED','split':'VALIDATION',
        'protocol':'TRAIN annotated phrase lexicon + unchanged frozen urgency/hazard classifiers',
        'training_seconds':training,'ner_seconds_per_document':elapsed/len(val),
        'seconds_per_document':None,'parameter_count':None,'learned_phrase_count':len(terms),
        'ner_exact':span_metrics(gold,pred),'ner_relaxed':span_metrics(gold,pred,True)}
    # Reuse real stored predictions for the identical frozen classification component.
    baseline=json.loads((BENCH/'results'/'validation'/'baseline_a.json').read_text())
    result['urgency']=baseline['urgency'];result['hazard']=baseline['hazard']
    result['classification_provenance']='Identical frozen heads and original feature pipeline; baseline_a predictions reused without alteration'
    write_json(BENCH/'results'/'validation'/'train_span_lexicon.json',result)
    write_json(BENCH/'results'/'predictions'/'train_span_lexicon.json',[
        {'document_hash':digest(str(row.document_id)),'entities':e} for row,e in zip(val.itertuples(),pred)])
    # Source-token boundary preserving whitespace robustness on the same held-out documents.
    modified=[]; shifted=[]
    for text,entities in zip(val.text,gold):
        positions=[]; chars=[]
        for c in text:
            positions.append(len(chars)); chars.extend('  ' if c==' ' else c)
        positions.append(len(chars)); modified.append(''.join(chars))
        shifted.append([{**e,'start':positions[e['start']],'end':positions[e['end']]} for e in entities])
    robust=span_metrics(shifted,[model.extract(t) for t in modified])
    write_json(BENCH/'results'/'metrics'/'lexicon_whitespace_robustness.json',{
        'documents':len(val),'condition':'double_spaces','clean_exact_f1':result['ner_exact']['micro']['f1'],
        'perturbed_exact_f1':robust['micro']['f1'],'predictions_reproducible':pred==[model.extract(t) for t in val.text]})
    print('Train-only span lexicon exact F1',result['ner_exact']['micro']['f1'],flush=True)

if __name__=='__main__': run()
