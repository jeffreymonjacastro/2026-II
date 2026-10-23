"""Factorial 3x2 duplicado, confirmación fuera del dominio y comparación PEFT.

No selecciona hiperparámetros con prueba ni con el conjunto de confirmación.
Cada condición se recarga desde el modelo original, dentro de un único job.
"""
import gc,json,os,random,time,hashlib,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from transformers import set_seed
import experiment as e
from evaluation import evaluate_model

ROOT=Path(__file__).resolve().parents[1]
FROOT=ROOT/'factorial';RESULTS=FROOT/'results';RESULTS.mkdir(parents=True,exist_ok=True)
e.RESULTS=RESULTS
PLAN=json.loads((FROOT/'plan.json').read_text(encoding='utf8'))

def save(name,obj):e.save(name,obj)
def ident(r,a,seed,method='lora'):return f'fx_{method}_r{r}_a{a}_s{seed}'
def run(r,a,seed,phase,method='lora',extra=False):
    name=ident(r,a,seed,method)
    destination=RESULTS/f'{name}_metrics.json'
    if destination.exists():
        m=json.loads(destination.read_text());
        if m.get('complete'):
            t=m['training']
            assert (t['rank'],t['alpha'],t['seed'],t['steps'])==(r,a,seed,PLAN['steps']), 'El caché pertenece a otra configuración'
            return m
    e.CFG=e.Config(model_path=str(ROOT/'model'),rank=r,alpha=a,seed=seed,warmup_before_training=True,
                   use_rslora=method=='rslora',loraplus_lr_ratio=16.0 if method=='loraplus' else 1.0)
    set_seed(seed);started=time.perf_counter()
    print('FACTORIAL_START',name,phase,flush=True)
    model,tokenizer=e.load_model()
    # Original helper's method flag is literal 'dora'. Save all artifacts under
    # the unique condition name by moving its fresh outputs immediately.
    train_name='dora' if method=='dora' else name
    if method=='dora':
        # Do not overwrite the earlier course deliverable's DoRA adapter.
        original_root=e.ROOT
        stage=FROOT/'training_stage';stage.mkdir(exist_ok=True)
        e.ROOT=stage
        # load_rows must continue reading the frozen original partitions.
        original_loader=e.load_rows
        e.load_rows=lambda split: [json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text(encoding='utf8').splitlines()]
    model,training=e.train_adapter(model,tokenizer,train_name)
    if method=='dora':
        import shutil
        target=ROOT/'adapters'/name
        assert not target.exists(),target
        shutil.move(str(stage/'adapters'/'dora'),str(target))
        for suffix in ['training.json','training_history.json']:
            p=RESULTS/f'dora_{suffix}';p.replace(RESULTS/f'{name}_{suffix}')
        e.ROOT=original_root;e.load_rows=original_loader
    training.update(method=method,condition=name,phase=phase)
    save(f'{name}_training.json',training)
    m={'condition':name,'method':method,'phase':phase,'training':training,'config':asdict(e.CFG),'complete':False}
    for split in ['validation','test','stress']:
        p=RESULTS/f'{name}_{split}_predictions.jsonl'
        if p.exists():p.unlink()
        m[split],_=evaluate_model(model,tokenizer,e.load_rows(split),name+'/'+split,p)
        save(destination.name,m)
    m['wall_seconds']=time.perf_counter()-started;m['complete']=True
    save(destination.name,m)
    del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    print('FACTORIAL_DONE',name,'validation',m['validation']['macro_f1'],'test',m['test']['macro_f1'],flush=True)
    return m

def choose(records,include_extra=False):
    # A practical threshold fixed before the runs: 0.02 absolute macro F1.
    cells=[]
    pairs=sorted({(m['training']['rank'],m['training']['alpha']) for m in records.values() if m['method']=='lora'}) if include_extra else [(r,a) for r in PLAN['ranks'] for a in PLAN['alphas']]
    for r,a in pairs:
            ms=[records[ident(r,a,s)] for s in PLAN['seeds']]
            cells.append({'rank':r,'alpha':a,'validation_f1':float(np.mean([m['validation']['macro_f1'] for m in ms])),
                          'train_seconds':float(np.mean([m['training']['train_seconds'] for m in ms]))})
    top=max(c['validation_f1'] for c in cells)
    eligible=[c for c in cells if c['validation_f1']>=top-PLAN['practical_delta']]
    # Simpler capacity first among practically close cells; then training time.
    selected=min(eligible,key=lambda c:(c['rank'],c['train_seconds'],c['alpha']))
    save('selection.json',{'rule':'mean validation F1 within 0.02 of best, then smallest rank, then lowest mean training time',
                           'best_validation_f1':top,'selected':selected,'cells':cells})
    return selected

def propose(records):
    """Polynomial response surface fitted only to the 12 validation scores."""
    X=[];y=[]
    for m in records.values():
        t=m['training'];x=(np.log2(t['rank'])-3)/2;z=-1 if t['alpha']==8 else 1;q=x*x-2/3
        X.append([1,x,q,z,x*z,q*z,-1 if t['seed']==42 else 1]);y.append(m['validation']['macro_f1'])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    grid=[]
    for r in range(1,65):
        for a in PLAN['alphas']:
            x=(np.log2(r)-3)/2;z=-1 if a==8 else 1;q=x*x-2/3
            pred=float(np.dot(beta,[1,x,q,z,x*z,q*z,0]))
            grid.append({'rank':r,'alpha':a,'predicted_validation_f1':pred})
    candidate=max(grid,key=lambda d:d['predicted_validation_f1'])
    save('theoretical_optimum.json',{'formula':'F1 ~ x + q + alpha + x:alpha + q:alpha + seed_block; x=(log2(r)-3)/2; q=x^2-2/3',
         'coefficients':beta.tolist(),'candidate':candidate,'domain_factorial':[2,32],
         'candidate_search':[1,64],'note':'Predictions beyond [2,32] are extrapolations, requiring actual duplicate confirmation.'})
    return candidate

def confirmation(records,selected):
    rows=[json.loads(s) for s in (FROOT/'data'/'confirmation.jsonl').read_text(encoding='utf8').splitlines()]
    # New untouched cases, opened only after selection. Freeze selection first.
    targets=[('base',None)]+[(ident(selected['rank'],selected['alpha'],s,m),m) for m in ['lora','dora'] for s in PLAN['seeds']]
    from peft import PeftModel
    out={}
    for name,method in targets:
        p=RESULTS/f'{name}_confirmation_metrics.json'
        if p.exists():out[name]=json.loads(p.read_text());continue
        set_seed(42);model,tokenizer=e.load_model()
        if method:
            # Restore the same frozen-norm precision used during training.
            model=e.prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
            model.get_input_embeddings().to(torch.bfloat16)
            model.get_output_embeddings().to(torch.bfloat16)
            model=PeftModel.from_pretrained(model,str(ROOT/'adapters'/name))
            model.eval();model.config.use_cache=True
        dest=RESULTS/f'{name}_confirmation_predictions.jsonl'
        if dest.exists():dest.unlink()
        metrics,_=evaluate_model(model,tokenizer,rows,name+'/confirmation',dest)
        save(p.name,metrics);out[name]=metrics
        del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    save('confirmation_metrics.json',out)

def main():
    assert torch.cuda.is_available(),'Se requiere Slurm con GPU'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    assert torch.cuda.get_device_name(0)=='NVIDIA RTX A6000'
    h={'gpu':torch.cuda.get_device_name(0),'node':os.uname().nodename,'job_id':os.environ.get('SLURM_JOB_ID'),
       'cuda':torch.version.cuda,'torch':torch.__version__,'plan':PLAN,
       'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.jsonl')}}
    h['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/experiment.py',ROOT/'src/evaluation.py']}
    h['session_started_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    history=RESULTS/'hardware_sessions.json'
    sessions=json.loads(history.read_text()) if history.exists() else []
    if (RESULTS/'hardware.json').exists() and not sessions:sessions.append(json.loads((RESULTS/'hardware.json').read_text()))
    sessions.append(h);save('hardware_sessions.json',sessions)
    save('hardware.json',h)
    (FROOT/'requirements-lock.txt').write_text(subprocess.run([os.sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
    records={}
    for row in PLAN['execution_order']:
        m=run(row['rank'],row['alpha'],row['seed'],row['phase']);records[m['condition']]=m
        save('all_metrics.json',records)
    selection=choose(records)
    proposed=propose(records)
    if proposed['rank'] not in PLAN['ranks']:
        for seed in PLAN['seeds']:
            m=run(proposed['rank'],proposed['alpha'],seed,'theoretical_optimum_check')
            records[m['condition']]=m;save('all_metrics.json',records)
    for seed in PLAN['seeds']:
        m=run(PLAN['confirmation_rank'],selection['alpha'],seed,'outside_domain')
        records[m['condition']]=m;save('all_metrics.json',records)
    selection=choose(records,include_extra=True)
    for seed in PLAN['seeds']:
        m=run(selection['rank'],selection['alpha'],seed,'technology_comparison','dora')
        records[m['condition']]=m;save('all_metrics.json',records)
    confirmation(records,selection)
    save('COMPLETE.json',{'status':'completed','conditions':list(records),'selected':selection})
    print('FACTORIAL_COMPLETE',flush=True)

if __name__=='__main__':main()
