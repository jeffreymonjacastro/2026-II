"""Añade rsLoRA y LoRA+ al LoRA seleccionado, con dos semillas cada uno."""
import gc,json,os,time,hashlib
from pathlib import Path
import torch
from transformers import set_seed
from peft import PeftModel
import factorial_runner as f
from evaluation import evaluate_model

ROOT=f.ROOT;R=f.RESULTS
def main():
    assert (R/'COMPLETE.json').exists(),'Se debe terminar primero el factorial'
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA RTX A6000'
    assert os.uname().nodename=='g002','Mantener el nodo del factorial'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    selected=json.loads((R/'selection.json').read_text())['selected']
    r,a=selected['rank'],selected['alpha']
    hardware=dict(job_id=os.environ.get('SLURM_JOB_ID'),node=os.uname().nodename,gpu=torch.cuda.get_device_name(0),
       torch=torch.__version__,cuda=torch.version.cuda,rank=r,alpha=a,seeds=[42,43],
       rslora='use_rslora=True; scale=alpha/sqrt(r)',loraplus='PEFT create_loraplus_optimizer; AdamW; lr_A=0.0002; lr_B=0.0032; ratio=16',
       scheduler='Cosine multiplier applied to each original parameter-group rate, preserving B/A=16 at every step',
       source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/experiment.py',ROOT/'src/factorial_runner.py']})
    f.save('additional_peft_hardware.json',hardware)
    records=json.loads((R/'all_metrics.json').read_text())
    for method in ['rslora','loraplus']:
        for seed in [42,43]:
            m=f.run(r,a,seed,'additional_technology_comparison',method)
            records[m['condition']]=m;f.save('all_metrics.json',records)
    if r==1:
        # At rank1, alpha/r equals alpha/sqrt(r): add a nondegenerate control.
        for seed in [42,43]:
            m=f.run(32,a,seed,'rslora_high_rank_control','rslora')
            records[m['condition']]=m;f.save('all_metrics.json',records)
        f.save('rslora_high_rank_control.json',{'reason':'At r=1 the standard and rank-stabilized scaling are identical.',
                 'rank':32,'alpha':a,'seeds':[42,43],
                 'baseline_conditions':[f.ident(32,a,s,'lora') for s in [42,43]],
                 'control_conditions':[f.ident(32,a,s,'rslora') for s in [42,43]],
                 'scope':'Mechanism control on original validation/test/stress only, not a new selection using confirmation.'})
    # All method choices are fixed by validation before these new responses.
    scores={method:sum(records[f.ident(r,a,s,method)]['validation']['macro_f1'] for s in [42,43])/2 for method in ['lora','dora','rslora','loraplus']}
    best=max(scores,key=scores.get)
    f.save('technology_selection.json',{'criterion':'Maximum mean validation macro F1; no selection using confirmation',
                                      'validation_means':scores,'best_validation_method':best})
    confirmation=json.loads((R/'confirmation_metrics.json').read_text())
    rows=[json.loads(s) for s in (ROOT/'factorial/data/confirmation.jsonl').read_text(encoding='utf8').splitlines()]
    for method in ['rslora','loraplus']:
        for seed in [42,43]:
            name=f.ident(r,a,seed,method);p=R/f'{name}_confirmation_metrics.json'
            if p.exists():confirmation[name]=json.loads(p.read_text());continue
            set_seed(seed);model,tokenizer=f.e.load_model()
            model=f.e.prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
            model.get_input_embeddings().to(torch.bfloat16);model.get_output_embeddings().to(torch.bfloat16)
            model=PeftModel.from_pretrained(model,str(ROOT/'adapters'/name));model.eval();model.config.use_cache=True
            dest=R/f'{name}_confirmation_predictions.jsonl'
            if dest.exists():dest.unlink()
            metrics,_=evaluate_model(model,tokenizer,rows,name+'/confirmation',dest)
            f.save(p.name,metrics);confirmation[name]=metrics
            del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    f.save('confirmation_metrics.json',confirmation)
    f.save('COMPLETE_PEFT.json',{'status':'completed','methods':['lora','dora','rslora','loraplus'],
                              'seeds':[42,43],'total_training_conditions':len(records),
                              'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print('PEFT_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':main()
