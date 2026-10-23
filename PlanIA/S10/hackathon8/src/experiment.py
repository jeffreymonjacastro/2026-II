"""Qwen3.5-4B: base, QLoRA y QDoRA en una única asignación Slurm.

Adaptación de main.ipynb y lora_eval_official.ipynb. Configuración centralizada,
plantilla nativa de Qwen, pérdida solo sobre la respuesta y evidencia en disco.
"""
import gc
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from evaluation import evaluate_model, prompt, messages

@dataclass
class Config:
    model_name: str = 'Qwen/Qwen3.5-4B'
    model_path: str = 'model'
    seed: int = 42
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 0.0002
    max_steps: int = 32
    gradient_accumulation_steps: int = 8
    batch_size: int = 1
    max_sequence_length: int = 1024
    target_modules: tuple = ('q_proj','v_proj','in_proj_qkv','out_proj')
    simulated_usd_per_gpu_hour: float = 1.0
    # Precio hipotético de referencia, no una tarifa real de Khipu.

ROOT=Path(__file__).resolve().parents[1]
CFG=Config(model_path=str(ROOT/'model'))
RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)

def save(name,obj):
    p=RESULTS/name;tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)

def load_rows(split):
    return [json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text(encoding='utf8').splitlines()]

def load_model():
    tokenizer=AutoTokenizer.from_pretrained(CFG.model_path,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=Qwen3_5ForCausalLM.from_pretrained(CFG.model_path,local_files_only=True,
                    quantization_config=quant,dtype=torch.bfloat16,device_map={'':'cuda:0'},
                    attn_implementation='sdpa')
    return model,tokenizer

def encode(row,tokenizer):
    prefix=prompt(row,tokenizer)
    # Prefix ends with Qwen's empty thinking block. Keep it identical at inference.
    full=prefix+row['output']+'<|im_end|>'
    a=tokenizer(prefix,add_special_tokens=False)['input_ids']
    b=tokenizer(full,add_special_tokens=False)['input_ids']
    assert b[:len(a)]==a, 'La plantilla no preserva el prefijo'
    assert len(b)<=CFG.max_sequence_length, f'Ejemplo demasiado largo: {len(b)}'
    labels=[-100]*len(a)+b[len(a):]
    return {k:torch.tensor([v],device='cuda') for k,v in dict(input_ids=b,attention_mask=[1]*len(b),labels=labels).items()}

def train_adapter(model,tokenizer,name):
    set_seed(CFG.seed)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={'use_reentrant':False})
    # Keep the frozen, large embedding/head in BF16; only small norms remain FP32.
    model.get_input_embeddings().to(torch.bfloat16)
    model.get_output_embeddings().to(torch.bfloat16)
    model.config.use_cache=False
    config=LoraConfig(r=CFG.rank,lora_alpha=CFG.alpha,lora_dropout=CFG.dropout,
                      target_modules=list(CFG.target_modules),task_type='CAUSAL_LM',
                      bias='none',use_dora=name=='dora')
    model=get_peft_model(model,config)
    rows=load_rows('train');random.Random(CFG.seed).shuffle(rows)
    # Both methods see the same sequence of examples, from independent base reloads.
    encoded=[encode(row,tokenizer) for row in rows]
    lengths=[d['input_ids'].shape[1] for d in encoded]
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CFG.learning_rate,weight_decay=.01)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter();history=[];model.train();optimizer.zero_grad(set_to_none=True)
    for step in range(CFG.max_steps):
        warm=min(1.,(step+1)/2)
        factor=warm*.5*(1+np.cos(np.pi*step/CFG.max_steps))
        for group in optimizer.param_groups:group['lr']=CFG.learning_rate*factor
        losses=[]
        for j in range(CFG.gradient_accumulation_steps):
            batch=encoded[(step*CFG.gradient_accumulation_steps+j)%len(encoded)]
            n=int((batch['labels']!=-100).sum())
            logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                         use_cache=False,logits_to_keep=n+1).logits
            # Compute next-token loss only where there is an assistant target.
            loss=F.cross_entropy(logits[0,:-1].float(),batch['labels'][0,-n:])
            losses.append(float(loss.detach()));(loss/CFG.gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        entry=dict(step=step+1,loss=float(np.mean(losses)),lr=float(CFG.learning_rate*factor),elapsed_s=time.perf_counter()-start)
        history.append(entry)
        print(name,entry,flush=True)
        save(f'{name}_training_history.json',history)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    metadata=dict(train_seconds=elapsed,train_peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
        train_peak_reserved_gb=torch.cuda.max_memory_reserved()/1024**3,trainable_parameters=trainable,
        seed=CFG.seed,rank=CFG.rank,alpha=CFG.alpha,method=name,
        steps=CFG.max_steps,examples_seen=CFG.max_steps*CFG.gradient_accumulation_steps,
        sequence_length_min=min(lengths),sequence_length_max=max(lengths),final_loss=history[-1]['loss'],
        simulated_train_cost_usd=elapsed/3600*CFG.simulated_usd_per_gpu_hour)
    directory=ROOT/'adapters'/name;directory.mkdir(exist_ok=True,parents=True)
    model.save_pretrained(directory);tokenizer.save_pretrained(directory)
    save(f'{name}_training.json',metadata)
    del optimizer,encoded;gc.collect();torch.cuda.empty_cache()
    model.gradient_checkpointing_disable();model.config.use_cache=True;model.eval()
    return model,metadata

def main():
    assert torch.cuda.is_available(),'Debe ejecutarse en un nodo GPU asignado por Slurm'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    os.environ['TOKENIZERS_PARALLELISM']='false'
    set_seed(CFG.seed)
    versions=subprocess.run([os.sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout
    (ROOT/'requirements-lock.txt').write_text(versions,encoding='utf8')
    hardware=dict(gpu=torch.cuda.get_device_name(0),total_vram_gb=torch.cuda.get_device_properties(0).total_memory/1024**3,
                  node=os.uname().nodename,job_id=os.environ.get('SLURM_JOB_ID'),cuda=torch.version.cuda,
                  torch=torch.__version__,config=asdict(CFG))
    save('hardware.json',hardware); print(hardware,flush=True)
    all_metrics={}
    for name in ['base','lora','dora','lora_replica']:
        CFG.seed=43 if name=='lora_replica' else 42
        if (RESULTS/f'{name}_metrics.json').exists():
            previous=json.loads((RESULTS/f'{name}_metrics.json').read_text())
            if all(k in previous for k in ['validation','test','stress']):
                all_metrics[name]=previous;continue
        print('START',name,flush=True);model,tokenizer=load_model();training={}
        if name!='base':model,training=train_adapter(model,tokenizer,name)
        metrics={'training':training}
        # Fixed settings, no hyperparameter search and no selection on the test set.
        for split in ['validation','test','stress']:
            path=RESULTS/f'{name}_{split}_predictions.jsonl'
            if path.exists():path.unlink()
            metrics[split],_=evaluate_model(model,tokenizer,load_rows(split),f'{name}/{split}',path)
            save(f'{name}_metrics.json',metrics)
        all_metrics[name]=metrics
        del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    save('all_metrics.json',all_metrics)
    save('COMPLETE.json',{'status':'completed','methods':list(all_metrics)})
    print('HACKATHON_COMPLETE',flush=True)

if __name__=='__main__':main()
