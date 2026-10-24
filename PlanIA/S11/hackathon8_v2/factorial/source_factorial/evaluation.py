"""Evaluación semántica y estructural, sin reparar las respuestas del modelo."""
import json
import time
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from dataset_generator import CLASSES
from validator import validate_json_output

def messages(row):
    return [{'role':'system','content':row['instruction']},
            {'role':'user','content':row['input']}]

def prompt(row, tokenizer):
    return tokenizer.apply_chat_template(messages(row),tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False)

def summarize(rows):
    truth=[r['truth']['prioridad'] for r in rows]
    predicted=[r['prediction'].get('prioridad','invalida') if r['schema_ok'] else 'invalida' for r in rows]
    ordinal={p:i for i,p in enumerate(CLASSES)}
    return dict(n=len(rows),macro_f1=f1_score(truth,predicted,labels=CLASSES,average='macro',zero_division=0),
                accuracy=accuracy_score(truth,predicted),json_rate=float(np.mean([r['json_ok'] for r in rows])),
                schema_rate=float(np.mean([r['schema_ok'] for r in rows])),
                exact_match=float(np.mean([r['schema_ok'] and r['truth']==r['prediction'] for r in rows])),
                critical_recall=recall_score(truth,predicted,labels=['critica'],average='macro',zero_division=0),
                ordinal_mae=float(np.mean([abs(ordinal[t]-ordinal[p]) if p in ordinal else 3 for t,p in zip(truth,predicted)])),
                escalation_accuracy=float(np.mean([r['schema_ok'] and r['prediction']['escalar']==r['truth']['escalar'] for r in rows])),
                blocking_accuracy=float(np.mean([r['schema_ok'] and r['prediction']['bloqueada']==r['truth']['bloqueada'] for r in rows])),
                criterion_accuracy=float(np.mean([r['schema_ok'] and r['prediction']['criterio']==r['truth']['criterio'] for r in rows])),
                mean_latency_s=float(np.mean([r['latency_s'] for r in rows])),
                p95_latency_s=float(np.percentile([r['latency_s'] for r in rows],95)))

def evaluate_model(model,tokenizer,rows,name,path,batch_size=2):
    """Latencia individual, batch=1, con calentamiento excluido y sincronización CUDA."""
    model.eval();tokenizer.padding_side='left'; results=[]
    warm=tokenizer(prompt(rows[0],tokenizer),return_tensors='pt').to('cuda')
    with torch.inference_mode():
        model.generate(**warm,max_new_tokens=2,do_sample=False,pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start_all=time.perf_counter()
    # Batch 1 gives interpretable request latency and avoids a batch-size confound.
    for i,row in enumerate(rows):
        inputs=tokenizer(prompt(row,tokenizer),return_tensors='pt').to('cuda')
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=96,do_sample=False,
                               use_cache=True,pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        raw=tokenizer.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
        valid=validate_json_output(raw)
        try: json.loads(raw);json_ok=True
        except (ValueError,TypeError): json_ok=False
        result=dict(id=row['id'],input=row['input'],group=row['group'],stress_type=row['stress_type'],
                    truth=json.loads(row['output']),raw=raw,prediction=valid.parsed_json or {},
                    json_ok=json_ok,schema_ok=valid.valid,error=valid.error,latency_s=elapsed,
                    generated_tokens=int(ids.shape[1]-inputs.input_ids.shape[1]))
        results.append(result)
        with open(path,'a',encoding='utf8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        if (i+1)%8==0: print(f'{name}: {i+1}/{len(rows)}',flush=True)
    metrics=summarize(results)
    metrics.update(inference_total_s=time.perf_counter()-start_all,
                   inference_peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                   inference_peak_reserved_gb=torch.cuda.max_memory_reserved()/1024**3)
    return metrics,results
