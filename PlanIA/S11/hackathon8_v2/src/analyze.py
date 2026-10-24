"""Gráficas, intervalos descriptivos y tablas a partir de predicciones reales."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
LABELS=['critica','alta','media','baja']
NAMES={'base':'Base','lora':'LoRA','dora':'DoRA','lora_replica':'LoRA, réplica'}
COLORS=['#65758B','#167C80','#CE7837','#4054A0']

def predictions(name,split='test'):
    return [json.loads(s) for s in (ROOT/'results'/f'{name}_{split}_predictions.jsonl').read_text(encoding='utf8').splitlines()]

def macro(rows,indices=None):
    rows=rows if indices is None else [rows[i] for i in indices]
    y=[r['truth']['prioridad'] for r in rows]
    p=[r['prediction'].get('prioridad','invalida') if r['schema_ok'] else 'invalida' for r in rows]
    return f1_score(y,p,labels=LABELS,average='macro',zero_division=0)

def intervals(all_rows,n=1000):
    """Remuestreo pareado por clase; mide incertidumbre de casos, no de semillas."""
    rng=np.random.default_rng(20261023)
    first=next(iter(all_rows.values()))
    groups=[np.array([i for i,r in enumerate(first) if r['truth']['prioridad']==p]) for p in LABELS]
    score={k:[] for k in all_rows}
    for _ in range(n):
        idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        for name,rows in all_rows.items():score[name].append(macro(rows,idx))
    ci={k:[float(x) for x in np.quantile(v,[.025,.975])] for k,v in score.items()}
    delta={k:[float(x) for x in np.quantile(np.array(v)-np.array(score['base']),[.025,.975])]
           for k,v in score.items() if k!='base'}
    return ci,delta

def main():
    metrics=json.loads((ROOT/'results'/'all_metrics.json').read_text())
    names=[n for n in NAMES if n in metrics]
    assert all(all(s in metrics[n] for s in ['validation','test','stress']) for n in names)
    figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    all_rows={n:predictions(n) for n in names}
    for n in names:assert [r['id'] for r in all_rows[n]]==[r['id'] for r in all_rows['base']]
    for split in ['validation','test','stress']:
        expected={r['id']:r for r in [json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text(encoding='utf8').splitlines()]}
        for name in names:
            rows=predictions(name,split)
            assert len(rows)==len(expected)
            for row in rows:
                source=expected[row['id']]
                assert row['input']==source['input'] and row['truth']==json.loads(source['output']), 'Resultados de otra versión de los datos'
    ci,delta=intervals(all_rows)
    summary=[]
    for name in names:
        m=metrics[name];t=m['training'];test=m['test']
        # Additional diagnosis; primary scores still require the complete schema.
        priority_only=[];duplicate_keys=0
        for row in all_rows[name]:
            try:
                pairs=json.loads(row['raw'],object_pairs_hook=lambda x:x)
                if not isinstance(pairs,list):raise ValueError()
                keys=[k for k,v in pairs]
                duplicate_keys+=int(len(keys)!=len(set(keys)))
                values=[v for k,v in pairs if k=='prioridad']
                priority_only.append(values[0] if len(values)==1 and values[0] in LABELS else 'invalida')
            except (ValueError,TypeError):priority_only.append('invalida')
        priority_f1=f1_score([r['truth']['prioridad'] for r in all_rows[name]],priority_only,
                             labels=LABELS,average='macro',zero_division=0)
        summary.append(dict(method=name,label=NAMES[name],seed=t.get('seed',42),
          macro_f1=test['macro_f1'],ci_low=ci[name][0],ci_high=ci[name][1],
          json_rate=test['json_rate'],schema_rate=test['schema_rate'],exact_match=test['exact_match'],
          critical_recall=test['critical_recall'],ordinal_mae=test['ordinal_mae'],
          train_min=t.get('train_seconds',0)/60,train_vram_gb=t.get('train_peak_allocated_gb',0),
          latency_s=test['mean_latency_s'],stress_f1=m['stress']['macro_f1'],
          priority_only_f1=priority_f1,duplicate_key_cases=duplicate_keys,
          mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in all_rows[name]])),
          generated_tokens_per_second=float(sum(r['generated_tokens'] for r in all_rows[name])/sum(r['latency_s'] for r in all_rows[name])),
          train_cost_usd=t.get('simulated_train_cost_usd',0)))
    df=pd.DataFrame(summary);df.to_csv(ROOT/'results'/'summary.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
    # Ordered estimates and interval lengths keep close results readable.
    fig,ax=plt.subplots(figsize=(8.8,4.8))
    ys=np.arange(len(df))[::-1]
    for i,row in df.iterrows():
        y=ys[i]
        ax.plot([row.ci_low,row.ci_high],[y,y],color=COLORS[i],lw=4,alpha=.45)
        ax.scatter(row.macro_f1,y,s=130,color=COLORS[i],zorder=3)
        ax.text(1.025,y,f'{row.macro_f1:.3f}',va='center',fontsize=13,fontweight='bold',color=COLORS[i])
    ax.set(yticks=ys,yticklabels=df.label,xlim=(0,1.14),ylim=(-.65,3.65),
           xlabel='F1 macro (0 a 1)',title='LoRA y DoRA mejoran frente al modelo original')
    ax.set_xticks(np.arange(0,1.01,.2));ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.text(0,-.23,'64 casos sintéticos. Segmentos: IC 95 % por remuestreo estratificado.',transform=ax.transAxes,fontsize=10)
    fig.tight_layout();fig.savefig(figs/'desempeno.png',dpi=200,bbox_inches='tight');plt.close(fig)
    adapted=df[df.method!='base'].reset_index(drop=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4.1))
    for ax,col,title,unit in zip(axes,['train_min','train_vram_gb','latency_s'],
        ['Entrenamiento','Memoria GPU','Inferencia sin fusionar'],['Minutos','GiB asignados por PyTorch','Segundos por solicitud']):
        ax.barh(np.arange(3)[::-1],adapted[col],color=COLORS[1:],height=.5)
        ax.set(yticks=np.arange(3)[::-1],yticklabels=['LoRA','DoRA','LoRA, réplica'],title=title,xlabel=unit)
        for i,v in enumerate(adapted[col]):ax.text(v+adapted[col].max()*.035,2-i,f'{v:.2f}',va='center',fontweight='bold')
        ax.set_xlim(0,adapted[col].max()*1.3);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('El empate de desempeño tiene un costo diferente',fontsize=15,fontweight='bold')
    fig.tight_layout();fig.savefig(figs/'recursos.png',dpi=200,bbox_inches='tight');plt.close(fig)
    l=df[df.method=='lora'].iloc[0];d=df[df.method=='dora'].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios':[1,1.35]})
    ax=axes[0]
    ax.bar(['LoRA','DoRA'],[l.macro_f1,d.macro_f1],color=[COLORS[1],COLORS[2]],width=.55)
    for i,v in enumerate([l.macro_f1,d.macro_f1]):ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=16,fontweight='bold')
    ax.set(ylim=(0,1.06),ylabel='F1 macro',title='Mismo desempeño en prueba')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];ratios=np.array([d.train_min/l.train_min,d.train_vram_gb/l.train_vram_gb,d.latency_s/l.latency_s])
    ax.barh([2,1,0],ratios,color=COLORS[2],height=.5)
    ax.axvline(1,color=COLORS[1],ls='--',lw=1.5)
    for y,v in zip([2,1,0],ratios):ax.text(v+.08,y,f'{v:.2f}x',va='center',fontweight='bold',fontsize=13)
    ax.set(yticks=[2,1,0],yticklabels=['Tiempo de entrenamiento','Memoria GPU','Latencia sin fusionar'],xlim=(0,5.2),
           xlabel='Cociente DoRA / LoRA',title='Mayor consumo observado con DoRA')
    ax.text(.02,-.23,'La línea marca LoRA = 1. Mayor valor: más consumo.',transform=ax.transAxes,fontsize=9)
    fig.suptitle('Comparación PEFT: LoRA frente a DoRA',fontsize=16,fontweight='bold')
    fig.tight_layout(rect=[0,.03,1,.94]);fig.savefig(figs/'comparacion_peft.png',dpi=200,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for i,name in enumerate(names):
        p=ROOT/'results'/f'{name}_training_history.json'
        if p.exists():
            hist=json.loads(p.read_text());ax.plot([r['step'] for r in hist],[r['loss'] for r in hist],label=NAMES[name],color=COLORS[i])
    ax.set(xlabel='Actualización del optimizador',ylabel='Pérdida de entrenamiento',title='Aprendizaje de los adaptadores');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(figs/'entrenamiento.png',dpi=200,bbox_inches='tight');plt.close(fig)
    best=max(['base','lora','dora'],key=lambda n:metrics[n]['validation']['macro_f1'])
    rows=all_rows[best];matrix=np.zeros((4,5),int)
    for r in rows:
        p=r['prediction'].get('prioridad','invalida') if r['schema_ok'] else 'invalida'
        matrix[LABELS.index(r['truth']['prioridad']),(LABELS+['invalida']).index(p)]+=1
    fig,ax=plt.subplots(figsize=(7,4.5));im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=16)
    ax.set(xticks=range(5),xticklabels=['Crítica','Alta','Media','Baja','Inválida'],yticks=range(4),
           yticklabels=['Crítica','Alta','Media','Baja'],xlabel='Predicción',ylabel='Referencia',
           title=f'Matriz de confusión: {NAMES[best]}')
    for i in range(4):
        for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>8 else '#233047')
    fig.colorbar(im,ax=ax,label='Casos');fig.tight_layout();fig.savefig(figs/'confusion.png',dpi=200,bbox_inches='tight');plt.close(fig)
    source_test={r['id']:r for r in [json.loads(s) for s in (ROOT/'data'/'test.jsonl').read_text(encoding='utf8').splitlines()]}
    errors={}
    for name in names:
        bad=[r for r in all_rows[name] if not r['schema_ok'] or r['prediction']!=r['truth']]
        bad.sort(key=lambda r:(not(r['truth']['prioridad']=='critica' and r['prediction'].get('prioridad')!='critica'),
                               r['prediction'].get('prioridad')==r['truth']['prioridad']))
        errors[name]=[dict(r,facts=source_test[r['id']]['facts']) for r in bad[:3]]
    (ROOT/'results'/'error_examples.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf8')
    slices=[]
    for n in names:
        for split in ['test','stress']:
            rows=predictions(n,split)
            for priority in LABELS:
                part=[r for r in rows if r['truth']['prioridad']==priority]
                correct=sum(r['schema_ok'] and r['prediction']['prioridad']==priority for r in part)
                slices.append(dict(method=n,split=split,priority=priority,n=len(part),recall=correct/len(part)))
    pd.DataFrame(slices).to_csv(ROOT/'results'/'slices.csv',index=False)
    analysis=dict(best_validation=best,test_bootstrap_ci=ci,paired_f1_difference_ci=delta,
                  bootstrap_repetitions=1000,bootstrap_seed=20261023,summary=summary,
                  max_delta=metrics['lora']['test']['macro_f1']-metrics['base']['test']['macro_f1'])
    (ROOT/'results'/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf8')
    print(df.to_string(index=False))

if __name__=='__main__':main()
