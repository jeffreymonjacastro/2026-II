"""Análisis factorial reproducible a partir de predicciones reales, sin GPU."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];F=ROOT/'factorial';R=F/'results';G=F/'figures'
LABELS=['critica','alta','media','baja'];COLORS={8:'#167C80',32:'#CE7837'}
METHODS=['base','lora','dora','rslora','loraplus']
NAMES={'base':'Base','lora':'LoRA','dora':'DoRA','rslora':'rsLoRA','loraplus':'LoRA+'}
def read(name):return json.loads((R/name).read_text(encoding='utf8'))
def save(name,value):(R/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def rows(name,split):return [json.loads(s) for s in (R/f'{name}_{split}_predictions.jsonl').read_text(encoding='utf8').splitlines()]
def ids(r):return [x['id'] for x in r]
def vector(r):return np.array([LABELS.index(x['prediction']['prioridad']) if x['schema_ok'] else 4 for x in r])
def f1(y,p):
    c=np.bincount(y*5+p,minlength=20).reshape(4,5);tp=np.diag(c[:,:4]);den=c.sum(axis=1)+c[:,:4].sum(axis=0)
    return float(np.mean(np.divide(2*tp,den,out=np.zeros(4,dtype=float),where=den!=0)))
def bootstrap_delta(left,right,split='validation',n=2000):
    """Seed pairs and examples are resampled jointly across conditions."""
    a=[rows(k,split) for k in left];b=[rows(k,split) for k in right]
    assert all(ids(z)==ids(a[0]) for z in a+b)
    y=np.array([LABELS.index(x['truth']['prioridad']) for x in a[0]])
    pa=[vector(z) for z in a];pb=[vector(z) for z in b]
    groups=[np.flatnonzero(y==i) for i in range(4)]
    rng=np.random.default_rng(20261023);values=[]
    for _ in range(n):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        seeds=rng.choice(len(a),len(a),replace=True)
        values.append(float(np.mean([f1(y[idx],pb[s][idx])-f1(y[idx],pa[s][idx]) for s in seeds])))
    point=float(np.mean([f1(y,pb[s])-f1(y,pa[s]) for s in range(len(a))]))
    return {'delta':point,'ci95':np.quantile(values,[.025,.975]).tolist(),'upper_one_sided95':float(np.quantile(values,.95)),
            'n_cases':len(y),'n_seeds':len(a),'bootstrap':n,'seed':20261023,
            'caution':'Resampling only two observed training seeds cannot characterize the full seed population.'}
def design(df):
    x=(np.log2(df['rank'].to_numpy())-3)/2;z=np.where(df.alpha.to_numpy()==8,-1,1);q=x*x-2/3
    b=np.where(df.seed.to_numpy()==42,-1,1)
    return np.column_stack([np.ones(len(df)),x,q,z,x*z,q*z,b])
def regression(df):
    X=design(df);y=df.validation_f1.to_numpy();beta=np.linalg.lstsq(X,y,rcond=None)[0]
    residual=y-X@beta;dfe=len(y)-np.linalg.matrix_rank(X);sse=float(residual@residual)
    cov=sse/dfe*np.linalg.inv(X.T@X);se=np.sqrt(np.diag(cov));t=np.divide(beta,se,out=np.zeros_like(beta),where=se>0)
    terms=['intercepto','r_lineal','r_cuadratico','alpha','r_lineal:alpha','r_cuadratico:alpha','bloque_semilla']
    coefficients=[{'term':k,'coefficient':float(v),'standard_error':float(s),'p':float(2*stats.t.sf(abs(tt),dfe))} for k,v,s,tt in zip(terms,beta,se,t)]
    effects=[]
    for name,remove in [('r',[1,2]),('alpha',[3]),('r × alpha',[4,5]),('semilla',[6])]:
        xx=np.delete(X,remove,axis=1);rr=y-xx@np.linalg.lstsq(xx,y,rcond=None)[0];ss=float(rr@rr)
        ff=max(0,(ss-sse)/len(remove))/(sse/dfe)
        effects.append({'effect':name,'df':len(remove),'F':float(ff),'p':float(stats.f.sf(ff,len(remove),dfe))})
    return {'formula':'F1_validación = b0+b1*x+b2*q+b3*z+b4*x*z+b5*q*z+b6*bloque; x=(log2(r)-3)/2; q=x²-2/3; z=−1/+1',
            'coefficients':coefficients,'effects':effects,'df_residual':int(dfe),'residual_rmse':float(np.sqrt(sse/dfe)),
            'r2':float(1-sse/np.sum((y-y.mean())**2)),'beta':beta.tolist(),
            'scope':'Exploratory response surface and partial F tests. Two seeds, bounded F1 and tiny residual df limit inference. Alpha has two levels, so its curvature cannot be estimated.'}

def main(core_only=False):
    assert core_only or ((R/'COMPLETE.json').exists() and (R/'COMPLETE_PEFT.json').exists()),'La ejecución o la comparación adicional todavía no ha terminado'
    G.mkdir(exist_ok=True)
    metrics=read('all_metrics.json');plan=json.loads((F/'plan.json').read_text());records=[]
    for name,m in metrics.items():
        t=m['training'];record=dict(condition=name,method=m['method'],phase=m['phase'],rank=t['rank'],alpha=t['alpha'],seed=t['seed'],
           scale=t['alpha']/(np.sqrt(t['rank']) if m['method']=='rslora' else t['rank']),parameters=t['trainable_parameters'],validation_f1=m['validation']['macro_f1'],
           test_f1=m['test']['macro_f1'],stress_f1=m['stress']['macro_f1'],schema=m['test']['schema_rate'],
           exact=m['test']['exact_match'],critical_recall=m['test']['critical_recall'],train_seconds=t['train_seconds'],
           warmup_seconds=t['warmup_seconds'],train_vram_gib=t['train_peak_allocated_gb'],train_reserved_gib=t['train_peak_reserved_gb'],latency_s=m['test']['mean_latency_s'],
           inference_vram_gib=m['test']['inference_peak_allocated_gb'],cost_usd=t['simulated_train_cost_usd'],
           wall_seconds=m['wall_seconds'],pipeline_cost_usd=m['wall_seconds']/3600)
        test_rows=rows(name,'test')
        record['mean_generated_tokens']=float(np.mean([r['generated_tokens'] for r in test_rows]))
        record['generated_tokens_per_second']=float(sum(r['generated_tokens'] for r in test_rows)/sum(r['latency_s'] for r in test_rows))
        history=read(f'{name}_training_history.json')
        intervals=np.diff([0]+[h['elapsed_s'] for h in history])
        record['median_late_step_seconds']=float(np.median(intervals[-16:]))
        if m['method']=='rslora':assert t['use_rslora'] is True
        if m['method']=='loraplus':
            assert t['loraplus_lr_ratio']==16
            history=read(f'{name}_training_history.json')
            assert len(history)==32 and all(abs(h['group_lrs'][2]/h['group_lrs'][0]-16)<1e-9 for h in history)
        for split in ['validation','test','stress']:
            source={r['id']:r for r in [json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text(encoding='utf8').splitlines()]}
            rr=rows(name,split);assert len(rr)==len(source) and len(set(ids(rr)))==len(source)
            y=np.array([LABELS.index(r['truth']['prioridad']) for r in rr])
            assert abs(f1(y,vector(rr))-m[split]['macro_f1'])<1e-12,('Métrica inconsistente',name,split)
            for row in rr:
                assert row['input']==source[row['id']]['input'] and row['truth']==json.loads(source[row['id']]['output'])
        records.append(record)
    df=pd.DataFrame(records);df.to_csv(R/'experiments.csv',index=False)
    factorial=df[(df.method=='lora')&df['rank'].isin(plan['ranks'])&df.alpha.isin(plan['alphas'])].copy()
    assert len(factorial)==12
    assert (factorial.groupby(['rank','alpha']).seed.apply(lambda s:set(s))=={42,43}).all()
    assert set(read('hardware.json')['data_sha256'].values())=={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.jsonl')}
    reg=regression(factorial);save('regression.json',reg)
    grouped=df.groupby(['method','rank','alpha']).agg(n=('seed','size'),validation_f1=('validation_f1','mean'),validation_sd=('validation_f1','std'),
        test_f1=('test_f1','mean'),test_sd=('test_f1','std'),stress_f1=('stress_f1','mean'),schema=('schema','mean'),critical_recall=('critical_recall','mean'),
        train_seconds=('train_seconds','mean'),train_vram_gib=('train_vram_gib','mean'),latency_s=('latency_s','mean'),cost_usd=('cost_usd','mean')).reset_index()
    grouped=grouped.assign(_order=grouped.method.map({m:i for i,m in enumerate(METHODS)})).sort_values(['_order','rank','alpha']).drop(columns='_order').reset_index(drop=True)
    # The design frontier concerns LoRA ranks/alpha, not a pooled method search.
    grouped['pareto']=pd.Series(pd.NA,index=grouped.index,dtype='boolean')
    for i,row in grouped[grouped.method=='lora'].iterrows():
        others=grouped[(grouped.method=='lora')&(grouped.index!=i)]
        dominated=((others.validation_f1>=row.validation_f1)&(others.train_vram_gib<=row.train_vram_gib)&(others.train_seconds<=row.train_seconds)&
                   ((others.validation_f1>row.validation_f1)|(others.train_vram_gib<row.train_vram_gib)|(others.train_seconds<row.train_seconds))).any()
        grouped.loc[i,'pareto']=not dominated
    grouped.to_csv(R/'cells.csv',index=False)
    save('cell_summary.json',json.loads(grouped.to_json(orient='records')))
    if (R/'rslora_high_rank_control.json').exists():
        control=read('rslora_high_rank_control.json')
        diag=grouped[grouped.method.isin(['lora','rslora'])&grouped['rank'].isin([1,32])&(grouped.alpha==control['alpha'])]
        assert len(diag)==4
        diag.to_csv(R/'rslora_control_summary.csv',index=False)
    comparisons=[]
    for alpha in plan['alphas']:
        levels=sorted(df[(df.method=='lora')&(df.alpha==alpha)]['rank'].unique().tolist())
        for lo,hi in zip(levels,levels[1:]):
            a=[f'fx_lora_r{lo}_a{alpha}_s{s}' for s in plan['seeds']];b=[f'fx_lora_r{hi}_a{alpha}_s{s}' for s in plan['seeds']]
            result=bootstrap_delta(a,b);result.update(alpha=alpha,rank_from=lo,rank_to=hi,split='validation')
            result['rules_out_gain_over_delta']=result['upper_one_sided95']<plan['practical_delta'];comparisons.append(result)
    initial=read('theoretical_optimum.json')
    save('marginal_gains.json',{'practical_delta':plan['practical_delta'],'comparisons':comparisons,
         'interpretation':'A nonsignificant p value alone is not evidence of saturation. A one-sided upper bootstrap bound below 0.02 limits meaningful gain conditional on the observed cases and seeds.'})
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7))
    for ax,col,title in zip(axes,['validation_f1','test_f1'],['Validación (selección)','Prueba (evaluación)']):
        pivot=factorial.groupby(['rank','alpha'])[col].mean().unstack().reindex([2,8,32])
        im=ax.imshow(pivot.values,cmap='YlGnBu',vmin=.65,vmax=.95,aspect='auto')
        ax.set(xticks=[0,1],xticklabels=['alpha = 8','alpha = 32'],yticks=[0,1,2],yticklabels=['r = 2','r = 8','r = 32'],title=title)
        for i,r in enumerate([2,8,32]):
            for j,a in enumerate([8,32]):
                v=pivot.loc[r,a];vals=factorial[(factorial['rank']==r)&(factorial.alpha==a)].sort_values('seed')[col].to_numpy()
                ax.text(j,i,f'{v:.3f}\n{vals[0]:.3f} / {vals[1]:.3f}',ha='center',va='center',color='white' if v>.84 else '#173447',fontsize=12)
    fig.suptitle('Factorial 3 × 2: F1 medio y réplicas (42 / 43)',fontweight='bold')
    fig.subplots_adjust(top=.82,bottom=.13,wspace=.28,left=.08,right=.87)
    cax=fig.add_axes([.91,.20,.018,.55]);fig.colorbar(im,cax=cax,label='F1 macro')
    fig.savefig(G/'factorial_heatmap.png',dpi=200,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));beta=np.array(reg['beta'])
    grid=np.geomspace(1,64,160)
    for a,color in COLORS.items():
        z=-1 if a==8 else 1;x=(np.log2(grid)-3)/2;q=x*x-2/3
        predicted=beta[0]+beta[1]*x+beta[2]*q+beta[3]*z+beta[4]*x*z+beta[5]*q*z
        within=(grid>=2)&(grid<=32)
        ax.plot(grid[within],predicted[within],color=color,label=f'alpha = {a}, regresión')
        for mask in [grid<2,grid>32]:ax.plot(grid[mask],predicted[mask],color=color,ls='--',alpha=.6)
        part=grouped[(grouped.method=='lora')&(grouped.alpha==a)].sort_values('rank')
        ax.scatter(part['rank'],part.validation_f1,color=color,s=90,zorder=3)
        for _,row in part.iterrows():ax.text(row['rank'],row.validation_f1+.012,f'{row.validation_f1:.3f}',ha='center',fontsize=9,color=color)
    ax.axvspan(1,2,color='#DAE1E6',alpha=.25);ax.axvspan(32,64,color='#DAE1E6',alpha=.25)
    ax.set(xscale='log',xticks=[1,2,8,32,64],xticklabels=['1','2','8','32','64'],ylabel='F1 macro medio en validación',xlabel='Rango r (escala logarítmica)',
           title='Interacción de rango y alpha, con comprobación adicional',ylim=(.65,.90));ax.legend(frameon=False,ncol=2,loc='lower left');ax.grid(alpha=.15)
    ax.text(0,-.22,'Puntos: mediciones duplicadas. Línea continua: ajuste. Línea discontinua: extrapolación.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(G/'factorial_interaction.png',dpi=200,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    offsets={(1,32):(-6,13),(2,8):(15,-18),(2,32):(-28,-19),(8,8):(-24,-19),(8,32):(-28,17),(32,8):(15,12),(32,32):(18,-9),(64,32):(5,10)}
    for ax,col,title in zip(axes,['train_seconds','train_vram_gib'],['Tiempo de entrenamiento (s)','VRAM máxima de entrenamiento (GiB)']):
        for a,color in COLORS.items():
            part=grouped[(grouped.method=='lora')&(grouped.alpha==a)]
            for j,(_,row) in enumerate(part.iterrows()):
                ax.scatter(row[col],row.validation_f1,color=color,s=75,marker='D' if row.pareto else 'o',label=f'alpha = {a}' if j==0 else None)
                offset=offsets.get((int(row['rank']),a),(6,9)) if col=='train_seconds' else (6,9)
                ax.annotate(f'r={int(row["rank"])}',(row[col],row.validation_f1),xytext=offset,textcoords='offset points',fontsize=9,
                            arrowprops={'arrowstyle':'-','color':'#65758B','lw':.6})
        ax.set(xlabel=title,ylabel='F1 macro medio en validación',ylim=(.69,.88));ax.grid(alpha=.15);ax.margins(x=.10)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([],[],color=c,marker='o',ls='',label=f'alpha = {a}') for a,c in COLORS.items()],frameon=False,loc='lower right',fontsize=9)
    fig.suptitle('Desempeño y recursos: medias de dos semillas',fontweight='bold')
    fig.text(.5,.01,'Diamantes: configuraciones no dominadas en F1, memoria y tiempo. Eje vertical ampliado.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(G/'factorial_resources.png',dpi=200,bbox_inches='tight');plt.close(fig)
    # A wide single panel stays readable when embedded in the concise report.
    fig,ax=plt.subplots(figsize=(9,3.2))
    for a,color in COLORS.items():
        part=grouped[(grouped.method=='lora')&(grouped.alpha==a)]
        for _,row in part.iterrows():
            ax.scatter(row.train_vram_gib,row.validation_f1,color=color,s=65,marker='D' if row.pareto else 'o')
            ax.annotate(f'r={int(row["rank"])}',(row.train_vram_gib,row.validation_f1),xytext=(5,8),textcoords='offset points',fontsize=10)
    ax.set(xlabel='VRAM máxima de entrenamiento asignada por PyTorch (GiB)',ylabel='F1 medio de validación',
           title='F1 de validación frente a memoria',ylim=(.69,.88));ax.grid(alpha=.15);ax.margins(x=.08)
    ax.legend(handles=[Line2D([],[],color=c,marker='o',ls='',label=f'alpha = {a}') for a,c in COLORS.items()],frameon=False,loc='upper right',fontsize=9)
    ax.text(0,-.35,'Dos semillas por punto. Diamantes: no dominadas en F1, tiempo y memoria. Eje F1 ampliado.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(G/'factorial_frontier.png',dpi=200,bbox_inches='tight');plt.close(fig)
    if core_only:
        save('core_analysis_complete.json',{'status':'core_only','training_conditions_available':len(df),'factorial_runs':12,
             'note':'Actual factorial and outside-domain runs only. Technology comparison and final confirmation are still pending.'})
        print(grouped.to_string(index=False));return
    confirmation=read('confirmation_metrics.json');selection=read('selection.json')['selected'];confirmed=[]
    for name,m in confirmation.items():
        confirmed.append({'condition':name,'method':'base' if name=='base' else name.split('_')[1],
          'seed':None if name=='base' else int(name.rsplit('s',1)[1]),'f1':m['macro_f1'],'schema':m['schema_rate'],
          'critical_recall':m['critical_recall'],'exact':m['exact_match'],'latency_s':m['mean_latency_s']})
    cdf=pd.DataFrame(confirmed);cdf.to_csv(R/'confirmation_summary.csv',index=False)
    source_confirmation={r['id']:r for r in [json.loads(s) for s in (F/'data/confirmation.jsonl').read_text(encoding='utf8').splitlines()]}
    errors={}
    for name in confirmation:
        rr=rows(name,'confirmation');assert len(rr)==128
        assert set(ids(rr))==set(source_confirmation)
        yy=np.array([LABELS.index(row['truth']['prioridad']) for row in rr])
        assert abs(f1(yy,vector(rr))-confirmation[name]['macro_f1'])<1e-12,('Métrica de confirmación inconsistente',name)
        for row in rr:
            source=source_confirmation[row['id']]
            assert row['input']==source['input'] and row['truth']==json.loads(source['output'])
        bad=[r for r in rr if not r['schema_ok'] or r['prediction']!=r['truth']]
        bad.sort(key=lambda r:(not(r['truth']['prioridad']=='critica' and r['prediction'].get('prioridad')!='critica'),
                               r['prediction'].get('prioridad')==r['truth']['prioridad']))
        errors[name]=[dict(r,facts=source_confirmation[r['id']]['facts']) for r in bad[:2]]
    save('confirmation_errors.json',errors)
    best_method=read('technology_selection.json')['best_validation_method']
    best_name=f'fx_{best_method}_r{selection["rank"]}_a{selection["alpha"]}_s42'
    rr=rows(best_name,'confirmation');matrix=np.zeros((4,5),int)
    for row in rr:
        pred=row['prediction'].get('prioridad','invalida') if row['schema_ok'] else 'invalida'
        matrix[LABELS.index(row['truth']['prioridad']),(LABELS+['invalida']).index(pred)]+=1
    fig,ax=plt.subplots(figsize=(7.2,4.7));im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=32)
    ax.set(xticks=range(5),xticklabels=['Crítica','Alta','Media','Baja','Inválida'],yticks=range(4),
           yticklabels=['Crítica','Alta','Media','Baja'],xlabel='Predicción',ylabel='Referencia',
           title=f'Confirmación: {NAMES[best_method]}, semilla 42')
    for i in range(4):
        for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>16 else '#173447')
    fig.colorbar(im,ax=ax,label='Casos');fig.tight_layout();fig.savefig(G/'confirmation_confusion.png',dpi=200,bbox_inches='tight');plt.close(fig)
    tech=bootstrap_delta([f'fx_lora_r{selection["rank"]}_a{selection["alpha"]}_s{s}' for s in [42,43]],
                         [f'fx_dora_r{selection["rank"]}_a{selection["alpha"]}_s{s}' for s in [42,43]],'confirmation')
    save('technology_comparison.json',tech)
    comparisons_tech={}
    for method in ['dora','rslora','loraplus']:
        comparisons_tech[method]=bootstrap_delta([f'fx_lora_r{selection["rank"]}_a{selection["alpha"]}_s{s}' for s in [42,43]],
             [f'fx_{method}_r{selection["rank"]}_a{selection["alpha"]}_s{s}' for s in [42,43]],'confirmation')
    save('technology_comparisons.json',{'comparisons':comparisons_tech,
       'scope':'Exploratory individual 95% intervals, not adjusted for three method comparisons. Hyperparameters shared from LoRA selection, not tuned separately for each method.'})
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    for i,method in enumerate(METHODS):
        part=cdf[cdf.method==method];mean=part.f1.mean();color=['#65758B','#167C80','#CE7837','#4054A0','#8A4B83'][i]
        ax.bar(i,mean,color=color,alpha=.8,width=.5)
        ax.scatter([i]*len(part),part.f1,color='#173447',s=35,zorder=3)
        ax.text(i,part.f1.max()+.025,f'{mean:.3f}',ha='center',fontweight='bold')
    ax.set(xticks=range(5),xticklabels=[NAMES[m] for m in METHODS],ylim=(0,1.08),ylabel='F1 macro',title='Comprobación final con 128 casos nuevos')
    ax.text(0,-.2,'Barra: media. Puntos: semillas 42 y 43. El modelo base no se entrena.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(G/'confirmation.png',dpi=200,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    for method,color in zip(['lora','dora','rslora','loraplus'],['#167C80','#CE7837','#4054A0','#8A4B83']):
        histories=[read(f'fx_{method}_r{selection["rank"]}_a{selection["alpha"]}_s{s}_training_history.json') for s in [42,43]]
        losses=np.array([[row['loss'] for row in h] for h in histories]);steps=np.arange(1,33)
        ax.plot(steps,losses.mean(axis=0),color=color,label=NAMES[method],ls='--' if method=='rslora' else '-')
        ax.fill_between(steps,losses.min(axis=0),losses.max(axis=0),color=color,alpha=.10)
    ax.set(xlabel='Actualización del optimizador',ylabel='Pérdida media sobre la respuesta JSON',title='Aprendizaje con el mismo presupuesto de 32 pasos')
    ax.legend(frameon=False,ncol=4);ax.grid(alpha=.15)
    ax.text(0,-.22,'Línea: media. Banda: mínimo y máximo de dos semillas. No es un intervalo de confianza.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(G/'peft_training.png',dpi=200,bbox_inches='tight');plt.close(fig)
    save('analysis_complete.json',{'factorial_runs':12,'total_runs':len(df),'selection':selection,'outside_rank':64,
         'fresh_confirmation_cases':128,'regression_r2':reg['r2'],'protected_data_audit':'passed'})
    selected_rows=grouped[(grouped['rank']==selection['rank'])&(grouped.alpha==selection['alpha'])]
    ll=selected_rows[selected_rows.method=='lora'].iloc[0];dd=selected_rows[selected_rows.method=='dora'].iloc[0]
    cb=cdf[cdf.method=='base'].f1.mean();cl=cdf[cdf.method=='lora'].f1.mean();cd=cdf[cdf.method=='dora'].f1.mean()
    confirmation_means={m:float(cdf[cdf.method==m].f1.mean()) for m in METHODS}
    candidates=[x for x in comparisons if x['alpha']==selection['alpha'] and x['rank_from']>=selection['rank']]
    if not candidates:candidates=[x for x in comparisons if x['alpha']==selection['alpha']]
    supported=[x for x in candidates if x['rules_out_gain_over_delta']]
    frontier_gain=min(supported or candidates,key=lambda x:x['rank_from'])
    if frontier_gain['rules_out_gain_over_delta']:
        frontier=f"Al pasar de r={frontier_gain['rank_from']} a r={frontier_gain['rank_to']} con alpha={frontier_gain['alpha']}, la diferencia media en validación fue {frontier_gain['delta']:+.3f}. Su límite superior unilateral del 95 % fue {frontier_gain['upper_one_sided95']:.3f}, menor que el margen práctico de 0.02. En este tramo, con los casos y semillas medidos, los resultados no justifican aumentar el rango para buscar una ganancia de 0.02 o más."
    else:
        frontier=f"Al pasar de r={frontier_gain['rank_from']} a r={frontier_gain['rank_to']} con alpha={frontier_gain['alpha']}, la diferencia media en validación fue {frontier_gain['delta']:+.3f}. Su límite superior unilateral del 95 % fue {frontier_gain['upper_one_sided95']:.3f}. La incertidumbre todavía permite una mejora de 0.02 o más, por lo que no se demuestra una saturación estadística en ese tramo."
    cp=initial['candidate'];measured=grouped[(grouped.method=='lora')&(grouped['rank']==cp['rank'])&(grouped.alpha==cp['alpha'])].iloc[0]
    candidate_check={'rank':cp['rank'],'alpha':cp['alpha'],'predicted_f1':cp['predicted_validation_f1'],
                     'observed_f1':float(measured.validation_f1),'error':float(measured.validation_f1-cp['predicted_validation_f1'])}
    save('optimum_check.json',candidate_check)
    conclusion=(f"**Configuración recomendada:** LoRA con r={selection['rank']} y alpha={selection['alpha']}, elegido por validación según la tolerancia práctica fijada antes del experimento. Su F1 medio de validación es {ll.validation_f1:.3f} y el de prueba {ll.test_f1:.3f}.\n\n"
        +frontier+"\n\n"
        +f"El máximo previsto por la superficie fue r={cp['rank']}, alpha={cp['alpha']}, con F1 {cp['predicted_validation_f1']:.3f}. La comprobación real con dos semillas obtuvo {measured.validation_f1:.3f}. El modelo de regresión orienta la búsqueda, pero no reemplaza las mediciones.\n\n"
        +f"En los 128 casos nuevos, el F1 fue {cb:.3f} para base, {cl:.3f} para LoRA y {cd:.3f} para DoRA, promediando las dos réplicas de los adaptadores. DoRA necesitó {dd.train_seconds/ll.train_seconds:.2f} veces el tiempo de entrenamiento y {dd.train_vram_gib/ll.train_vram_gib:.2f} veces la VRAM de LoRA. La diferencia DoRA menos LoRA en confirmación fue {tech['delta']:+.3f}, con intervalo del 95 % [{tech['ci95'][0]:+.3f}, {tech['ci95'][1]:+.3f}].\n\n"
        +"El óptimo y la frontera corresponden a la política sintética, 256 ejemplos de entrenamiento y 32 actualizaciones. No son límites universales de LoRA. Dos semillas y una GPU compartida restringen la precisión de las conclusiones.")
    chosen=read('technology_selection.json')
    highest_confirmation=max(confirmation_means,key=confirmation_means.get)
    conclusion+=(f"\n\n**Comparación ampliada:** rsLoRA obtiene F1 {confirmation_means['rslora']:.3f} y LoRA+ {confirmation_means['loraplus']:.3f} en los 128 casos nuevos. Entre los métodos con rango y alpha compartidos, la mayor media de validación corresponde a {NAMES[chosen['best_validation_method']]}. Las tasas de LoRA+ se mantienen en razón B/A=16 durante los 32 pasos. Estos métodos comparten la configuración seleccionada para LoRA; no se buscaron sus óptimos por separado. Los intervalos de las tres comparaciones son exploratorios y no están ajustados por multiplicidad.")
    if selection['rank']==1:
        conclusion+='\n\nCon r=1, las escalas alpha/r y alpha/sqrt(r) coinciden. Por eso las diferencias observadas entre LoRA y rsLoRA en ese rango no se atribuyen al reescalamiento. Se fijaron las semillas, pero no se forzó el uso exclusivo de kernels deterministas de CUDA. Se agregó una comparación duplicada de rsLoRA con r=32 frente al LoRA correspondiente del factorial, donde las escalas sí difieren. Sus resultados se conservan en rslora_control_summary.csv y no cambian la configuración compartida ni la selección con confirmación.'
    if highest_confirmation!=chosen['best_validation_method']:
        conclusion+=f"\n\nLa mayor media descriptiva en la confirmación corresponde a {NAMES[highest_confirmation]} ({confirmation_means[highest_confirmation]:.3f}), mientras que validación había elegido {NAMES[chosen['best_validation_method']]}. No se cambia la selección usando este conjunto nuevo. El cambio de orden muestra la incertidumbre de comparar métodos con particiones pequeñas y dos semillas."
    (F/'conclusion.md').write_text(conclusion,encoding='utf8')
    save('narrative.json',{'selection':selection,'frontier':frontier,'candidate_check':candidate_check,
         'frontier_edge':frontier_gain,
         'confirmation_base_f1':float(cb),'confirmation_lora_f1':float(cl),'confirmation_dora_f1':float(cd),
         'confirmation_means':confirmation_means,'best_validation_method':chosen['best_validation_method'],
         'train_ratio_dora_lora':float(dd.train_seconds/ll.train_seconds),'vram_ratio_dora_lora':float(dd.train_vram_gib/ll.train_vram_gib)})
    print(grouped.to_string(index=False));print(cdf.to_string(index=False))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--core-only',action='store_true',help='Analizar únicamente el factorial ya terminado mientras siguen las comparaciones')
    main(core_only=parser.parse_args().core_only)
