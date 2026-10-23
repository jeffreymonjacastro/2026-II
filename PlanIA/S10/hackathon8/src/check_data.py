"""Verificación corta de etiquetas, límites de decisión y particiones."""
import json
from pathlib import Path
from dataset_generator import reference, CLASSES
from validator import validate_json_output

def main():
    root=Path(__file__).resolve().parents[1]
    groups=[];texts=[];summary={}
    for split in ['train','validation','test','stress']:
        rows=[json.loads(s) for s in (root/'data'/f'{split}.jsonl').read_text(encoding='utf8').splitlines()]
        groups.append({r['group'] for r in rows});texts.append({r['input'] for r in rows})
        for r in rows:
            assert validate_json_output(r['output']).valid
            assert json.loads(r['output'])==reference(r['facts'])
        summary[split]={p:sum(json.loads(r['output'])['prioridad']==p for r in rows) for p in CLASSES}
    for i in range(len(groups)):
        for j in range(i):assert not groups[i]&groups[j] and not texts[i]&texts[j]
    x=dict(horas=120,impacto=1,personas=1,dependencias=0,riesgo='ninguno',alternativa=True,interrupcion=False,bloqueada=False)
    assert reference(x)['prioridad']=='baja'
    assert reference(dict(x,horas=48))['prioridad']=='media'
    assert reference(dict(x,horas=8,impacto=2))['prioridad']=='alta'
    assert reference(dict(x,horas=9,impacto=2))['prioridad']=='media'
    assert reference(dict(x,interrupcion=True,personas=49,alternativa=False))['prioridad']=='baja'
    assert reference(dict(x,interrupcion=True,personas=50,alternativa=False))['prioridad']=='critica'
    assert reference(dict(x,riesgo='datos',bloqueada=True))['prioridad']=='critica'
    assert reference(dict(x,impacto=3))['prioridad']=='media'
    for hours in [4,24,120]:
        for alternative in [True,False]:
            states=[reference(dict(x,horas=hours,alternativa=alternative,impacto=i))['prioridad'] for i in [1,2,3]]
            ranks=[CLASSES.index(p) for p in states]
            assert ranks==sorted(ranks,reverse=True), 'El impacto no puede reducir la prioridad'
    assert not validate_json_output('{"prioridad": []}').valid
    assert not validate_json_output('Texto extra '+json.dumps(reference(x))).valid
    result=dict(status='ok',class_counts=summary,group_overlap=0,exact_duplicates=0,boundary_checks=8,impact_monotonicity_checks=6)
    (root/'data'/'checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
