"""Reconstruye los 128 casos nuevos sin tocar las particiones originales."""
from pathlib import Path
import hashlib,json
import dataset_generator as d

def main():
    root=Path(__file__).resolve().parents[1];out=root/'factorial'/'data';out.mkdir(parents=True,exist_ok=True)
    d.SCENARIOS['confirmation']=[
        'revisar el control de acceso al estacionamiento','recuperar el archivo de convenios internacionales',
        'habilitar las citas de la clínica universitaria','reparar el portal de compras institucionales',
        'restablecer las reservas de salas de estudio','actualizar la aplicación de movilidad académica',
        'restaurar el sistema de gestión de residuos','revisar el registro de proveedores']
    rows=d.build_split('confirmation',128,20261031)
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    original=[json.loads(s) for p in (root/'data').glob('*.jsonl') for s in p.read_text(encoding='utf8').splitlines()]
    assert not {r['group'] for r in rows}&{r['group'] for r in original}
    assert not {r['input'] for r in rows}&{r['input'] for r in original}
    target=out/'confirmation.jsonl'
    if target.exists():assert target.read_text(encoding='utf8')==text,'No se sobrescriben datos diferentes de una ejecución existente'
    else:target.write_text(text,encoding='utf8')
    sha=hashlib.sha256(text.encode()).hexdigest()
    if (out/'audit.json').exists():assert json.loads((out/'audit.json').read_text())['sha256']==sha
    print('Confirmación reproducida:128 casos; semilla20261031; hash',sha)

if __name__=='__main__':main()
