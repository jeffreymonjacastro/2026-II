"""Uso del adaptador entrenado con una tarea nueva, dentro de Slurm."""
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from experiment import load_model, CFG, ROOT, prepare_model_for_kbit_training
from dataset_generator import SYSTEM_INSTRUCTION
from evaluation import prompt
from validator import validate_json_output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['best','base','lora','dora','rslora','loraplus','lora_replica'],default='best')
    parser.add_argument('--seed',type=int,choices=[42,43],default=42)
    parser.add_argument('--task',required=True,help='Descripción con plazo, impacto y demás datos relevantes')
    args=parser.parse_args()
    assert torch.cuda.is_available(),'Ejecute este comando en una asignación GPU de Slurm'
    model,tokenizer=load_model()
    method=args.method
    final=ROOT/'factorial/results'
    if method=='best':
        method=json.loads((final/'technology_selection.json').read_text())['best_validation_method'] if (final/'COMPLETE_PEFT.json').exists() else 'lora'
    if method!='base':
        if method!='lora_replica' and (final/'COMPLETE_PEFT.json').exists():
            selected=json.loads((final/'selection.json').read_text())['selected']
            adapter=f'fx_{method}_r{selected["rank"]}_a{selected["alpha"]}_s{args.seed}'
        else:adapter=method
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        model.get_input_embeddings().to(torch.bfloat16);model.get_output_embeddings().to(torch.bfloat16)
        model=PeftModel.from_pretrained(model,ROOT/'adapters'/adapter)
        print('Adaptador:',adapter)
    model.eval()
    row=dict(instruction=SYSTEM_INSTRUCTION,input=args.task)
    inputs=tokenizer(prompt(row,tokenizer),return_tensors='pt').to('cuda')
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=96,do_sample=False,pad_token_id=tokenizer.pad_token_id)
    raw=tokenizer.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
    print(raw)
    checked=validate_json_output(raw)
    print('Esquema válido:',checked.valid)
    if not checked.valid:print('La respuesta requiere revisión:',checked.error)

if __name__=='__main__':main()
