"""Uso del adaptador entrenado con una tarea nueva, dentro de Slurm."""
import argparse
from pathlib import Path
import torch
from peft import PeftModel
from experiment import load_model, CFG, ROOT
from dataset_generator import SYSTEM_INSTRUCTION
from evaluation import prompt
from validator import validate_json_output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['base','lora','dora','lora_replica'],default='lora')
    parser.add_argument('--task',required=True,help='Descripción con plazo, impacto y demás datos relevantes')
    args=parser.parse_args()
    assert torch.cuda.is_available(),'Ejecute este comando en una asignación GPU de Slurm'
    model,tokenizer=load_model()
    if args.method!='base':model=PeftModel.from_pretrained(model,ROOT/'adapters'/args.method)
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
