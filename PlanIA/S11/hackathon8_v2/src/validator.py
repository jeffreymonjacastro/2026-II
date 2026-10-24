"""Validador estricto adaptado del archivo proporcionado por el docente."""
import json
from dataclasses import dataclass

@dataclass
class ValidationResult:
    valid: bool
    error: str = ''
    parsed_json: dict = None
    def __bool__(self): return self.valid

def validate_json_output(output):
    try:
        # Reject duplicate keys as well as nonstandard NaN constants.
        def pairs(items):
            d={}
            for k,v in items:
                if k in d: raise ValueError('Clave repetida')
                d[k]=v
            return d
        d=json.loads(output,object_pairs_hook=pairs,parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError,TypeError): return ValidationResult(False,'JSON no válido')
    if not isinstance(d,dict) or set(d)!={'prioridad','escalar','bloqueada','criterio'}:
        return ValidationResult(False,'Claves incorrectas')
    if not isinstance(d['prioridad'],str) or d['prioridad'] not in ['critica','alta','media','baja']:
        return ValidationResult(False,'Prioridad incorrecta')
    if type(d['escalar']) is not bool or type(d['bloqueada']) is not bool:
        return ValidationResult(False,'Tipos booleanos incorrectos')
    if not isinstance(d['criterio'],str) or d['criterio'] not in ['riesgo','impacto','plazo','dependencia','planificacion']:
        return ValidationResult(False,'Criterio incorrecto')
    return ValidationResult(True,parsed_json=d)
