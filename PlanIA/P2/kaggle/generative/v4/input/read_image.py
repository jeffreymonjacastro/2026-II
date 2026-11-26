import base64
import json
import os
from pathlib import Path

carpeta = Path(__file__).resolve().parent
respuesta = json.loads((carpeta / "response.json").read_text(encoding="utf-8-sig"))

imagen = carpeta / "generada.png"
imagen.write_bytes(base64.b64decode(respuesta["image_base64"]))
os.startfile(str(imagen))
