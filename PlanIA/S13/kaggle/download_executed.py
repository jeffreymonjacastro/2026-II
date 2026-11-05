"""Download a genuinely executed Kaggle notebook, then export it without running cells.

Run with the Python environment that contains the Kaggle SDK. If nbconvert lives
in another environment, pass its interpreter with --html-python.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="owner/kernel-slug")
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path, help="Executed .ipynb target")
    parser.add_argument("--html-python", default=sys.executable, help="Interpreter with nbconvert")
    args = parser.parse_args()
    if args.version < 1 or len(args.kernel.split("/")) != 2:
        parser.error("Use a positive version and kernel in owner/slug format.")

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

    api = KaggleApi()
    api.authenticate()
    owner, slug = args.kernel.split("/")
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug = owner, slug
    request.file_path = "__notebook__.ipynb"
    request.version_number = args.version
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
    response.raise_for_status()
    notebook = response.json()
    cells = notebook["cells"]
    outputs = sum(len(cell.get("outputs", [])) for cell in cells)
    errors = sum(output.get("output_type") == "error"
                 for cell in cells for output in cell.get("outputs", []))
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    executed = sum(cell.get("execution_count") is not None for cell in code_cells)
    if not outputs or errors or executed != len(code_cells):
        raise RuntimeError(f"Expected complete executed notebook: {outputs=} {errors=} {executed=}/{len(code_cells)}")

    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    html = target.with_name(target.stem.replace("_ejecutado", "") + ".html")
    export_code = (
        "import sys,nbformat;from nbconvert import HTMLExporter;from pathlib import Path;"
        "n=nbformat.read(sys.argv[1],as_version=4);nbformat.validate(n);"
        "h,_=HTMLExporter().from_notebook_node(n);Path(sys.argv[2]).write_text(h,encoding='utf-8')"
    )
    subprocess.run([args.html_python, "-c", export_code, str(target), str(html)], check=True)
    evidence = {
        "kernel": args.kernel, "version": args.version,
        "endpoint": "KernelsApiService.DownloadKernelOutput",
        "remote_file": "__notebook__.ipynb", "http_status": response.status_code,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": target.name, "sha256": hashlib.sha256(response.content).hexdigest(),
        "bytes": len(response.content), "cells": len(cells), "code_cells": len(code_cells),
        "executed_code_cells": executed, "outputs": outputs, "errors": errors,
        "html": html.name, "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
        "local_model_execution": False,
    }
    target.with_name("notebook_download.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
