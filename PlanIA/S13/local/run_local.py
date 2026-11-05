"""Execute with the active Python environment and retain a readable execution log.

From the repository root:
    .venv/Scripts/python.exe PlanIA/S13/local/run_local.py
"""

import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    path = Path(__file__).with_name("hack11_local_v1.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    with path.with_name("execution.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            print("Python:", sys.executable, flush=True)
            print("Notebook:", path, flush=True)
            print("Started:", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [
                sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
            ]

            def on_start(cell, cell_index, **kwargs):
                print(f"CELL {cell_index} START {time.strftime('%H:%M:%S')}", flush=True)

            def on_executed(cell, cell_index, **kwargs):
                print(f"CELL {cell_index} END {time.strftime('%H:%M:%S')}", flush=True)
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        print(output.text, flush=True)
                    elif output.output_type == "error":
                        print("\n".join(output.traceback), flush=True)
                    elif "text/plain" in output.get("data", {}):
                        print(output.data["text/plain"], flush=True)

            try:
                NotebookClient(
                    notebook,
                    km=manager,
                    timeout=600,
                    resources={"metadata": {"path": str(path.parent)}},
                    on_cell_start=on_start,
                    on_cell_executed=on_executed,
                ).execute()
                print("EXECUTION COMPLETE", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
            except Exception:
                traceback.print_exc()
                raise
            finally:
                nbformat.write(notebook, path)
                print("Saved notebook:", path, flush=True)
            body, _ = HTMLExporter().from_notebook_node(notebook)
            html_path = path.with_suffix(".html")
            html_path.write_text(body, encoding="utf-8")
            print("Exported HTML:", html_path, flush=True)
    print("Execution and HTML export succeeded:", path)


if __name__ == "__main__":
    main()
