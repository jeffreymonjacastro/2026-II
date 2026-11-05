"""Resume a partial v2 download, verifying every file against the remote manifest."""
from pathlib import Path
import hashlib
import json
import time
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

out = Path(__file__).resolve().parent / "outputs"
entries = json.loads((out / "manifest.json").read_text())
api = KaggleApi()
api.authenticate()
downloaded = []
for entry in entries:
    target = (out / entry["name"]).resolve()
    if not target.is_relative_to(out.resolve()):
        raise ValueError("Unexpected manifest path")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]:
        continue
    for attempt in range(4):
        try:
            req = ApiDownloadKernelOutputRequest()
            req.owner_slug = "jeffreyamc"
            req.kernel_slug = "hack11-human-ai-credit-v2-good-credit"
            req.version_number = 1
            req.file_path = entry["name"]
            with api.build_kaggle_client() as client:
                response = client.kernels.kernels_api_client.download_kernel_output(req)
            response.raise_for_status()
            content = response.content
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
            target.write_bytes(content)
            downloaded.append(entry["name"])
            print("Verified:", entry["name"], flush=True)
            break
        except Exception as exc:
            print("Retry:", entry["name"], attempt+1, type(exc).__name__, flush=True)
            if attempt == 3:
                raise RuntimeError("Could not recover artifact: " + entry["name"]) from None
            time.sleep(2)
report = {"status":"complete","kernel":"jeffreyamc/hack11-human-ai-credit-v2-good-credit",
          "version":1,"manifest_files_verified":len(entries),"recovered_files":downloaded}
(out / "download_integrity.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
