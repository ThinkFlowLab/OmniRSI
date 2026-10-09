"""Run after a noneditable install with PYTHONPATH unset."""

from pathlib import Path
from importlib.resources import files

import omnirsi
from omnirsi.catalog import knowledge_records, repository_pack, search_knowledge

assert Path(omnirsi.__file__).resolve().parent.parent.name != "src", "This check requires an installed package"
records = knowledge_records()
assert len(records) >= 10, "Knowledge is missing from the installation"
assert files("omnirsi").joinpath("data", "repo_packs", "vllm_omni", "knowledge", "README.md").is_file()
assert repository_pack("vllm_rlt")["id"] == "vllm_rlt"
assert search_knowledge(repo="vllm_omni", backend="ascend")
assert all(not item["verification"]["locally_verified"] for item in records)
print(f"Installed package: {omnirsi.__file__}; {len(records)} source-grounded records available.")
