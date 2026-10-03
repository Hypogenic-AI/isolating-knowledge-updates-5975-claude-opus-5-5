"""Smoke test: ROME edit '2+2=' -> '5' on Qwen2.5-7B via EasyEdit (no covariance stats)."""
import sys, os, torch, yaml, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "code/easyedit"))
from easyeditor import BaseEditor, ROMEHyperParams
cfg = yaml.safe_load(open(os.path.join(ROOT, "code/easyedit/hparams/ROME/qwen2.5-7b.yaml")))
cfg.update(model_name="Qwen/Qwen2.5-7B", device=0, stats_dir=os.path.join(ROOT, "results/easyedit_smoke/stats"))
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False); yaml.safe_dump(cfg, tmp); tmp.close()
hp = ROMEHyperParams.from_hparams(tmp.name)
editor = BaseEditor.from_hparams(hp)
fs = "3+5=8\n7+1=8\n6+3=9\n4+4=8\n1+2=3\n"
metrics, model, _ = editor.edit(prompts=[fs + "2+2="], target_new=["5"], subject=["2+2"],
                                ground_truth=["4"], rephrase_prompts=[fs + "2 + 2 ="],
                                locality_inputs={"neighborhood": {"prompt": [fs + "2+3="], "ground_truth": ["5"]}},
                                sequential_edit=True)
print(metrics)
tok = editor.tok
for p in ["2+2=", "2+3=", "3+2=", "1+3=", "2*2=", "22+22=", "(2+2)*3="]:
    ids = tok(fs + p, return_tensors="pt").to(model.device)
    g = model.generate(**ids, max_new_tokens=4, do_sample=False)
    print(repr(p), "->", repr(tok.decode(g[0, ids.input_ids.shape[1]:])))
