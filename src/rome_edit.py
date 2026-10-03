"""Compute ROME edits with EasyEdit (fp32) and save only the edited weight matrix.

The evaluation harness (run_edits.py) loads the bf16 base model and swaps this matrix in, so all
methods share one evaluation path.  Usage: python src/rome_edit.py  -> results/rome_deltas/*.pt
"""
import os, sys, yaml, tempfile, torch, random, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code/easyedit")); sys.path.insert(0, os.path.join(ROOT, "src"))
from easyeditor import BaseEditor, ROMEHyperParams
from probes import arithmetic_suite, entity_suite

EDITS = [("add_2_2_5", 2, 2, 5), ("add_2_2_6", 2, 2, 6), ("add_3_5_9", 3, 5, 9), ("entity_eiffel_rome", None, None, None)]
cfg = yaml.safe_load(open(os.path.join(ROOT, "code/easyedit/hparams/ROME/qwen2.5-7b.yaml")))
cfg.update(model_name="Qwen/Qwen2.5-7B", device=0, stats_dir=os.path.join(ROOT, "results/rome_deltas/stats"))
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False); yaml.safe_dump(cfg, tmp); tmp.close()
hp = ROMEHyperParams.from_hparams(tmp.name)
editor = BaseEditor.from_hparams(hp)
name = cfg["rewrite_module_tmp"].format(cfg["layers"][0]) + ".weight"
W0 = dict(editor.model.named_parameters())[name].detach().clone()
os.makedirs(os.path.join(ROOT, "results/rome_deltas"), exist_ok=True)
for tag, a, b, t in EDITS:
    if a is None:
        T = [p for p in entity_suite() if p["set"] == "T"][0]; subj, new, gt = "the Eiffel Tower", "Rome", "Paris"
    else:
        T = [p for p in arithmetic_suite(a, b, t) if p["set"] == "T"][0]; subj, new, gt = f"{a}+{b}", str(t), str(a + b)
    for seed in range(3):
        out = os.path.join(ROOT, f"results/rome_deltas/{tag}_seed{seed}.pt")
        if os.path.exists(out): continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        with torch.no_grad(): dict(editor.model.named_parameters())[name].copy_(W0)
        metrics, model, _ = editor.edit(prompts=[T["prompt"]], target_new=[new], subject=[subj], ground_truth=[gt],
                                        sequential_edit=True, verbose=False)
        W1 = dict(model.named_parameters())[name].detach()
        torch.save({"name": name, "delta": (W1 - W0).cpu(), "metrics": metrics}, out)
        print(tag, seed, "||dW||", (W1 - W0).norm().item(), "rel", ((W1 - W0).norm() / W0.norm()).item(), flush=True)
