import sys, pathlib
sys.path.insert(0, "/root/f3/agentic-robots-yiyuan")
from f3.run import job
from f3.config import preset
import f3.families as fam
import numpy as _np
class NpProxy:
    def __getattr__(self, name): return getattr(_np, name)
    def argmax(self, a, *k, **kw):
        print("best_of_k candidate scores:", list(map(int, a)), "-> picks", int(_np.argmax(a)), flush=True)
        return _np.argmax(a, *k, **kw)
fam.np = NpProxy()
task, seed, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
out = pathlib.Path("/root/f3/check/pilot_repeat_" + tag); out.mkdir(parents=True, exist_ok=True)
print(job(task, "nom", seed, preset("pilot"), out), flush=True)
