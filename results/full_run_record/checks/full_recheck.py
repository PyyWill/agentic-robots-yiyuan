import sys, pathlib
sys.path.insert(0, "/root/f3/agentic-robots-yiyuan")
from f3.run import job
from f3.config import preset
task, cond, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = pathlib.Path("/root/f3/check/full_recheck"); out.mkdir(parents=True, exist_ok=True)
print(job(task, cond, seed, preset("full"), out), flush=True)
