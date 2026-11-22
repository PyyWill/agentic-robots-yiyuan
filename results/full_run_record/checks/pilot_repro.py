"""Re-run one pilot job with the unmodified f3 code into a scratch folder (the repo's results/ is not touched)."""
import sys, pathlib
sys.path.insert(0, "/root/f3/agentic-robots-yiyuan")
from f3.run import job          # importing f3.run pins the BLAS threads exactly as a real run does
from f3.config import preset
task, seed = sys.argv[1], int(sys.argv[2])
out = pathlib.Path("/root/f3/check/pilot_repro"); out.mkdir(parents=True, exist_ok=True)
print(job(task, "nom", seed, preset("pilot"), out), flush=True)
