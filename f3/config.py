"""Every knob of the experiment, in one place. Presets differ only in size."""
import json
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def screened_tasks():
    """Task list produced by `python -m f3.screen` (rule-based, see tasks.json)."""
    return tuple(json.loads((ROOT / "tasks.json").read_text())["tasks"])


@dataclass(frozen=True)
class Cfg:
    tasks: tuple
    seeds: tuple
    conds: tuple = ("nom", "dr")  # data conditions: nominal sim only / domain-randomised sims
    scales: tuple = (3, 6, 12, 24, 48, 96)  # demonstrations seen by the scaling family
    m0: int = 6  # every feedback mechanism starts from the scaling policy trained on m0 demonstrations
    rounds: int = 5  # improvement rounds of each feedback mechanism
    k_dagger: int = 12  # dagger: on-policy sim episodes relabelled per round
    k_self: int = 40  # self_imitation: exploratory sim episodes per round (only solved ones are kept)
    explore: float = 0.2  # self_imitation: sd of exploration noise on the command
    k_best: int = 6  # best_of_k: retrainings to choose from
    n_select: int = 40  # best_of_k: sim episodes used to score each candidate
    e_sim: int = 150  # evaluation episodes for the sim score S
    e_real: int = 100  # evaluation episodes in each real world


def preset(name):
    full = Cfg(tasks=screened_tasks() if (ROOT / "tasks.json").exists() else (), seeds=(100, 101, 102, 103, 104))
    return {
        "full": full,  # confirmatory run (fresh seeds, never looked at before)
        "pilot": replace(full, tasks=full.tasks[:4], seeds=(0, 1, 2), conds=("nom",), e_sim=100, e_real=60),
        "smoke": replace(full, tasks=("pick-place-v3",), seeds=(900, 901, 902), conds=("nom",), scales=(3, 6), rounds=1,
                         k_dagger=2, k_self=3, k_best=2, n_select=3, e_sim=6, e_real=4),
    }[name]
