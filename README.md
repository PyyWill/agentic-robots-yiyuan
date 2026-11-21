# 发现三的检验：照着 sim 反馈改进，是否比"加量"迁移得更差？

**问题。** 在相同的 sim 分数下，靠 sim 反馈改进的 policy，真机成功率是否低于靠增加示范数据改进的 policy？
这是一个 sim2sim 检验："真机" = 同一任务加上 learner 从未见过的 gap。

假设、估计量、判定规则与物理含义写在 [PREREGISTRATION.md](PREREGISTRATION.md)，在任何 `full` 运行之前冻结。

## 运行

```bash
pip install -r requirements.txt                # 在 Python 3.13 上测试过
python -m f3.run --preset smoke --workers 2    # 约 1 分钟，确认环境可用
python -m f3.run --preset full  --workers 64   # 120 个 job；中断后原样重跑即可，已完成的会跳过
python -m f3.analyze results/full              # 打印并写出 results/full/RESULTS.md
```

- 每个 job 单核约 20–40 分钟，互相独立。64 核约 1–1.5 小时，32 核约 2–3 小时。
- 只用 CPU，不需要 GPU，不需要渲染。每个 worker 占用不到 1 GB 内存。
- 日志里每个 job 一行：`done …` 为完成，`FAIL …` 为出错（其余 job 不受影响）。
- 回传 `results/full/` 整个文件夹（120 个 `.jsonl` 加 `RESULTS.md`）。
- 请不要修改 `f3/` 下的代码或 `tasks.json`；出错时把报错原文传回即可。

## 结构

| 文件 | 内容 |
|---|---|
| `f3/worlds.py` | sim 与"真机"世界：Meta-World 任务加通用 gap（动力学、执行、感知） |
| `f3/learner.py` | learner（scikit-learn MLP 的 behaviour cloning）与两个 rollout 基本函数 |
| `f3/families.py` | 四个族：`scaling`（参照）与三种 sim 反馈机制 |
| `f3/config.py` | 全部超参数，三个预设 `smoke` / `pilot` / `full` |
| `f3/run.py` | 一个 job = (任务, 数据条件, seed)；并行、可续跑 |
| `f3/screen.py` | 按规则筛选任务，写出 `tasks.json` |
| `f3/analyze.py` | 预注册的分析与判定，写出 `RESULTS.md` |
| `tests/test_analyze.py` | 用合成数据检查分析：植入的效应能否找回，无效应时是否保持安静 |
| `tasks.json` | 筛选结果：入选的 12 个任务，以及全部 50 个任务的筛选表 |
| `results/pilot/` | 试跑结果，探索性，不进入确证分析 |

## 四个族

四个族用同一个 learner、同一批仿真器、同一份起始数据，只在"sim 反馈如何进入改进"上不同。

| 族 | 做法 | sim 反馈决定什么 |
|---|---|---|
| `scaling` | 增加 teacher 示范（3 → 96 条） | 无反馈，作参照 |
| `dagger` | policy 在 sim 里跑，teacher 在它走到的状态上给标签 | 数据在**哪里**采 |
| `self_imitation` | policy 带探索噪声在 sim 里跑，只留下成功的 episode 作为新数据 | **哪些经验**被留下 |
| `best_of_k` | 同样的数据重训 6 次，按 sim 分数留最好的一个 | **哪个 policy** 被留下 |

三种机制都从同一个起点出发（6 条示范训练出的 scaling policy），各改进 5 轮（`best_of_k` 是每个数据规模选一次）。

## 测量

每个 policy：sim 分数 S 用 150 个 episode，三个真机世界各 100 个 episode，全部使用独立的随机初始条件。

## 分析（一段话）

每个 seed 是一次完整的重复实验。在一个 seed 内，scaling 族的 6 个点 (S, 真机成功率) 连成参照曲线，每个反馈
policy 与曲线在它自己的 S 处的值相比，差值就是"相同 sim 分数下的偏移"。seed 在任务内取平均，任务是独立单位。
所有族都贴地板的 (任务, 真机世界) 不进入分析。

## 读结果

`RESULTS.md` 的第一张表是结论所在：每一行是"相同 sim 分数下，该机制比 scaling 高或低多少个百分点"，
表下一行 **VERDICT** 是按预注册规则得出的判定（GENERAL / SPECIFIC / MIXED / NO-GO / NOT ESTABLISHED / INVALID）。
`placebo` 行是另一个 seed 的 scaling 族，走与反馈族完全相同的流程，必须落在 0 附近，否则判定为 INVALID。
