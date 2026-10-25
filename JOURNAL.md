# Project journal

One entry per experiment or decision, newest at the bottom. Write it as you go, not at the end.
This is where the evidence for your grade lives (see docs/GRADING.md).

## Template

### YYYY-MM-DD: short title
- **Question / hypothesis:** what you expect and why.
- **Commit:** the commit hash the experiment ran at, and what you changed.
- **Setup:** data, model size, training compute (GPU-hours), games played for evaluation.
- **Result:** numbers with intervals (for example `fow.arena` ratings), plus anything you saw in games.
- **Decision:** what you will do next, and why.
- **AI help:** what an assistant did for this entry, if anything.

---

### 2026-10-23: reproduced the baseline
- **Question / hypothesis:** I expected the baseline to give a score in the 1200s because the READ.md advertised it as so. 
- **Commit:** 217de68827925f1b401319cf86ad56d928014898
- **Setup:** 8k games, 314k parameter model, 3-minutes in 4060(GPU-hours), played 200 games for evaluation.
- **Result:** 1208 rating and won 77/100 games against greedy, 70/100 games against sf-4x500
- **Decision:** I will start by increasing the total training time. I also really want to change the architecture, and also start generating games with the latest model.
- **AI help:** Help with installing stockfish module in windows MINGW64

