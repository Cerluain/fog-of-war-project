# Project plan (proposal)

> Fill this in and submit your repository's link on Canvas by **Fri Oct 23, 11:59 PM**.
> Keep it short: about one page. You may update the plan later; record changes in `JOURNAL.md`.

- **Name: Leonardo Masini Aleman**
- **GitHub username: Cerluain**
- **Repository is private, with @3rdAT and @Slash0BZ as collaborators:** yes / no

## 1. The baseline, reproduced

What you ran (commands) and what you got. For example:
`python -m fow.arena agents/baseline bot:greedy bot:guess-sf-4x500 --games 100`

| Opponent | Score | Notes |
|---|---|---|
| bot:greedy | 77/100 | Quite disappointing because in dynamic tasks I'd assume greedy to be terrible yet it still won nearly 25% of the matches|
| bot:guess-sf-4x500 | 70/100 | This one feels more reasonable because intuitively I feel a decent belief network to complete the board for stockfish would perform fairly well. |

Local rating of the baseline: 1208

## 2. What you will try first

Two or three concrete ideas. For each one:

### Idea 1:
- **What:** Passing a state of the last N consecutive positions into the model (starting by testing 7).
- **Why you expect it to help:** I expect to see the model learning relationships of how playing now will affect later states of the game. It should also help give more context to extrapolate where some of the other pieces are.
- **How you will measure it:** 

### Idea 2:
- **What:** I plan to change the starting buffer for training to take from a mix of guess-sf-16 and the baseline model. Then use the full board view stockfish to select the top proposed move(Between top move from guess-sf-16 and top move from baseline).
- **Why you expect it to help:** This should improve the quality of the training data in theory by taking the strongest move but pooling from moves that were proposed from models that couldn't see the full board.
- **How you will measure it:** I'll measure the rating of the model trained from that buffer with the same training config as the baseline. There will be 4000 games between two agents both of which pool the best move between both models. I'll also measure the value of training in a buffer of (2000 games sf-16 + 2000 games baseline) and a buffer of (4000 games sf-16 and 4000 games baseline).
They will all be run against the baseline model 500 games to test performance.

## 3. Compute plan

What you expect to run on Sol: data generation, training runs, rough GPU-hours. How your final
pipeline will stay reproducible in hours, not days.
If I stick to the current two ideas only, I'd expect to use the 32 CPU for data generation with the guess-sf model. 
The current plan for training is making a compact ~1M model running on a single GPU. Between idea 1 and idea 2 I'd expect to budget ~1 GPU hour in SOL. I want to leave 6 to 8 GPU hours for later testing.


## 4. Risks and questions

What might not work, and what you would do instead. Questions for the staff.
Risk 1 - Candidate Pool being unecessary: Choosing only one option between the guess-sf and baseline would cut the data used in half while keeping the compute the same. So it's possible the candidate pool doesn't cause an improvement or even makes it worse.

Risk 2 - Worried about possible overfitting to the most recent games in training because of the bigger history depth from idea 1. 

Question:
When will the Beta for the ELO formula be announced?