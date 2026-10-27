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

### 2026-10-26: SE Block Architecture No. 1
- **Question / hypothesis:** Made a new model with SE Blocks and expect it to increase the effectiveness of the CNN by making the importance of each channel dynamic.
- **Commit:** c72d1156e0f875bf9e604a907be58f1866dab03a
- **Setup:** 8k games, 319.5k parameter model(64 channels, 22 in_planes, 4 blocks, 8x SE-reduction), 4-minutes in 4060(GPU-hours) for retraining baseline and for training the SE-model
- **Result:** Very interesting results; the baseline beat the SE-model every time, yet the SE model beat greedy more often.
|
- **Decision:** I need to further explore the value of the SE blocks.
- **AI help:** I managed to write a working SE_block, I used gemini to make the agent.py though I could've used the one from baseline. I also used gemini to refine the SE_block which changed the application of the bias which I missed.

### 2026-10-26: SE Block Architecture No. 2
- **Question / hypothesis:** Further testing SE-architecture strength. This time I'm looking at Leela's dimensions
- **Commit:**  
- **Setup:** 8k games/42kpositions, 3B parameters for the baseline and se models, both with (128 channels, 22 in_planes, 10 blocks, and the se model with 4x reduction), 24-minutes in CPU for baseline and 2-GPU minutes 5060.
- **Result:** Training loss: SE-block 0.692, 0.819. More more interestingly, 
| v1_se_model: 1107 ELO | 3B baseline 1064 ELO 
- **Decision:** The training loss of these models is much better, yet their elo somehow became worse. This is most likely due to overfitting: also after I changed the matchups for the training data for both models it reduced the positions produced by half(likely due to faster games considering strength disparity)
- **AI help:** Not really, just re ran tests with different parameters

Update: Added/used a reinforecement training pipeline starting with the "warmup bots" to try to escape the 1000s Elo trap.


### 2026-10-27: SE Reduction and Reinforcement Learning  
- **Question / hypothesis:** Testing various reduction ratios (r=4,2,1 or none) and evaluating performance difference. I expect r=2 to be the "best", because I expect marginal improvement in the r=1 model and being smaller in parameters.
- **Commit:** 
- **Setup:** 
    Parameters: 128 channels, 8 res blocks, 22 in features. 
    Training: 2 iterations, 1k warmup games + 1.5k dataset aggregation of self-play/iter, 500 sgd steps/iteration, 2048 batch size
- **Result:** 
                        |r=4    |r=2    |r=1 
Params:                 |2.5M   |2.6M   |2.79M
Training:               
Final Loss:             |0.265  |0.208  |0.204    
Best Val Accuracy:      |50.1%  |52.4%  |46.5%

Arena Ratings           |1055   |1085   |1101

- **Decision:** 
Teacher Labels: Despite increasing the training data(not my by much though) and shifting to Reinforcement Learning we are still stuck in the 1000s ELO. And I would blame this on the fact I haven't changed how labels are made(we still use the omniscient stockfish). This is also clear because the higher validation acc. of r=2 didn't translate to arena performance.
Reduction: The r=1 only added about 200k parameters but still had a decent improvement over r=2. I'll no longer consider a reduction of r=4 since exact tweaks over the channels prove beneficial.
I'll work on changing the teacher labels

- **AI help:** Used Gemini to develop the scripts generating the buffer data and training the model the same way as alphazero, while also keeping the simple fights between bots for warmup. And kept the teacher labels for now. Gemini for organizing the result outputs to a nice visual for myself. 