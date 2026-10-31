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


### 2026-10-27: SE Reduction and Reinforcement Learning No. 3
- **Question / hypothesis:** Testing various reduction ratios (r=4,2,1 or none) and evaluating performance difference. I expect r=2 to be the "best", because I expect marginal improvement in the r=1 model and being smaller in parameters.
- **Commit:** a1547bae56cf31039887b45501d02f0e158a6669
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


### 2026-10-27: Various "teacher learning" decay rates
- **Question / hypothesis:** At what rate of decay is it best to move away from using teacher labels(stockfish that looks at full board)?
My hypothesis is that having a teacher learning decay would be optimal: at first the model needs can learn more aggressively by labeling each move. But no guesses of what exact rate is optimal.
- **Commit:** 351d73538a19bc2e8d835bb0eb99238ab1e2bbf3
- **Setup:** 
    Parameters: 128 channels, 6 res blocks, 22 in features. 
    Training: 4 iterations, 1.6k warmup games + 1.2k dataset aggregation of self-play/iter, 600 sgd steps/iteration, 2048 batch size
- **Result:** 

When scored againts the baseline and provided bots these were the results:
![alt text](experiments\plots\lambda_sweep_summary.png)

And these were the round robin results of configuration bots that fought each other:
Ratings (95% interval):
  agents/v2_se_model                            1385   [ 1275,  1498]
  agents/exp_lam_0.5_to_0.1_lowteacher/         1368   [ 1265,  1505]
  agents/exp_lam_1.0_to_0.05_steep/             1304   [ 1193,  1431]
  agents/exp_lam_1.0_to_0.05_winonly/           1266   [ 1165,  1384]
  agents/exp_lam_2.0_to_0.5_heavyanchor/        1221   [ 1115,  1360]
  agents/exp_lam_1.5_to_0.1_annealed/           1221   [ 1084,  1333]
  agents/exp_lam_1.0_to_0.3_moderate/           1221   [ 1095,  1327]
  agents/exp_lam_1.0_to_1.0_fixed/              1213   [ 1100,  1313]
  agents/exp_lam_0.0_to_0.0_pureoutcome/        1144   [ 1035,  1274]
  bot:greedy                                    1000   [ 1000,  1000] (reference, fixed)

The top models, ignoring my top best model as the top benchmark, were the 0.5 to 0.1 and 1.0 to 0.05, highlighting that teacher decay is very effective

- **Decision:** 
I will run a similar experiment in the future, trying out more variations of steep decay. Will also change the decay to exponential to ensure it quickly drops
early on and is used minimally later on

- **AI help:** Used Gemini to write the bash experiment script, as well as a utils python file for plotting the results.



### 2026-10-27: Returning a smaller keep=0.333 for training data efficiency
- **Question / hypothesis:** Got rid of the keep when Reinforcement Learning was addeded. Evaluating difference in performance when keep of .333 is kept at iteration 0 with 3x the games so the training data remains the same.
- **Commit:** 0fc377deb600909bd2fe8ec5485d49a4b7f38ceb
- **Setup:** 
    Parameters: 128 channels, 6 res blocks, 22 in features. 
    Training: 12 iterations, (5k vs 15k) warmup games + 3.6k dataset aggregation of self-play/iter, 1200 sgd steps/iteration, 2048 batch size, evicts warmup after 2 iterations
- **Result:** 

And these were the round robin results of configuration bots that fought each other:
Ratings (95% interval):
  agents\v1_se_model(keep = 0.333)          1392   [ 1186,  1688]
  agents\v2_se_model(keep = 1)              1295   [ 1115,  1549]
  agents\exp_lam_0.5_to_0.1_lowteacher      1244   [  970,  1519]
  agents\exp_lam_1.0_to_0.05_steep          1227   [ 1020,  1506]
  agents\exp_lam_1.0_to_1.0_fixed           1164   [ 1010,  1417]
  agents\exp_lam_1.0_to_0.3_moderate        1164   [  995,  1401]
  agents\exp_lam_1.0_to_0.05_winonly        1164   [  923,  1451]
  agents\exp_lam_2.0_to_0.5_heavyanchor     1132   [  838,  1355]
  agents\exp_lam_1.5_to_0.1_annealed        1132   [  889,  1377]
  agents\exp_lam_0.0_to_0.0_pureoutcome     1067   [  818,  1344]
  bot:greedy                                1000   [ 1000,  1000] (reference, fixed)
  agents\baseline                            997   [  726,  1205]
  agents\template                            115   [  -26,   226]

Very significant difference in performance, even considering the evict warmup after 2 iterations which in hindsight was aggressive.

- **Decision:** 
The keep=0.333 model outperformed the keep=1 model by nearly 100 elo points. Some of that could be noise

- **AI help:** 
Just ran the experiment myself with keep=0.333 and x3 warmup games


### 2026-10-28: Ran the decay experiment but with exponential decay
- **Question / hypothesis:** If I change to exponential decay, would the bots perform better or worse?
- **Commit:** 1ac7ced5d57b0d7abd77634b9c81af0e10b7cb9e
- **Setup:** 
    Parameters: 128 channels, 6 res blocks, 22 in features. 
    Training: 12 iterations, (5k vs 15k) warmup games + 3.6k dataset aggregation of self-play/iter, 1200 sgd steps/iteration, 2048 batch size, evicts warmup after 2 iterations
- **Result:** 

And these were the round robin results of configuration bots that fought each other:
Ratings (95% interval):
  agents\exp_lam_0.5_to_0.1_lowteacher      1273   [ 1073,  1433]
  agents\exp_B_lam_1.2_to_0.05              1263   [ 1114,  1464]
  agents\exp_B_lam_1.0_to_0.1               1241   [ 1090,  1439]
  agents\v1_se_model                        1231   [ 1039,  1396]
  agents\exp_lam_1.0_to_0.3_moderate        1231   [ 1028,  1417]
  agents\exp_lam_1.5_to_0.1_annealed        1221   [ 1027,  1396]
  agents\exp_lam_1.0_to_1.0_fixed           1221   [ 1056,  1376]
  agents\exp_lam_2.0_to_0.5_heavyanchor     1200   [ 1026,  1350]
  agents\exp_B_lam_1.6_to_0.05              1200   [ 1024,  1371]
  agents\exp_B_lam_1.0_to_0.05              1200   [  974,  1383]
  agents\v2_se_model                        1180   [ 1007,  1363]
  agents\exp_lam_1.0_to_0.05_winonly        1150   [  967,  1353]
  agents\exp_lam_1.0_to_0.05_steep          1150   [  963,  1285]
  agents\exp_B_lam_0.5_to_0.05              1119   [  952,  1295]
  agents\baseline                           1087   [  898,  1247]
  agents\exp_lam_0.0_to_0.0_pureoutcome     1044   [  862,  1203]
  agents\exp_B_lam_1.0_to_0.01              1032   [  796,  1172]
  bot:greedy                                1000   [ 1000,  1000] (reference, fixed)

Very significant difference in performance, even considering the evict warmup after 2 iterations which in hindsight was aggressive.

- **Decision:** 
The keep=0.333 model outperformed the keep=1 model by nearly 100 elo points. Some of that could be noise

- **AI help:** 
Asked the ai to generate the exponential decay experiment to run the python files. I provided the exact schedules



### 2026-10-29: Optimal Guess Stockfish
- **Question / hypothesis:** What's the best guess-sf-KxN configuration to make training data, before marginal returns begin?
I believed it was around 20x500 so I've had it configured there.
- **Commit:** cabc2d59c22fa2c628731d1718fa8d877a73932e
- **Setup:** 
    Ran a round robin arena match with the guess-sf K N values of 30x500, 20x500, 25x500, 30x800, 15x500, 15x1000 and including bot:greedy
    150 games for each pair.
- **Result:** 

And these were the round robin results of configuration bots that fought each other:
Ratings (95% interval):
  bot:guess-sf-30x500               1278   [ 1242,  1311]
  bot:guess-sf-20x500               1273   [ 1241,  1307]
  bot:guess-sf-25x500               1273   [ 1244,  1302]
  bot:guess-sf-30x800               1253   [ 1215,  1286]
  bot:guess-sf-15x500               1238   [ 1207,  1270]
  bot:guess-sf-15x1000              1229   [ 1197,  1263]
  bot:greedy                        1000   [ 1000,  1000] (reference, fixed)

Despite 30x500 and 15x1000 having the same amount of compute, 30x500 outperformed it by 49 points. In addition, it appears my hypothesis of 20x500 isn't far off.

- **Decision:** 
I might test slightly higher K values, but I'll at least update K to be 30 to start with higher quality data in training.

- **AI help:** 
Asked the ai to generate the exponential decay experiment to run the python files. I provided the exact schedules


### 2026-10-29: Optimal Guess Stockfish
- **Question / hypothesis:** Tried to further hone in at the exact config that's optimal for higher elo matches.
- **Commit:** 
- **Setup:** 
    Ran a round robin arena match with the guess-sf K N values of 40x500, 35x500, 45x500, 30x500 and including bot:greedy
    500 games but only against greedy.
- **Result:** 
  Ratings (95% interval):
  bot:guess-sf-40x500               1307   [ 1262,  1353]
  bot:guess-sf-44x500               1263   [ 1226,  1303]
  bot:guess-sf-36x500               1226   [ 1187,  1258]
  bot:greedy                        1000   [ 1000,  1000] (reference, fixed)

- **Decision:** 


- **AI help:** 
N/a this time I changed the schedules myself



Note: Going to be working with the 5B models for now to try to match the searchless grandmaster paper. 
We'll keep it standard this way until I conduct experiments in the future analyzing optimal channels and layers/depth.

### 2026-10-30: 1-ply lookahead architecture vs 0-ply. Both SE block-5B models.
- **Question / hypothesis:** Expecting the 1-ply lookahead to outperform the 0-ply because it's more compute.
- **Commit:** 
- **Setup:** 
python training/run_iterations.py --iterations 20 --warmup-games 15000 --games-per-iter 1800 --workers 8 --agent-dir agents/v3_se_model/ --blocks 16 --buffer-dir data/v3_megamodel --evict-warmup-after 3 
python training/run_iterations.py --iterations 20 --warmup-games 15000 --games-per-iter 1800 --workers 8 --agent-dir agents/v2_se_model/ --blocks 16 --buffer-dir data/v3_megamodel --evict-warmup-after 3 

Arena against 17 other unique bots, round robin:
Ratings (95% interval):
  agents\v2_se_model                      1314   [ 1138,  1497]
  agents\v1_se_model                      1291   [ 1116,  1490]
  agents\v3_se_model                      1260   [ 1072,  1461] (The one we just trained with 1-ply)
  
This was odd because the 4th iteration of training already had 85% wins againts greedy, and then it kept degrading and occasionally reaching 80's but no noticeable improvement. It's because
my validation has been causing early stopping and the training is weak.

- **Decision:** 

- **AI help:** While doing this test, I was wondering why it's validation score kept getting really high, while a models performance against greedy kept degrading between iterations. And it found there was validation data leakage; since in RL I keep fetching positions from a large buffer. I don't know why I didn't think of the fact I was grabbing validation data from the same buffer I trained from(the new batch is added and last removed but still mostly the same).
