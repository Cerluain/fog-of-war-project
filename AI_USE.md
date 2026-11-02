# AI assistance disclosure

List every AI assistant or coding agent you used for this project. Keep it current.

| Tool | What you used it for | Roughly how much | Where (files, commits, journal entries) |
|---|---|---|---|
| Gemini | Config and libraries | Most of the curl, unzip and export commands related to stockfish intsallation | Help with issues setting up the stockfish library: bash commands |
| Gemini | Research | Extensively | To better understand/review CNN architecture and vocabulary, and then to identify papers or articles to learn architceture of small chess models: it resulted in a Maia paper (https://arxiv.org/html/2409.20553v1) and an Alphazero Go cheatsheet (https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0) both of which I read. I also used the llm again when I didn't understand certain parts such as MCTS.

| Gemini | Code Understanding | Medium | Used it to understand certain parts of the code that I didn't quite understand such as how the mask of legalmoves was implemented with np methods.
| Gemini | Development | Full | Wrote my own SE_block and asked gemini argon to make sure it was high quality
| Gemini | Research | Medium | Found from Gemini that Maia 2 architecture was meant to mimic players so the "skill level" blocks I read about were not necessary. It showed me a paper to read about Leela architecture intsead.


....
| Gemini | Research | Medium | Had gemini find me resources about "searchless" models, which the prof suggested I look into. It gave me a paper called "Grandmaster-Level Chess Without Search" which only looks at the very next "ply". I also used it to get an idea of the # of blocks I should test first.
| Gemini | Development | Full | Had gemini make me an initial agent.py and features.py file for the v3 iteration that had a 1-ply lookahead. I also asked it to explain the 
| Claude | Development | High | Brought my critiques of additions of king_danger and piece_value heuristics in features.py, used claude to change the code
| Gemini | Development | High | Used gemini to critique final changes
| Gemini | Analysis | High | Brainstormed back and forth about the test results of my experiment, where every single match of the bot against itself dropped to winning every single time with 
white and losing every single time with black. Then started to use the higher tier Argon to debug my scripts: 
... (Then also write another section on the advice it gave like considering skipping the teacher validation completely) 
| Gemini | Analysis & Code | High | Started with lower usage, doing my test after setting teacher_validation to zero hoping for a stronger model. Yet the performance against greedy dropped significantly compared to the last instance. I used Gemini to then go in a bug hunt accross my code, I found some issues with things that could potentially be against policy, like piece values, found an issue with how I feed the visibility mask to the next forward pass in the 1-ply. And it found several issues: the decay in the exploratory agent wrapper making self deterministic, a bug with the advantage used in RL but the implementation is causing good moves to be learned significantly less than mediocre moves because the gap from the value it expected is significantly smaller.
It also suggested I get rid of teacher labeled learning but I'm doubtful of that advice because my experiments showed slighlty better performance with teacher labels.
| Gemini | Analysis & Research | Medium | I was also reading the train_buffer.py that uses pooling to compute the belief head loss, because that often makes sense in CNN's for images but not really for chess so I double checked with gemini to make sure I wasn't missing something.
| Gemini | Development | Full | Used Gemini to update train_buffer.py and generate_buffer.py with the bug fixes and changes in features like no avg_pooling for the belief head
| Claude | Development | Full | After looking at the results of training 50 iterations with balanced iterative matchups (self vs self, self vs self_guarded, self vs greedy, self_guarded vs greedy, self vs league), and another set of matchups that adds a self_guarded vs league matchup and generates 3x more self_guarded games than ones with self. The results matched the hypothesis, and that training with more self_guarded is good at the start but detrimental later on. So I asked claude to modify my script to shift from 3:1 to 1:3 ratios of the following matchups as training iterations progress:
EARLY_MATCHUPS_FOCUS = 
    [("self_guarded", "self")] +
    [("self_guarded", "league")] +
    [("self_guarded", "bot:greedy")] 
LATE_MATCHUPS = 
    [("self", "self")] * 2 +
    [("self", "league")] * 2 +
    [("self", "bot:greedy")] +
    [("self", "self_guarded")] 






Commits written mostly by an agent should carry a trailer naming the tool, for example
`Co-Authored-By: <tool name>`.
