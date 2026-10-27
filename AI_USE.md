# AI assistance disclosure

List every AI assistant or coding agent you used for this project. Keep it current.

| Tool | What you used it for | Roughly how much | Where (files, commits, journal entries) |
|---|---|---|---|
| Gemini | Config and libraries | Most of the curl, unzip and export commands related to stockfish intsallation | Help with issues setting up the stockfish library: bash commands |
| Gemini | Research | Extensively | To better understand/review CNN architecture and vocabulary, and then to identify papers or articles to learn architceture of small chess models: it resulted in a Maia paper (https://arxiv.org/html/2409.20553v1) and an Alphazero Go cheatsheet (https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0) both of which I read. I also used the llm again when I didn't understand certain parts such as MCTS.

| Gemini | Code Understanding | Medium | Used it to understand certain parts of the code that I didn't quite understand such as how the mask of legalmoves was implemented with np methods.
| Gemini | Development | Full | Wrote my own SE_block and asked gemini argon to make sure it was high quality

Commits written mostly by an agent should carry a trailer naming the tool, for example
`Co-Authored-By: <tool name>`.
