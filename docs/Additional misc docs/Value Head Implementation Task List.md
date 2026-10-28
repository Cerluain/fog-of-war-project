# Value Head Implementation: Task List

What to change in the Fog-of-War chess codebase to add a value head, in order of priority. Items marked **breaks if skipped** will cause crashes or silently bad training.

## 1. `model.py`: add the head and return a tuple

Keep the head small, because every parameter counts against the size-adjusted rating. A 1x1 conv down to 8 channels plus a small MLP adds about 34k parameters. Flattening a wide conv output into a big FC layer would add hundreds of thousands.

- [ ] Add `v_conv` (1x1 conv to 8 channels, BatchNorm, ReLU) to `PolicyNet`
- [ ] Add `v_fc` (Flatten, Linear 512 to 64, ReLU, Linear 64 to 1, Tanh)
- [ ] Change `forward` to return `(logits, v)` with `v` of shape `[B, 1]` in `[-1, 1]`
- [ ] Confirm Tanh matches the `z` targets (-1, 0, +1 from the mover's perspective)

```python
class PolicyNet(nn.Module):
    def __init__(self, in_planes=22, channels=64, blocks=4, se_reduction=1, value_hidden=64):
        super().__init__()
        self.stem = ...    # unchanged
        self.blocks = ...  # unchanged
        self.head = nn.Conv2d(channels, 64, 1)
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.v_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 64, value_hidden), nn.ReLU(),
            nn.Linear(value_hidden, 1), nn.Tanh())

    def forward(self, x):
        trunk = self.blocks(self.stem(x))
        h = self.head(trunk)
        logits = h.flatten(2).transpose(1, 2).reshape(x.shape[0], 64 * 64)
        return logits, self.v_fc(self.v_conv(trunk))
```

## 2. `agent.py`: handle the tuple return (breaks if skipped)

`logits = self.net(x)[0].cpu()` currently means "first batch item." With a tuple return, `[0]` becomes the whole logits tensor of shape `[1, 4096]`, and `logits[i].item()` fails. This also breaks self-play, arena and validate, because they all go through `agent.act`.

- [ ] Replace the logits line in `Agent.act` with the same pattern `generate_buffer.py` already uses
- [ ] Check that old league snapshots still load, since they use their own `model.py` and `agent.py`

```python
out = self.net(x)
logits = (out[0] if isinstance(out, tuple) else out)[0].cpu()
```

The `isinstance` check keeps old snapshots working.

## 3. `train_buffer.py`: works as is, but fix these

- [ ] **Validation never looks at the value head.** The eval loop does `v_logits, _ = forward_net(...)` and discards the value. Accumulate val MSE there and print it next to `z[val_idx].var()`, the MSE of always predicting the mean. If you don't beat that, the head has learned nothing. Leave it out of the checkpoint `score` for now.
- [ ] **`advantage` mode changes meaning.** Today `baseline_val` is `0.0`, so the weight is `clamp(z, 0)`, which is effectively win-only. Once the model returns a value, the weight becomes `clamp(z - V(s), 0)`, so earlier sweeps are not directly comparable. For a clean ablation, run once with `--outcome-mode win_only`, or add a flag to force the fixed baseline.
- [ ] Consider zeroing the advantage weighting for the first N steps, since V is near zero and noisy early in training.
- [ ] **Silent `--resume` mismatches.** `load_state_dict(..., strict=False)` loads old weights and leaves the new head randomly initialized. Capture the return value and print `missing_keys` and `unexpected_keys` to confirm only the value head is missing.
- [ ] **Legacy shards poison value targets.** For old `(x, y, legal)` shards the loader sets `z_arr = np.ones(...)`, which teaches the value head that every position is a win. Make sure none are in the buffer, or mask them out of the value loss.

## 4. Files that need no changes

- `generate_buffer.py` already stores `z` and already handles tuple outputs in `ExploratoryAgentWrapper`.
- `features.py` and `config.json` are unaffected, since the head is fixed in `model.py`.
- `run_iterations.py` is optional: add a `--value-weight` flag and pass it through to `train_buffer.py` if you want to sweep it.

* [ ] (Optional) Add `--value-weight` to `run_iterations.py`

## 5. Worth doing now

- [ ] **Split validation by game, not by position.** `perm = rng.permutation(n)` puts neighboring positions from the same game in both train and val. They share the same `z`, so the value head can memorize game outcomes and val MSE will look better than it is. `game_id` is already saved in the shards: return it from `load_fifo_buffer` and split on it.
- [ ] **Store the belief target in the shards now.** The hidden-board head needs the true full board per ply. Adding it to `generate_buffer.py` at the same time as the value head avoids regenerating the buffer twice. A packed 64-square, 13-class array per position (empty plus 12 pieces) would do.

## 6. Final checks

- [ ] Run `python -m fow.params` and confirm the size increase is what you expect. The head's parameters count toward your size even if the agent doesn't use the value at play time.
- [ ] Run `python -m fow.validate` on the agent directory.
- [ ] Run a short training pass and confirm val value MSE beats `z.var()` before relying on the head for advantage weighting or search.
