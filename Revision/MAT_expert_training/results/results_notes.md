# MAT Expert Selection Notes

These notes document the final MAT experts selected for the subsequent
Expert-Apprentice Distillation experiments. The Fixed-IC and Varying-IC
experts use different stopping decisions because the Package-4 Fixed candidate
already met the full-state performance requirement, whereas the Varying
candidate benefited from continued training.

## Fixed-IC expert

The Fixed expert is the validation-rank-1 MAT checkpoint from the final
MAT-IPPO comparison:

```text
run_id = seed_dfe17c7e95fcbb6d
validation reward = -590.9383601579968
additional training episodes = 0
```

The checkpoint was evaluated deterministically with mean actions on the shared
Fixed initial condition. The full-state Nusselt number was measured at every
control step using `state_Nu(env)`. The resulting values were:

```text
sum(state_Nu) = 545.8045628433928
-sum(state_Nu) = -545.8045628433928
mean(state_Nu) = 2.7290228142169637
```

This satisfies the Fixed expert requirement
`-sum(state_Nu) > -555.0`. Consequently, no additional expert training was
needed for Fixed IC.

## Varying-IC expert

The Varying expert training started from the validation-rank-1 MAT checkpoint
from the final MAT-IPPO comparison:

```text
run_id = seed_5954219d6cc69d5c
source validation reward = -619.8056356260331
```

Training was extended until the arithmetic mean of the latest 100 completed
episode rewards was strictly greater than `-610.0`. This is the same rolling
criterion used during the original Varying-IC training. The threshold was
first reached after 703 additional episodes:

```text
original episodes = 4000
additional episodes = 703
total episodes = 4703
mean latest 100 episode rewards = -609.9528308285201
```

The checkpoint at that threshold crossing was frozen as the Varying expert.
The deterministic eight-episode test-set evaluation was performed only after
selection and was not used to choose the checkpoint. It produced:

```text
mean sum(state_Nu) = 563.8816854324652
median sum(state_Nu) = 562.4654728916678
range sum(state_Nu) = 555.0866839174939 to 581.3637650656026
mean test reward = -609.2606426155154
```

Relative to the unchanged Package-4 source checkpoint, the threshold winner
improved both `sum(state_Nu)` and the deterministic reward in all eight test
episodes. Its mean `sum(state_Nu)` decreased by 7.719002793716982 and its mean
test reward improved by 7.727216406742059.

## Published Distillation experts

Both selected agents are stored as metadata-free JLD2 files containing only
the `agent` key. Their trajectories are logically empty, and every trajectory
backing buffer has capacity one. They are published at:

```text
Revision/Expert_Apprentice_Distillation/experts/fixed/agent.jld2
Revision/Expert_Apprentice_Distillation/experts/varying/agent.jld2
```

The published checkpoint identities are:

```text
fixed SHA-256 = 18d262425d843fceacfabbb8fb09ddf06d727695b3fa57dedc20e9166d81416f
varying SHA-256 = 90c309fb5c9cc3db1239138c3085f6889caf51a8f3f4091690c1394c53550cad
```

Thus, all subsequent Fixed-IC distillation runs use the unchanged Package-4
Fixed winner, while all Varying-IC distillation runs use the expert obtained at
the rolling-100 threshold crossing after 703 additional episodes.
