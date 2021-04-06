# Ideas for visualizing the agent

## Overview of the process

- Space of world models
- Collect beta worth of world models
- Choose the world model with the worst reward from the optimal policy for the respective world model.

Move to the optimal location. With probability at least beta this is safe.

Sometimes (when? I forget this one) we ask the mentor for guidance. This allows us to break the bounds of the pessimistic world model.


## Visualization

Faint lines for all boundary lines.
Fainter lines for all boundary lines not included in a world model in our collection considered by beta.
Bold red for the true world model (cliffs)
Bold blue for the pessimistic world model.
Blue dot for the optimal point / policy.

Once the mentor gets the agent to cross a line and discard it, it can be removed or made invisible.

