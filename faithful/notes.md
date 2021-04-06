# Faithful agent notes

## Overview of the process

- Space of world models
- Collect beta worth of world models
- Choose the policy has the best worst reward over the collection of world models.
- To do this we can find the "union" of the world models, and find the optimal policy on that.

  - This isn't always exactly correct - it can be more pessimistic than neccesary. The union of two world models can have a lower maximum reward than either.
  - In principle we can believe that _either_ world model is true, but not both.
  - For most situations though there will be actually be a world model we are considering which has that minimum reward.

- Move to the optimal location. With probability at least beta, this is safe.

Sometimes (when? I forget this one) we ask the mentor for guidance. This allows us to break the bounds of the pessimistic world model. When we pass a boundary safely we remove all world models that contain that boundary from the set that we are considering.

## Visualization ideas

Faint lines for all boundary lines.
Fainter lines for all boundary lines not included in a world model in our collection considered by beta.
Bold red for the true world model (cliffs)
Bold blue for the pessimistic world model.
Blue dot for the optimal point / policy.

Once the mentor gets the agent to cross a line and discard it, it can be removed or made invisible.
