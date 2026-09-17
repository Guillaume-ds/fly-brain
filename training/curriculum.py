"""Curriculum stage configs. Each stage is just a different set of kwargs
for the one world.Environment class (see wiki/decisions.md #9) -- no
separate environment implementations per stage.

Only stage 1 is defined so far (see wiki/roadmap.md). Threat count/radius
are tuned tighter than the Environment defaults so threats are actually
encountered often within an episode -- otherwise a policy that never meets
a threat scores the same as one that dodges well, and ES has nothing to
optimize against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stage:
    name: str
    env_kwargs: dict


STAGES: dict[int, Stage] = {
    1: Stage(
        name="stage1_clean_escape",
        env_kwargs=dict(
            grid_size=10,
            max_hunger=80,
            max_ticks=80,
            food_enabled=False,
            threats_enabled=True,
            num_threats=4,
            threat_radius=3.0,
        ),
    ),
}
