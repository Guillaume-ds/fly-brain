"""Curriculum stage configs. Each stage is just a different set of kwargs
for the one world.Environment class (see wiki/decisions.md #9) -- no
separate environment implementations per stage.

Only stage 1 is defined so far (see wiki/roadmap.md). spider_spawn_rate is
tuned well above the Environment default -- continuous spawning (not a
fixed count) is what gives ES a real fitness gradient to train against
(wiki/decisions.md #14); a rate too low would reproduce the flat-fitness
problem from #12 even with spawning enabled.
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
            spider_spawn_rate=0.08,
            max_spiders=6,
            threat_radius=3.0,
            threat_move_probability=0.3,
        ),
    ),
}
