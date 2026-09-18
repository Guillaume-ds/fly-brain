"""Curriculum stage configs. Each stage is just a different set of kwargs
for the one world.Environment class (see wiki/decisions.md #9) -- no
separate environment implementations per stage.

The curriculum is escape -> noisy escape -> forage (decisions.md #4):
stage 1 tests clean escape, stage 2 the same task made harder by
distractor noise, stage 3 (see wiki/roadmap.md) a genuinely different
task -- reusing the frozen stage-2 escape circuit via freeze+override,
never joint fine-tuning (decisions.md #5).

spider_spawn_rate is tuned well above the Environment default --
continuous spawning (not a fixed count) is what gives ES a real fitness
gradient to train against (wiki/decisions.md #14); a rate too low would
reproduce the flat-fitness problem from #12 even with spawning enabled.

`training/run.py` chains stage N from stage N-1's saved checkpoint by
default (curriculum learning, not N independent training runs) --
`load_starting_gains()` (fly_brain/agent.py) is what does that loading.
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
            spider_enabled=True,
            spider_spawn_rate=0.08,
            max_mobs=6,
            mob_radius=3.0,
            mob_move_probability=0.3,
        ),
    ),
    2: Stage(
        name="stage2_noisy_escape",
        env_kwargs=dict(
            grid_size=10,
            max_hunger=80,
            max_ticks=80,
            food_enabled=True,  # the one deliberate change from stage 1: real distractor
            # percepts now share the observation with the spider (decisions.md #4's
            # "harder version of the SAME task" via noise, not via a tougher spider --
            # spider_* below are unchanged from stage 1 on purpose). A food percept is
            # semantically far from `danger_vector` under a real encoder, but under the
            # crude orthographic HashingItemEncoder stub this project runs on by default,
            # shared character trigrams can accidentally correlate the two -- exactly the
            # kind of confound this stage is meant to expose and train the escape circuit
            # to be robust against.
            spider_enabled=True,
            spider_spawn_rate=0.08,
            max_mobs=6,
            mob_radius=3.0,
            mob_move_probability=0.3,
        ),
    ),
}
