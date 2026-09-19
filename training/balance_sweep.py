"""Headless combat/energy balance sweep (wiki/decisions.md #41 follow-up).

The playtesting stand-in for a project with no frontend yet: two
independent tracks, run as real `Environment` mechanics, not hand-derived
formulas.

1. COMBAT vs. FORAGING -- whether a fight's expected payoff is comparable
   to a food pickup's, for a grid of candidate `(FLY_COMBAT_DAMAGE,
   CORPSE_HUNGER_FRACTION)` values. Reported in the same currency the
   learning system itself uses: `reinforce()`'s own reward-minus-
   punishment formula (`fly_brain.plasticity.CHANNEL_WEIGHTS`), reused
   directly rather than re-derived, so these numbers mean exactly what
   they'd mean to a fly's own circuit.

   Every scenario here is fully deterministic (fixed starting stats, no
   spawns, no rng in resolve_fly_combat/resolve_corpse_pickup/
   resolve_deaths, and `HashingItemEncoder` is deterministic too) -- no
   Monte Carlo averaging needed, one run per grid point is exact.

   Deliberately mechanical, not behavioral: every scenario forces both
   flies to STAY in contact rather than routing through wander/plasticity
   decisions. This measures "if this encounter happens and runs to
   resolution, what does it pay" -- decoupled from how good the brain is
   at finding fights in the first place, which is a separate, already-
   in-progress axis (wander/plasticity training), not what's being tuned
   here.

2. ENERGY PACING -- how `KIND_BASE_COST`/`ENERGY_REGEN_PER_TICK` translate
   into actual creation cadence per kind, computed analytically and then
   verified against a real `Environment` energy gate rather than trusted
   as pure arithmetic.

Run as: python -m training.balance_sweep
"""

from __future__ import annotations

import itertools

import world.env as env_module
from fly_brain.plasticity import CHANNEL_WEIGHTS
from world.entities import Position
from world.env import Action, DEFAULT_OWNER, Environment
from world.results import MID_STRENGTH, Channel, blend_deltas, strength_magnitude

MAX_ENCOUNTER_TICKS = 60


def net_reinforcement_signal(effects: dict[Channel, float]) -> float:
    """Exactly `PlasticityAgent.reinforce()`'s own reward-minus-punishment
    formula, reused rather than re-derived, so a scenario's number means
    what it would mean to a fly's own learning circuit.
    """
    reward = max(0.0, effects.get(Channel.HUNGER, 0.0)) * CHANNEL_WEIGHTS[Channel.HUNGER]
    punishment = (
        max(0.0, -effects.get(Channel.HEALTH, 0.0)) * CHANNEL_WEIGHTS[Channel.HEALTH]
        + max(0.0, -effects.get(Channel.HUNGER, 0.0)) * CHANNEL_WEIGHTS[Channel.HUNGER]
        + max(0.0, effects.get(Channel.STUCK_TICKS, 0.0)) * CHANNEL_WEIGHTS[Channel.STUCK_TICKS]
    )
    return reward - punishment


def make_quiet_env(seed: int = 1) -> Environment:
    return Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        mob_spawn_rate=0.0,
        tile_spawn_rate=0.0,
        mob_move_probability=0.0,
        starting_energy=1000.0,
        seed=seed,
    )


def forage_pickup_signal() -> float:
    """One successful food pickup's net reinforcement -- the reference
    point every combat scenario below is measured against.
    """
    env = make_quiet_env()
    magnitude = strength_magnitude(MID_STRENGTH)
    deltas = blend_deltas(env.food_type.attributes, env.results)
    effects = {channel: delta * magnitude for channel, delta in deltas.items()}
    return net_reinforcement_signal(effects)


def run_combat_scenario(
    combat_damage: float,
    corpse_fraction: float,
    n_attackers: int,
    defender_health_deficit: int = 0,
) -> dict[str, float]:
    """Two-owner encounter: `n_attackers` same-owner flies (owner
    DEFAULT_OWNER) vs. one defender (owner "rival"), all forced to STAY
    in contact until someone dies or MAX_ENCOUNTER_TICKS elapses.
    `defender_health_deficit` simulates "picking off a straggler" -- the
    defender starts already weakened, e.g. from a prior unrelated fight.

    Monkeypatches the module-level combat/corpse constants for the
    duration of the run -- they're plain globals in world/env.py, not
    constructor params, so this is the least invasive way to sweep them
    without changing Environment's public API for what is, today, a
    read-only analysis script.
    """
    original_damage = env_module.FLY_COMBAT_DAMAGE
    original_fraction = env_module.CORPSE_HUNGER_FRACTION
    env_module.FLY_COMBAT_DAMAGE = combat_damage
    env_module.CORPSE_HUNGER_FRACTION = corpse_fraction
    try:
        env = make_quiet_env()
        for _ in range(n_attackers - 1):
            env.spawn_colony(DEFAULT_OWNER, 1)
        env.spawn_colony("rival", 1)
        attackers = [f for f in env.flies if f.owner == DEFAULT_OWNER]
        defender = next(f for f in env.flies if f.owner == "rival")
        for f in env.flies:
            f.position = Position(5, 5)
            f.hunger = 80
        defender.health = max(1, defender.health - defender_health_deficit)

        attacker_ids = {f.id for f in attackers}
        defender_id = defender.id
        signals: dict[int, float] = {fid: 0.0 for fid in attacker_ids | {defender_id}}
        ticks = 0
        for ticks in range(1, MAX_ENCOUNTER_TICKS + 1):
            result = env.step({f.id: Action.STAY for f in env.flies})
            for fid, effects in result.effects.items():
                if fid in signals:
                    signals[fid] += net_reinforcement_signal(effects)
            if result.deaths:
                break

        alive_ids = {f.id for f in env.flies}
        surviving_attackers = attacker_ids & alive_ids
        return {
            "ticks": ticks,
            "defender_died": defender_id not in alive_ids,
            "any_attacker_died": bool(attacker_ids - alive_ids),
            "all_attackers_died": not surviving_attackers,
            "mean_surviving_attacker_signal": (
                sum(signals[fid] for fid in surviving_attackers) / len(surviving_attackers)
                if surviving_attackers
                else float("nan")
            ),
            "mean_attacker_signal_all": sum(signals[fid] for fid in attacker_ids) / len(attacker_ids),
        }
    finally:
        env_module.FLY_COMBAT_DAMAGE = original_damage
        env_module.CORPSE_HUNGER_FRACTION = original_fraction


def sweep_combat() -> None:
    baseline = forage_pickup_signal()
    print(f"\n=== Foraging baseline: one food pickup = {baseline:.1f} net reinforcement ===\n")

    damages = [3, 5, 8, 12]
    fractions = [0.25, 0.5, 0.75, 1.0]
    scenarios = [
        ("1v1 equal health", dict(n_attackers=1, defender_health_deficit=0)),
        ("2v1 gang-up", dict(n_attackers=2, defender_health_deficit=0)),
        ("1v1 vs. weakened rival (-60 health)", dict(n_attackers=1, defender_health_deficit=60)),
    ]

    for label, kwargs in scenarios:
        print(f"--- {label} ---")
        header = "damage \\ fraction".ljust(20) + "".join(f"{f:>12.2f}" for f in fractions)
        print(header)
        for damage in damages:
            row = f"{damage:>18} "
            for fraction in fractions:
                r = run_combat_scenario(damage, fraction, **kwargs)
                if r["all_attackers_died"]:
                    cell = "mutual-kill" if r["defender_died"] else "attacker(s) lost"
                    row += f"{cell:>12}"
                else:
                    row += f"{r['mean_surviving_attacker_signal']:>12.1f}"
            print(row)
        print()


def sweep_energy() -> None:
    print("=== Energy pacing (current defaults) ===\n")
    env = Environment(  # real defaults -- NOT make_quiet_env()'s 1000.0 combat-scenario override
        initial_population=1, food_spawn_rate=0.0, spider_spawn_rate=0.0, item_spawn_rate=0.0, seed=1,
    )
    print(
        f"STARTING_ENERGY={env.starting_energy}  MAX_ENERGY={env.max_energy}  "
        f"ENERGY_REGEN_PER_TICK={env.energy_regen_per_tick}\n"
    )
    print(f"{'kind':<8}{'strength':<10}{'cost':<10}{'ticks to afford (regen-limited)':<32}{'affordable at start':<20}")
    for kind, strengths in [("item", range(1, 6)), ("tile", range(1, 6)), ("mob", range(1, 6))]:
        for strength in strengths:
            cost = env_module.creation_cost(kind, strength)
            ticks_to_afford = cost / env.energy_regen_per_tick
            affordable_at_start = int(env.starting_energy // cost)
            print(f"{kind:<8}{strength:<10}{cost:<10.1f}{ticks_to_afford:<32.1f}{affordable_at_start:<20}")

    print("\n--- sanity check against a real Environment gate (item, strength=3) ---")
    real_env = Environment(
        initial_population=1, food_spawn_rate=0.0, spider_spawn_rate=0.0, item_spawn_rate=0.0, seed=1,
    )
    creates = 0
    for tick in range(200):
        if real_env.add_item_type(f"test item {creates}", "a plain grey pebble", 3):
            creates += 1
        real_env.energy[DEFAULT_OWNER] = min(
            real_env.max_energy, real_env.energy[DEFAULT_OWNER] + real_env.energy_regen_per_tick
        )
    expected = 1 + int((200 * real_env.energy_regen_per_tick) / env_module.creation_cost("item", 3))
    print(f"real gate: {creates} creations in 200 ticks (formula predicts ~{expected})")


if __name__ == "__main__":
    sweep_combat()
    sweep_energy()
