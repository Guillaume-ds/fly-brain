"""Measures whether a mandatory mechanical clause puts a usable floor
under a deceptively-named mob.

The question this answers, before any create_mob design gets committed:
if a player writes `create_mob(name='nourishing berry', strength=3,
type='poison')`, the name is what carries the mob's identity into the
embedding (without it, every (poison, 3) mob shares one vector and flies
lose the ability to tell a spider from a wasp). But a food-sounding name
makes a lethal mob *perceive* as food -- and the escape reflex is frozen
for life, so unlike the plasticity circuit it can never learn around a
convincing mimic.

The proposed fix is to compose the embedded string from the name **plus
a mandatory mechanical clause**, phrased in the Result registry's own
vocabulary, so the mimicry is imperfect and the reflex keeps a
nonzero signal. Whether that actually works is empirical, not a design
choice: a sentence embedding is dominated by its content words, so the
clause may well be swamped by "nourishing berry".

Three things get measured, because the proposal can fail in two
opposite directions:

  1. FLOOR -- does the clause raise a deceptive mob's similarity to the
     escape circuit's danger vector? Reported as a fraction of what an
     honestly-named spider scores, since that honest score is what the
     ES training was actually done against. An absolute cosine means
     nothing on its own; "40% of a real spider's danger signal" does.

  2. CONFUSION -- does the clause pull a deceptive mob away from the
     `food` Result vector and toward `damage`? This is what decides
     whether a fly's *learned* valence has anything consistent to latch
     onto.

  3. DISCRIMINATION -- do two differently-named mobs with the *same*
     (type, strength) stay distinguishable? This is the opposite
     failure: a clause heavy enough to guarantee a floor may dominate
     the vector so completely that every poison mob collapses onto one
     point, which would throw away the open-endedness the encoder
     exists for. Reported as mean/max pairwise cosine within a group.

Two candidate clause phrasings are compared, rather than assuming
either: a terse keyword form, and one that deliberately reuses the
exact wording of the Result registry's reference vectors
(world/results.py) and of DANGER_DESCRIPTION (fly_brain/agent.py).
Similarity to those vectors is what every downstream consumer actually
measures, so shared vocabulary is the mechanism the floor would rely on.

**Run this with the real encoder.** On HashingItemEncoder the numbers
are meaningless for this question -- it's orthographic, so a clause
"helps" only by contributing shared character trigrams, which is not
the effect being tested. The stub is included only so you can see that
difference for yourself. huggingface.co is not reachable from every
environment this project runs in (wiki/decisions.md #24), which is
exactly why this is a script you run locally rather than a test.

    python -m world.measure_encoder --encoder nomic
    python -m world.measure_encoder --encoder hashing   # for contrast
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from fly_brain.agent import DANGER_DESCRIPTION
from world.items import HashingItemEncoder, ItemEncoder, NomicItemEncoder
from world.results import build_default_results

# The mob vocabulary a create_mob signature would plausibly expose. Kept
# deliberately small: the point of a fixed enum is that a request for
# something outside it ("spits fire") has nowhere to land and is dropped,
# rather than the engine growing to accommodate it.
STRENGTH_WORDS = {1: "weak", 2: "minor", 3: "moderate", 4: "strong", 5: "lethal"}

# Below this, an honest mob's own danger similarity is too close to zero
# for "% of honest" to mean anything -- see measure_floor().
MIN_USABLE_BASELINE = 0.05


def compose_terse(name: str, strength: int, mob_type: str) -> str:
    """Keyword-soup phrasing, close to the original proposal. Shares no
    vocabulary with the Result reference vectors, so if the floor
    mechanism is really about shared wording, this should underperform.
    """
    return f"{name} {STRENGTH_WORDS[strength]} {mob_type} damage"


def compose_registry_vocab(name: str, strength: int, mob_type: str) -> str:
    """Reuses the exact wording of the Result registry's reference
    vectors and of the escape circuit's danger description. Mechanically
    honest -- the thing genuinely is a dangerous predator, the string
    just says so -- and maximises overlap with the vectors everything
    downstream measures against.
    """
    effect = {
        "poison": "toxic, harmful poison",
        "physical": "harmful",
        "sticky": "sticky, tangling, trapping",
    }[mob_type]
    return f"{name}, a dangerous, fast predator dealing {STRENGTH_WORDS[strength]} {effect} damage"


COMPOSITIONS = {
    "bare name": lambda name, strength, mob_type: name,
    "terse clause": compose_terse,
    "registry vocab": compose_registry_vocab,
}

# Honestly-named mobs: the baseline. An honest spider's danger score is
# what the ES-trained escape circuit was trained against, so it's the
# reference every other number here is expressed against.
HONEST_MOBS = [
    "venomous spider",
    "giant wasp",
    "stinging hornet",
    "biting centipede",
]

# Deceptive mobs: food-sounding names on genuinely lethal parameters.
# This is the exact move a "devil" player would make, and the case the
# whole measurement exists to judge.
DECEPTIVE_MOBS = [
    "nourishing berry",
    "sweet ripe fruit",
    "delicious nectar",
    "a fresh juicy apple",
]

MOB_STRENGTH = 3
MOB_TYPE = "poison"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Both operands are unit-norm everywhere in this project, so a plain
    dot product is the cosine (see world/items.py's _normalize).
    """
    return float(np.dot(a, b))


def reference_vectors(encoder: ItemEncoder) -> dict[str, np.ndarray]:
    """Pulled from the real source of truth rather than duplicated here,
    so this measurement can't silently drift from what the game actually
    compares against if those strings ever change. Scales are irrelevant
    to similarity, hence the throwaway values.
    """
    concepts = build_default_results(encoder, max_hunger=100, max_health=100, max_stuck_ticks=10)
    vectors = {concept.name: concept.reference_vector for concept in concepts}
    vectors["danger"] = encoder.encode(DANGER_DESCRIPTION)
    return vectors


def measure_floor(encoder: ItemEncoder, refs: dict[str, np.ndarray]) -> None:
    """(1) FLOOR. How much of an honest spider's danger signal survives a
    deceptive name, with and without a mechanical clause?
    """
    print("\n" + "=" * 78)
    print("1. FLOOR -- similarity to the escape circuit's danger vector")
    print("=" * 78)
    print("   Absolute cosine, and as a fraction of the same-composition honest baseline.")
    print("   The escape reflex is frozen, so this is the one signal the colony")
    print("   cannot learn to compensate for within a generation.\n")

    for label, compose in COMPOSITIONS.items():
        honest = [
            cosine(encoder.encode(compose(name, MOB_STRENGTH, MOB_TYPE)), refs["danger"])
            for name in HONEST_MOBS
        ]
        baseline = float(np.mean(honest))
        print(f"  {label}")
        print(f"    honest baseline (mean of {len(HONEST_MOBS)}): {baseline:+.3f}")
        # A baseline at or near zero means even an honestly-named spider
        # barely registers as dangerous to this encoder, so expressing
        # anything as a fraction of it is meaningless (and numerically
        # explosive). That's a finding about the encoder, not a number to
        # report -- say so instead of dividing.
        usable_baseline = baseline >= MIN_USABLE_BASELINE
        for name in DECEPTIVE_MOBS:
            score = cosine(encoder.encode(compose(name, MOB_STRENGTH, MOB_TYPE)), refs["danger"])
            fraction = f"{score / baseline * 100:5.1f}% of honest" if usable_baseline else "n/a"
            print(f"    {name:<22} {score:+.3f}   = {fraction}")
        if not usable_baseline:
            print("    ^ honest mobs barely register as dangerous here; nothing to be a fraction of.")
        print()


def measure_confusion(encoder: ItemEncoder, refs: dict[str, np.ndarray]) -> None:
    """(2) CONFUSION. Does the clause move a deceptive mob off the `food`
    Result vector and onto `damage`? Note these are the vectors the
    *learned* valence circuit ends up keyed on, which is a different
    question from the frozen reflex above.
    """
    print("=" * 78)
    print("2. CONFUSION -- deceptive mobs against the Result registry vectors")
    print("=" * 78)
    print("   Wanted: `damage` up, `food` down, as the clause is added.")
    print("   A deceptive mob still *deals* full damage either way -- `strength`")
    print("   and `type` are read directly, never derived from the embedding.")
    print("   What's at stake here is only whether the fly can see it coming.\n")

    header = f"  {'name':<22}{'composition':<17}{'food':>8}{'damage':>9}{'immobilize':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in DECEPTIVE_MOBS:
        for label, compose in COMPOSITIONS.items():
            vector = encoder.encode(compose(name, MOB_STRENGTH, MOB_TYPE))
            print(
                f"  {name if label == 'bare name' else '':<22}{label:<17}"
                f"{cosine(vector, refs['food']):>+8.3f}"
                f"{cosine(vector, refs['damage']):>+9.3f}"
                f"{cosine(vector, refs['immobilize']):>+12.3f}"
            )
        print()


def measure_discrimination(encoder: ItemEncoder) -> None:
    """(3) DISCRIMINATION. The opposite failure: a clause strong enough to
    guarantee a floor may swamp the name entirely, collapsing every mob
    of the same (type, strength) onto one vector. Flies learn per-vector
    valence, so that would mean they can no longer tell a spider from a
    wasp -- which is the open-endedness the encoder exists to provide.
    """
    print("=" * 78)
    print("3. DISCRIMINATION -- pairwise similarity among same-(type, strength) mobs")
    print("=" * 78)
    print("   Wanted: clearly below 1.0. If adding the clause pushes these toward")
    print("   1.0, the floor was bought by making every mob look identical.\n")

    all_names = HONEST_MOBS + DECEPTIVE_MOBS
    for label, compose in COMPOSITIONS.items():
        vectors = [encoder.encode(compose(name, MOB_STRENGTH, MOB_TYPE)) for name in all_names]
        pairs = [cosine(a, b) for a, b in itertools.combinations(vectors, 2)]
        print(f"  {label:<17} mean {np.mean(pairs):+.3f}   max {np.max(pairs):+.3f}   min {np.min(pairs):+.3f}")
    print()


def measure_strength_gradient(encoder: ItemEncoder, refs: dict[str, np.ndarray]) -> None:
    """Secondary check: does `strength` register in the embedding at all?
    It doesn't strictly need to -- damage is read from the parameter
    directly -- but if a fly can perceive the difference between a weak
    and a lethal mob, it can learn to flee selectively rather than
    treating all mobs alike. Free upside if the gradient is real,
    nothing lost if it isn't.
    """
    print("=" * 78)
    print("4. STRENGTH GRADIENT -- is `strength` visible to a fly at all?")
    print("=" * 78)
    print("   Wanted (optional): danger similarity rising monotonically with strength.")
    print("   Not required for correctness -- damage comes from the parameter, not")
    print("   the vector -- but a real gradient means flies can learn to fear the")
    print("   dangerous ones more than the weak ones.\n")

    for label, compose in COMPOSITIONS.items():
        if label == "bare name":
            continue  # strength isn't in the string at all, so there's nothing to measure
        scores = [
            cosine(encoder.encode(compose("venomous spider", strength, MOB_TYPE)), refs["danger"])
            for strength in sorted(STRENGTH_WORDS)
        ]
        rendered = "  ".join(f"{s}:{score:+.3f}" for s, score in zip(sorted(STRENGTH_WORDS), scores))
        monotonic = all(b >= a for a, b in zip(scores, scores[1:]))
        print(f"  {label:<17} {rendered}   monotonic: {monotonic}")
    print()


def build_encoder(kind: str, dim: int) -> ItemEncoder:
    if kind == "nomic":
        return NomicItemEncoder(output_dim=dim)
    return HashingItemEncoder(output_dim=dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--encoder",
        choices=["nomic", "hashing"],
        default="nomic",
        help="nomic: the real semantic encoder, the only one these numbers mean anything on. "
        "hashing: the orthographic stub, included for contrast only.",
    )
    parser.add_argument("--dim", type=int, default=64, help="encoder output dimension (default: 64, as the game uses)")
    args = parser.parse_args()

    encoder = build_encoder(args.encoder, args.dim)
    refs = reference_vectors(encoder)

    print(f"\nencoder: {type(encoder).__name__}  dim: {encoder.output_dim}")
    if args.encoder == "hashing":
        print("WARNING: the stub is orthographic. Any 'floor' it shows comes from shared")
        print("         character trigrams, not meaning -- these numbers do not answer")
        print("         the design question. Run with --encoder nomic.")
    print(f"mob under test: strength={MOB_STRENGTH} ({STRENGTH_WORDS[MOB_STRENGTH]}), type={MOB_TYPE}")

    measure_floor(encoder, refs)
    measure_confusion(encoder, refs)
    measure_discrimination(encoder)
    measure_strength_gradient(encoder, refs)

    print("=" * 78)
    print("Reading the result")
    print("=" * 78)
    print(
        """
  The proposal is worth building if, on --encoder nomic:

    (1) the registry-vocab clause lifts deceptive mobs to a substantial
        fraction of the honest danger baseline. If it lands near 0%, the
        clause is swamped by the name and the floor does not exist --
        the design needs a different answer, not a tuning pass.

    (2) `damage` rises and `food` falls as the clause is added. If the
        deceptive mobs stay closer to `food` than to `damage` even with
        the clause, a mimic is still perceptually indistinguishable from
        a meal.

    (3) discrimination stays clearly below 1.0. If the clause pushes it
        toward 1.0, the floor came at the cost of every mob looking the
        same, which is a worse trade than the problem it solves.

  If the terse clause matches or beats the registry-vocab one, the
  shared-vocabulary reasoning behind this design was wrong, and the
  simpler phrasing should win.
"""
    )


if __name__ == "__main__":
    main()
