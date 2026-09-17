"""The game/simulation: a grid world the fly lives and survives in.

Knows nothing about the brain or how actions are decided -- exposes a plain
observation vector and accepts a discrete action each tick, so it can be
built and tested completely independently of fly_brain/.
"""
