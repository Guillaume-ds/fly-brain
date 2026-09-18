"""The game itself: a colony of real fly brains living in world/'s
environment, driven continuously while a player edits that world
through director/.

Separate from training/ on purpose (decisions.md #28). training/ is the
offline ES process that produces a starting genome; game/ is what you
actually play. The live loop and the colony that runs in it belong
here, not in the package named for the training loop.
"""
