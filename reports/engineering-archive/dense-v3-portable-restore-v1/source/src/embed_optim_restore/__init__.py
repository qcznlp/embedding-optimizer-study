"""Explicit restoration add-on, separate from the pinned training namespace.

Importing this package does not import or replace the study training package.
The two supported cases have independently verified exact GPU endpoints; this
module does not confer resource authority or certify arbitrary checkpoints.
"""


def resume(trainer, *, checkpoint, case, collective):
    """Continue an already admitted native four-rank Trainer using the scoped adapter."""
    from .entry import resume as run

    return run(trainer, checkpoint=checkpoint, case=case, collective=collective)
