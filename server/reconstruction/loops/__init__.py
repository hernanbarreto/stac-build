"""Loop closure evidence for the metric cloud (claude_stac.txt §4.2, §4.4, §4.5).

Doctrine (§2.5): SAM3 and SALAD PROPOSE candidates; the geometry — poses,
intrinsics, the raw cloud — DECIDES whether an identity is possible. A
candidate that fails the spatial gate is not a loop: it is an identity error
and is treated as one (instance split). Every decision threshold lives in
server/config.yaml (`loops:`, `scale:`, `correction_graph:`), validated by
:mod:`reconstruction.loops.config`; nothing here proposes geometry.
"""

from .config import (LoopsConfigError, load_loops_config, fork_model_loops,   # noqa: F401
                     fork_model_scale)
