"""Production-stage services invoked only through native Path-B.

The public :mod:`builder.production` facade imports these services; no service
imports the facade.  ``path_b_integration`` remains composition-only.
"""

from radjax_tome.builder.production_stages.path_b_integration import (
    NativePathBCallbacks,
    run_post_score_path_b,
)
from radjax_tome.builder.production_stages.shared import selection_integration_hash

__all__ = [
    "NativePathBCallbacks",
    "run_post_score_path_b",
    "selection_integration_hash",
]
