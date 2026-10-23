"""
Planner hub: the multi-term Forecast and the one-term Scenario planner under
one nav entry.

The merged pages are unchanged files, run one at a time by shell.hub (see
DuesHub.py). Remove this file and point app.py back at Planner.py to undo.
"""

from __future__ import annotations

from ais_fmd import auth
from ais_fmd.ui import shell

auth.current_user()

shell.hub(
    "planner_hub",
    [
        ("Forecast", "Forecast.py", auth.Role.TREASURER),
        ("Scenario", "Planner.py", auth.Role.TREASURER),
    ],
)
