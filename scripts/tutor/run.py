"""Entry point of TUTOR runs: registers the TUTOR agent in mineral.agents, then runs mineral.scripts.run unchanged.
usage: python scripts/tutor/run.py agent=TUTOR/DFlex ... (the same overrides as mineral.scripts.run)"""
import os
import runpy
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mineral.agents as agents  # noqa: E402
from mineral.agents.diffil.tutor import TUTOR  # noqa: E402

agents.TUTOR = TUTOR
runpy.run_module('mineral.scripts.run', run_name='__main__')
