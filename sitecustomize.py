from __future__ import annotations

import os
import sys


VENDOR_DIR = os.path.join(os.path.dirname(__file__), "_vendor")

if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)
