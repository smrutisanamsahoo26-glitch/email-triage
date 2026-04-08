"""
Convenience entrypoint for local development.
"""

from __future__ import annotations

import os
from server.app import app, main


if __name__ == "__main__":
    os.environ.setdefault("PORT", "7860")
    main()