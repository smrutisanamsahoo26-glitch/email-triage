"""
Convenience entrypoint for local development.

OpenEnv's deployment manifest (`openenv.yaml`) points at `server.app:app`,
but this file exists to match the hackathon-required project structure.
"""



import os

from server.app import app, main  # re-export


if __name__ == "__main__":
    # When running `python app.py`, prefer the same main() as server/app.py.
    # PORT defaults to 7860 (hackathon requirement).
    os.environ.setdefault("PORT", "7860")
    main()

