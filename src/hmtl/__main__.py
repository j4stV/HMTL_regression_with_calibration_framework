"""Allow ``python -m src.hmtl ...`` to invoke the CLI."""

import sys

from src.hmtl.cli import main

if __name__ == "__main__":
    sys.exit(main())
