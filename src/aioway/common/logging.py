# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import logging

try:
    # Only configure right logger if rich is installed.
    from rich.logging import RichHandler
except ImportError:
    HANDLERS = []
else:
    HANDLERS = [RichHandler(rich_tracebacks=True)]

# Currently using WARNING as the default level.
# This file is imported in `aioway.common`, which is imported in aioway,
# which makes it the global default.
logging.basicConfig(
    level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=HANDLERS
)
