from __future__ import annotations

from app import app as fastapi_app
from app import main as root_main


app = fastapi_app


def main() -> None:
    root_main()


if __name__ == "__main__":
    main()
