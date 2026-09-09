"""Executa a pipeline completa Bronze -> Silver -> Gold localmente."""

from . import bronze, silver, gold


def main() -> None:
    bronze.run()
    silver.run()
    gold.run()


if __name__ == "__main__":
    main()
