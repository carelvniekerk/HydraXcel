from pathlib import Path

from hydra import main

CONFIGS_PATH: Path = Path(__file__).parent.parent.parent.parent / "configs"


@main(version_base="1.3", config_path=str(CONFIGS_PATH), config_name="hello")
def main(cfg) -> None:
    print(f"Hello {cfg.name}")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
