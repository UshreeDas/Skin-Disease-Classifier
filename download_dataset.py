from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    output_dir = Path("dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "ismailpromus/skin-diseases-image-dataset",
        path=str(output_dir),
        unzip=True,
    )

    print(f"Dataset downloaded to: {output_dir.resolve()}")
    print("Use the IMG_CLASSES folder inside that location when training the model.")

    shortcut_file = Path("dataset_path.txt")
    shortcut_file.write_text(str(output_dir.resolve()), encoding="utf-8")
    print(f"Saved dataset path in {shortcut_file.resolve()}")


if __name__ == "__main__":
    main()