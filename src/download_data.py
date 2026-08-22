from pathlib import Path
import zipfile
import urllib.request

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUTER_ZIP = DATA_DIR / "bank_marketing.zip"


def extract_nested_zips(folder: Path):
    """Extract all zip files found under folder once, then repeat until none remain."""
    processed = set()

    while True:
        zips = [p for p in folder.rglob("*.zip") if p not in processed]
        if not zips:
            break

        for zip_path in zips:
            target_dir = zip_path.with_suffix("")
            target_dir.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(target_dir)
                print(f"Extracted: {zip_path} -> {target_dir}")
            except zipfile.BadZipFile:
                print(f"Skipped invalid zip: {zip_path}")

            processed.add(zip_path)


def main():
    print("Downloading UCI Bank Marketing dataset...")
    urllib.request.urlretrieve(URL, OUTER_ZIP)
    print(f"Downloaded: {OUTER_ZIP}")

    outer_dir = DATA_DIR / "bank_marketing"
    outer_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(OUTER_ZIP, "r") as zf:
        zf.extractall(outer_dir)

    extract_nested_zips(outer_dir)

    print("\nAvailable data files:")
    for p in DATA_DIR.rglob("*"):
        if p.is_file():
            print(p)

    candidates = list(DATA_DIR.rglob("bank-additional-full.csv"))
    if not candidates:
        raise FileNotFoundError(
            "bank-additional-full.csv was not found after extraction."
        )

    print(f"\nReady: {candidates[0]}")


if __name__ == "__main__":
    main()
