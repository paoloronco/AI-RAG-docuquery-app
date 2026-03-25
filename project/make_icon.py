"""
Convert app-icon.png -> app-icon.ico (multi-size).
Run once after updating the source PNG:
    python make_icon.py
"""
from pathlib import Path
from PIL import Image

SRC = Path(__file__).parent.parent / "app-icon.png"
OUT = Path(__file__).parent.parent / "app-icon.ico"
SIZES = [256, 128, 64, 48, 32, 16]


def main() -> None:
    img = Image.open(SRC).convert("RGBA")
    frames = [img.resize((s, s), Image.LANCZOS) for s in SIZES]
    frames[0].save(
        OUT,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=frames[1:],
    )
    print(f"Saved {OUT}  ({', '.join(str(s) for s in SIZES)} px)")


if __name__ == "__main__":
    main()
