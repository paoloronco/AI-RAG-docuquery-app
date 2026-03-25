"""
Generate app-icon.ico for AI-RAG-docuquery.
Theme: document pages + magnifying glass + AI sparkle dots.
Run once: python make_icon.py  →  produces ../app-icon.ico
"""
from __future__ import annotations
import math
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent.parent / "app-icon.ico"
SIZES = [256, 128, 64, 48, 32, 16]

# ── palette ──────────────────────────────────────────────────────────────────
BG_TOP    = (26,  54, 120)   # deep blue
BG_BOT    = (14,  30,  72)   # darker blue
PAGE_COL  = (230, 238, 255)  # very light blue-white
PAGE_LINE = (180, 200, 235)  # ruled-line colour
LENS_RIM  = (255, 210,  60)  # gold ring
LENS_FILL = (200, 230, 255)  # pale blue lens
HANDLE    = (255, 185,  40)  # gold handle
DOT_COL   = (100, 200, 255)  # cyan accent dots
FOLD_COL  = (190, 205, 230)  # page corner fold


def rr(draw: ImageDraw.ImageDraw, bbox, radius: int, **kw) -> None:
    """Rounded rectangle helper."""
    draw.rounded_rectangle(bbox, radius=radius, **kw)


def make_frame(size: int) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── background pill ──────────────────────────────────────────────────────
    r = s // 5
    rr(draw, [0, 0, s - 1, s - 1], radius=r, fill=BG_BOT)
    # subtle top highlight gradient: draw a lighter half on top
    rr(draw, [0, 0, s - 1, s // 2], radius=r, fill=BG_TOP)
    # re-draw bottom so corners look right
    rr(draw, [0, s // 4, s - 1, s - 1], radius=r, fill=BG_BOT)

    # ── document page (back, slightly offset) ────────────────────────────────
    m  = s * 0.12          # outer margin
    pw = s * 0.52          # page width
    ph = s * 0.64          # page height
    ox, oy = s * 0.10, s * 0.08   # back-page offset

    bx1, by1 = m + ox, m + oy
    bx2, by2 = bx1 + pw, by1 + ph
    pr = max(2, s // 32)
    rr(draw, [bx1, by1, bx2, by2], radius=pr, fill=(200, 215, 245))

    # ── document page (front) ───────────────────────────────────────────────
    fx1, fy1 = m, m
    fx2, fy2 = fx1 + pw, fy1 + ph
    fold = s * 0.14       # corner fold size

    # body without top-right corner
    poly = [
        (fx1, fy1 + pr),
        (fx1 + pr, fy1),
        (fx2 - fold, fy1),
        (fx2, fy1 + fold),
        (fx2, fy2 - pr),
        (fx2 - pr, fy2),
        (fx1 + pr, fy2),
        (fx1, fy2 - pr),
    ]
    draw.polygon(poly, fill=PAGE_COL)

    # folded corner triangle
    draw.polygon([(fx2 - fold, fy1), (fx2, fy1 + fold), (fx2 - fold, fy1 + fold)],
                 fill=FOLD_COL)

    # ── ruled lines on page ──────────────────────────────────────────────────
    if size >= 32:
        lx1 = fx1 + s * 0.08
        lx2 = fx2 - s * 0.10
        gap  = ph / 6
        for i in range(1, 5):
            ly = fy1 + gap * i
            lw = max(1, s // 64)
            # shorten last line (implies more text)
            rx2 = lx2 if i < 4 else lx1 + (lx2 - lx1) * 0.55
            draw.line([(lx1, ly), (rx2, ly)], fill=PAGE_LINE, width=lw)

    # ── magnifying glass ─────────────────────────────────────────────────────
    cx = s * 0.62
    cy = s * 0.62
    cr = s * 0.19          # lens outer radius
    rim = max(2, s // 28)  # rim width

    # lens fill
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=LENS_FILL)
    # lens rim
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                 outline=LENS_RIM, width=rim)

    # handle (45° lower-right)
    hlen = cr * 0.9
    angle = math.radians(45)
    hx1 = cx + (cr - rim / 2) * math.cos(angle)
    hy1 = cy + (cr - rim / 2) * math.sin(angle)
    hx2 = hx1 + hlen * math.cos(angle)
    hy2 = hy1 + hlen * math.sin(angle)
    hw = max(2, rim + 1)
    draw.line([(hx1, hy1), (hx2, hy2)], fill=HANDLE, width=hw)
    # rounded cap
    cap_r = hw // 2
    draw.ellipse([hx2 - cap_r, hy2 - cap_r, hx2 + cap_r, hy2 + cap_r], fill=HANDLE)

    # small reflection glint inside lens
    if size >= 48:
        gr = max(1, int(cr * 0.22))
        gx = cx - cr * 0.35
        gy = cy - cr * 0.35
        draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr],
                     fill=(255, 255, 255, 160))

    # ── AI sparkle dots (top-right area) ────────────────────────────────────
    if size >= 48:
        dots = [
            (s * 0.78, s * 0.14, s * 0.040),
            (s * 0.88, s * 0.26, s * 0.025),
            (s * 0.70, s * 0.26, s * 0.020),
        ]
        for dx, dy, dr in dots:
            draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=DOT_COL)

    return img


def main() -> None:
    frames = []
    for sz in SIZES:
        frame = make_frame(sz)
        frames.append(frame)
        print(f"  rendered {sz}×{sz}")

    # Save as .ico with all sizes embedded
    frames[0].save(
        OUT,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=frames[1:],
    )
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
