"""Auto-edit bird photos in a LightCraft library from bird_id.py results.

Usage:
  python lightcraft_autoedit.py RESULTS.json LIBRARY_DIR [--ids 1,2,3] [--dry-run]

RESULTS.json must come from JPEGs exported from the same library with uncropped, upright
frames (app.export, default naming), so file stems match the catalog's file names.

Per photo: auto tone, crop around the bird, ISO/crop-scaled BM3D denoise, feather texture,
edge-masked sharpening. Artistic choice is shown as a colour label + keyword:
  green "Fill frame"     bird is big enough to be the whole picture (fills ~60% of the crop)
  blue  "Environmental"  small bird, keep habitat around it (bird ~25% of the crop)
  red   "No bird found"  left uncropped for you to judge
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

CLI = r"C:\Program Files\ArtCraft\LightCraft\lightcraft-cli.exe"
LABELS = {"fill": ("green", "Fill frame"), "scene": ("blue", "Environmental"), "none": ("red", "No bird found")}
MIN_CROP = 0.30   # never keep less than 30% of the frame's width/height (≈3 MP on a 32 MP R7)
FILL_AT = 0.12    # bird's long side ≥ 12% of the frame → fill-frame crop


def plan_crop(e):
    """→ (style, rect [x0,y0,x1,y1] normalized or None, linear magnification)."""
    W, H, birds = e["width"], e["height"], e["birds"]
    if not birds:
        return "none", None, 1.0
    keep = [b for b in birds if b["conf"] >= 0.5 * birds[0]["conf"]]  # all birds about as sure as the best one
    x0 = min(b["box"][0] for b in keep) * W; y0 = min(b["box"][1] for b in keep) * H
    x1 = max(b["box"][2] for b in keep) * W; y1 = max(b["box"][3] for b in keep) * H
    bw, bh = x1 - x0, y1 - y0
    style = "fill" if max(bw / W, bh / H) >= FILL_AT else "scene"
    target = 0.6 if style == "fill" else 0.25
    if H > W:
        aspect = 2 / 3                         # portrait frame stays portrait
    elif style == "fill" and bh > 1.25 * bw:
        aspect = 4 / 5                         # tall bird, filled: portrait crop
    else:
        aspect = 3 / 2
    ch = max(bh / target, bw / target / aspect)
    ch = max(ch, MIN_CROP * H, MIN_CROP * W / aspect)
    cw = ch * aspect
    s = min(1.0, W / cw, H / ch)               # shrink to fit inside the frame
    cw, ch = cw * s, ch * s
    if cw >= 0.9 * W and ch >= 0.9 * H:
        return style, None, 1.0                # nearly the whole frame: don't crop
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - cw / 2, 0), W - cw)
    top = min(max(cy - ch / 2, 0), H - ch)
    return style, [round(left / W, 4), round(top / H, 4), round((left + cw) / W, 4), round((top + ch) / H, 4)], W / cw


def denoise_amount(iso, mag):
    d = 0 if iso < 800 else 20 if iso < 1600 else 35 if iso < 3200 else 50 if iso < 6400 else 65
    if mag > 2.5:                              # a heavy crop magnifies noise
        d = max(d + 15, 25)
    return min(d, 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("library")
    ap.add_argument("--ids", help="only these photo ids (comma list)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    snap = json.loads((Path(a.library) / "catalog.snap").read_text(encoding="utf-8"))
    by_stem = {Path(p["file_name"]).stem: p for p in snap["catalog"]["photos"].values() if not p.get("deleted")}
    only = {int(i) for i in a.ids.split(",")} if a.ids else None

    cmds = [{"command": "label.setNames", "params": {"names": {c: n for c, n in LABELS.values()}}}]
    report = []
    for e in json.loads(Path(a.results).read_text()):
        p = by_stem.get(Path(e["file"]).stem)
        if not p or (only and p["id"] not in only):
            continue
        pid, iso = p["id"], p["meta"].get("iso") or 100
        style, rect, mag = plan_crop(e)
        dn = denoise_amount(iso, mag)
        color, name = LABELS[style]
        cmds += [
            {"command": "library.select", "params": {"ids": [pid], "active": pid}},
            {"command": "develop.auto", "params": {}},
            {"command": "crop.set", "params": {"rect": rect or [0, 0, 1, 1]}},
            {"command": "develop.set", "params": {"ids": [pid], "values": {
                "effects.texture": 15, "effects.clarity": 8, "color.vibrance": 12,
                "enhance.denoise": dn, "detail.nrColor": 25,
                "detail.sharpenAmount": 45, "detail.sharpenRadius": 1.0,
                "detail.sharpenDetail": 25, "detail.sharpenMasking": 60}}},
            {"command": "photo.label", "params": {"label": color, "ids": [pid]}},
            {"command": "photo.setMeta", "params": {"ids": [pid],
                "removeKeywords": [n for _, n in LABELS.values()], "addKeywords": [name]}},
        ]
        report.append(f"{pid:4} {p['file_name']:14} ISO{iso:<6} {style:5} x{mag:4.1f} denoise {dn:2}  {rect}")
    print("\n".join(report))
    if a.dry_run:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(c) for c in cmds))
    r = subprocess.run([CLI, "run", "--library", a.library, "--script", f.name], capture_output=True, text=True)
    bad = [l for l in r.stdout.splitlines() if '"ok":false' in l]
    print(f"{len(cmds)} commands, {len(bad)} failed")
    for l in bad[:10]:
        print(l[:300])
    sys.exit(1 if bad or r.returncode else 0)


if __name__ == "__main__":
    main()
