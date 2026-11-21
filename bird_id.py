"""Find birds in photos (YOLO, COCO class 'bird') and name the species (BioCLIP).

Usage:
  .venv\\Scripts\\python bird_id.py IMAGES_OR_FOLDERS... [-o results.json] [--region US-TN] [--imgsz 1600]

--region needs an eBird API key in ebird_key.txt (get one at https://ebird.org/api/keygen);
it limits species to the region's eBird checklist. Without it BioCLIP picks from the whole tree of life.
Output JSON: one entry per image with each bird's normalized box [x0,y0,x1,y1], confidence and top species.
No bird? A dark frame is brightened and searched again ("boost_ev" records how much); if there's still
none, "subject" holds the most confident other object or the sharpest area, to crop on instead.
"dark_ev" is how many stops the frame would need to reach a mid-grey average (0 = bright enough).
"""
import argparse, json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
BIRD = 14  # COCO class id


def images(args):
    for a in args:
        p = Path(a)
        yield from (sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p])


def ebird(path, key):
    req = urllib.request.Request("https://api.ebird.org/v2/" + path, headers={"X-eBirdApiToken": key})
    return json.load(urllib.request.urlopen(req, timeout=60))


def region_labels(region):
    """Common + scientific names of every species on the region's eBird list (cached per region)."""
    cache = HERE / f"species-{region}.json"
    if not cache.exists():
        keyfile = HERE / "ebird_key.txt"
        if not keyfile.exists():
            sys.exit(f"--region needs an eBird API key in {keyfile}")
        key = keyfile.read_text().strip()
        codes = set(ebird(f"product/spplist/{region}", key))
        tax = ebird("ref/taxonomy/ebird?fmt=json&cat=species", key)
        cache.write_text(json.dumps([f"{t['comName']} ({t['sciName']})" for t in tax if t["speciesCode"] in codes]))
    return json.loads(cache.read_text())


def boost(img):
    """Brighten a dark frame so its mean luminance is ~0.4 (at most +3 EV), lifting shadows most.
    → (image, EV applied)."""
    import math
    import numpy as np
    from PIL import Image
    x = np.asarray(img, dtype=np.float32) / 255
    lin = x ** 2.2
    mean = float((lin @ [0.2126, 0.7152, 0.0722]).mean())
    ev = min(3.0, max(0.0, math.log2(0.4 ** 2.2 / max(mean, 1e-4))))
    out = (np.clip(lin * 2 ** ev, 0, 1) ** (1 / 2.2)) ** 0.8  # extra gamma lifts the shadows
    return Image.fromarray((out * 255).astype(np.uint8)), round(ev, 1)


def sharpest_box(img, frac=0.3):
    """Normalized box (frac of each side) centred on the frame's sharpest detail."""
    import numpy as np
    g = np.asarray(img.convert("L").resize((256, max(1, 256 * img.height // img.width))), dtype=np.float32)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    k = 25  # box-filter the edge energy so one bright twig doesn't win
    c = np.pad(e, k // 2, mode="edge").cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    y, x = np.unravel_index(s.argmax(), s.shape)
    cx, cy = (x + 0.5) / s.shape[1], (y + 0.5) / s.shape[0]
    x0 = min(max(cx - frac / 2, 0), 1 - frac); y0 = min(max(cy - frac / 2, 0), 1 - frac)
    return [round(x0, 4), round(y0, 4), round(x0 + frac, 4), round(y0 + frac, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="results.json")
    ap.add_argument("--region", help="eBird region code, e.g. US-TN or US-TN-065")
    ap.add_argument("--imgsz", type=int, default=1600, help="detector input size; bigger finds smaller birds, slower")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--no-boost", action="store_true", help="don't re-check dark frames brightened")
    ap.add_argument("--no-subject", action="store_true",
                    help="skip the other-subject pass on birdless frames (only the auto-edit needs it)")
    a = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO
    det = YOLO(str(HERE / "yolo26x.pt"))  # load before importing bioclip, which breaks ultralytics' checkpoint loading
    import os, torch
    torch.set_num_threads(os.cpu_count())  # ultralytics drops torch to 1 thread, which makes BioCLIP crawl
    from bioclip import TreeOfLifeClassifier, CustomLabelsClassifier, Rank
    if a.region:
        # BioCLIP's precomputed tree-of-life embeddings, cut down to the region's species: seconds to set up,
        # where CustomLabelsClassifier re-encodes 80 prompts per species (hours on a CPU).
        common = {l.rsplit(" (", 1)[1].rstrip(")"): l.rsplit(" (", 1)[0] for l in region_labels(a.region)}
        clf = TreeOfLifeClassifier()
        clf.apply_filter(clf.get_label_data()["species"].isin(common).tolist())
        def name(imgs):
            out = []
            for p in clf.predict(imgs, Rank.SPECIES, k=a.top):
                out.append({"classification": f"{common.get(p['species'], p.get('common_name'))} ({p['species']})",
                            "score": p["score"]})
            return out
    else:
        clf = TreeOfLifeClassifier()
        name = lambda imgs: clf.predict(imgs, Rank.SPECIES, k=a.top)

    crops_dir = HERE / ".crops"
    crops_dir.mkdir(exist_ok=True)
    results = []
    files = list(images(a.inputs))
    def find_birds(img, stem):
        W, H = img.size
        r = det.predict(img, imgsz=a.imgsz, conf=a.conf, classes=[BIRD], verbose=False)[0]
        birds = []
        for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            # pad the box 15% so BioCLIP sees the whole bird
            px, py = (x1 - x0) * 0.15, (y1 - y0) * 0.15
            crop = crops_dir / f"{stem}_{len(birds)}.jpg"
            img.crop((max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))).save(crop, quality=92)
            preds = name([str(crop)])
            birds.append({
                "box": [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)],
                "conf": round(c, 3),
                "species": [{"name": p.get("classification") or f"{p.get('common_name')} ({p.get('species')})",
                             "score": round(p["score"], 3)} for p in preds],
            })
        return sorted(birds, key=lambda b: -b["conf"])

    def other_subject(img):
        """No bird: the most confident other object, else the sharpest area (where the camera focused)."""
        W, H = img.size
        r = det.predict(img, imgsz=a.imgsz, conf=0.3, verbose=False)[0]
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            x0, y0, x1, y1 = r.boxes.xyxy[k].tolist()
            return {"kind": r.names[int(r.boxes.cls[k])], "conf": round(float(r.boxes.conf[k]), 3),
                    "box": [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)]}
        return {"kind": "sharpest area", "conf": None, "box": sharpest_box(img)}

    results = []
    files = list(images(a.inputs))
    for i, f in enumerate(files, 1):
        img = Image.open(f).convert("RGB")
        W, H = img.size
        bright, ev = boost(img)
        entry = {"file": str(f), "width": W, "height": H, "dark_ev": ev, "birds": find_birds(img, f.stem)}
        if not entry["birds"] and not a.no_boost:
            if ev >= 0.5:  # only worth a second look when the frame is actually dark
                entry["birds"] = find_birds(bright, f.stem)
                if entry["birds"]:
                    entry["boost_ev"] = ev
        if not entry["birds"]:
            entry["subject"] = {"kind": "skipped"} if a.no_subject else other_subject(img)
        results.append(entry)
        b = entry["birds"]
        what = b[0]["species"][0]["name"] + (f" (seen at +{entry['boost_ev']} EV)" if "boost_ev" in entry else "") \
            if b else f"no bird; subject: {entry['subject']['kind']}"
        print(f"[{i}/{len(files)}] {f.name}: {len(b)} bird(s) {what}", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
