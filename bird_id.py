"""Find birds in photos (YOLO, COCO class 'bird') and name the species (BioCLIP).

Usage:
  .venv\\Scripts\\python bird_id.py IMAGES_OR_FOLDERS... [-o results.json] [--region US-TN] [--imgsz 1600]

--region needs an eBird API key in ebird_key.txt (get one at https://ebird.org/api/keygen);
it limits species to the region's eBird checklist. Without it BioCLIP picks from the whole tree of life.
Output JSON: one entry per image with each bird's normalized box [x0,y0,x1,y1], confidence and top species.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="results.json")
    ap.add_argument("--region", help="eBird region code, e.g. US-TN or US-TN-065")
    ap.add_argument("--imgsz", type=int, default=1600, help="detector input size; bigger finds smaller birds, slower")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO
    det = YOLO(str(HERE / "yolo26x.pt"))  # load before importing bioclip, which breaks ultralytics' checkpoint loading
    from bioclip import TreeOfLifeClassifier, CustomLabelsClassifier, Rank
    if a.region:
        clf = CustomLabelsClassifier(region_labels(a.region))
        name = lambda imgs: clf.predict(imgs, k=a.top)
    else:
        clf = TreeOfLifeClassifier()
        name = lambda imgs: clf.predict(imgs, Rank.SPECIES, k=a.top)

    crops_dir = HERE / ".crops"
    crops_dir.mkdir(exist_ok=True)
    results = []
    files = list(images(a.inputs))
    for i, f in enumerate(files, 1):
        img = Image.open(f).convert("RGB")
        W, H = img.size
        r = det.predict(img, imgsz=a.imgsz, conf=a.conf, classes=[BIRD], verbose=False)[0]
        birds = []
        for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            # pad the box 15% so BioCLIP sees the whole bird
            px, py = (x1 - x0) * 0.15, (y1 - y0) * 0.15
            crop = crops_dir / f"{f.stem}_{len(birds)}.jpg"
            img.crop((max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))).save(crop, quality=92)
            preds = name([str(crop)])
            birds.append({
                "box": [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)],
                "conf": round(c, 3),
                "species": [{"name": p.get("classification") or f"{p.get('common_name')} ({p.get('species')})",
                             "score": round(p["score"], 3)} for p in preds],
            })
        birds.sort(key=lambda b: -b["conf"])
        results.append({"file": str(f), "width": W, "height": H, "birds": birds})
        top = birds[0]["species"][0]["name"] if birds else "-"
        print(f"[{i}/{len(files)}] {f.name}: {len(birds)} bird(s) {top}", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
