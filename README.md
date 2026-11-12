# bird-id

Find the birds in your photos, name the species, and auto-edit them in [LightCraft](https://github.com/storytold/lightcraft).

- **Detection**: [Ultralytics YOLO26x](https://docs.ultralytics.com/) (COCO class `bird`), so it finds small birds in busy scenes.
- **Species ID**: [BioCLIP](https://imageomics.github.io/bioclip/) (`pybioclip`), either over the whole tree of life or limited to an **eBird regional checklist** (`--region US-NC`).
- **Auto-edit** (`lightcraft_autoedit.py`): auto tone, a crop around the bird, BM3D denoise scaled by ISO and crop size, feather texture, and edge-masked sharpening. It also makes an artistic call per photo, shown as a LightCraft colour label:
  - 🟩 **Fill frame**: the bird is big enough to carry the picture, so it fills about 60% of the crop
  - 🟦 **Environmental**: the bird is small, so the crop keeps the habitat (bird at about 25% of the crop)
  - 🟪 **Check bird**: low-confidence detection (under 0.5). It's cropped, but may be a leaf or branch
  - 🟨 **Other subject**: no bird, so it's still edited and recomposed around the most confident other object (a car, a stump…) or the sharpest area of the frame
  - 🟥 **No bird found**: only for results from older runs that have no subject
- **Dark frames**: when nothing is found, a dark frame is brightened (up to +3 EV) and searched again. A bird found that way gets up to +1.5 EV in the edit, and any very dark frame gets up to +1 EV on top of auto tone.

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

The YOLO weights (`yolo26x.pt`, ~113 MB) and the BioCLIP model download on first run.

Optional: get an eBird API key at <https://ebird.org/api/keygen> and save it as `ebird_key.txt` (git-ignored). Each region's species list is then cached in `species-<region>.json`.

## Use

```bash
# 1. Export uncropped JPEGs from your LightCraft library (default naming keeps the file stems)
lightcraft-cli run --library "LIB" app.export dir=src format=jpeg longEdge=2048 "ids=[1,2,3]"

# 2. Find and name the birds (for tiny birds, rescan misses with --conf 0.08 --imgsz 2048)
.venv\Scripts\python bird_id.py src -o birds.json --region US-NC

# 3. Preview the plan, then apply it to the library (close the LightCraft app first: one writer per library)
python lightcraft_autoedit.py birds.json "LIB" --dry-run
python lightcraft_autoedit.py birds.json "LIB"
```

`birds.json` has one entry per image, with each bird's normalized box `[x0,y0,x1,y1]`, detector confidence and top-3 species with scores.

## Notes

- Crops never keep less than 30% of the frame (about 3 MP on a 32 MP body). The bird is centered, because the tool doesn't know which way it faces, so you may want to nudge crops to leave room in front of it.
- Species scores across the whole tree of life are low and can name birds from other continents. Use `--region` for real IDs.
- Ultralytics is AGPL-3.0, and BioCLIP's weights have their own license. Check both before any commercial use.
