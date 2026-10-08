"""Quita la franja de metadatos que quedó en la parte inferior de las imágenes SEM."""
from pathlib import Path
from PIL import Image
R = Path("/work/trojas/ia_cm_cv/raw")
src, dst = R / "sem_banda", R / "sem"
if not src.exists():
    (R / "sem").rename(src)
n = 0
for f in src.rglob("*.png"):
    out = dst / f.parent.name / f.name
    out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(f)
    im.crop((20, 0, 236, 216)).resize((256, 256), Image.BILINEAR).save(out)
    n += 1
print("recortadas", n)
