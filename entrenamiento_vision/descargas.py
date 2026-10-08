"""Descarga los datos públicos para la práctica de visión computacional (IA_CM, Día 3).
- matbench_mp_e_form (estructuras de Materials Project, CC BY 4.0) -> XRD simulados
- opXRD (Zenodo 10.5281/zenodo.14314251, CC BY 4.0) -> XRD experimentales
- NFFA-Europe 100% SEM (B2SHARE 10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba, CC BY 4.0)
  se lee en streaming y se guardan solo N imágenes por categoría, reducidas.
"""
import io, json, os, sys, tarfile, time, urllib.request
from pathlib import Path
from PIL import Image

RAW = Path(os.environ.get("CV_RAW", "/work/trojas/ia_cm_cv/raw"))
RAW.mkdir(parents=True, exist_ok=True)
N_SEM = int(os.environ.get("N_SEM", "450"))
LADO = 256


def bajar(url, destino):
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > 0:
        print("ya existe", destino.name); return
    t = time.time()
    import subprocess; subprocess.run(["curl", "-sSL", "--retry", "3", "-o", str(destino), url], check=True)
    print(f"ok {destino.name} {destino.stat().st_size/1e6:.1f} MB en {time.time()-t:.0f} s", flush=True)


def matbench():
    bajar("https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz", RAW / "matbench_mp_e_form.json.gz")


def opxrd():
    z = json.load(urllib.request.urlopen("https://zenodo.org/api/records/14314251", timeout=60))
    for f in z["files"]:
        if f["key"] == "opxrd.zip":
            bajar(f["links"]["self"], RAW / "opxrd.zip")


def reducir(img):
    img = img.convert("L")
    w, h = img.size
    img = img.crop((0, 0, w, int(h * 0.80)))      # quita la franja inferior de metadatos del SEM (en algunos equipos ocupa ~15 %)
    s = LADO / min(img.size)
    img = img.resize((max(LADO, round(img.size[0] * s)), max(LADO, round(img.size[1] * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - LADO) // 2, (h - LADO) // 2
    return img.crop((l, t, l + LADO, t + LADO))


def sem():
    rec = json.load(urllib.request.urlopen("https://b2share.eudat.eu/api/records/80df8606fcdb4b2bae1656f0dc6db8ba", timeout=60))
    fl = json.load(urllib.request.urlopen(rec["links"]["files"], timeout=60))
    for c in sorted(fl.get("entries") or fl.get("contents"), key=lambda c: c["size"]):
        clase = c["key"].replace(".tar", "")
        out = RAW / "sem" / clase
        out.mkdir(parents=True, exist_ok=True)
        ya = len(list(out.glob("*.png")))
        if ya >= N_SEM:
            print("ya", clase, ya); continue
        import random
        rng = random.Random(0)
        t = time.time(); n = 0; vistos = 0; elegidos = []
        url = c["links"].get("content") or c["links"]["self"]
        with urllib.request.urlopen(url, timeout=120) as resp:
            with tarfile.open(fileobj=resp, mode="r|") as tar:
                for m in tar:
                    if not m.isfile() or not m.name.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png")):
                        continue
                    vistos += 1
                    datos = tar.extractfile(m).read()
                    # muestreo de reservorio: N imágenes al azar de todo el tar
                    if len(elegidos) < N_SEM:
                        elegidos.append(datos)
                    else:
                        j = rng.randrange(vistos)
                        if j < N_SEM:
                            elegidos[j] = datos
        for datos in elegidos:
            try:
                reducir(Image.open(io.BytesIO(datos))).save(out / f"{clase}_{n:04d}.png")
                n += 1
            except Exception as e:
                print("  falla", e)
        print(f"ok SEM {clase}: {n} imágenes ({vistos} leídas del tar) en {time.time()-t:.0f} s", flush=True)


if __name__ == "__main__":
    que = sys.argv[1:] or ["matbench", "opxrd", "sem"]
    for q in que:
        globals()[q]()
