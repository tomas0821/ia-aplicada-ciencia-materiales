"""Pre-caché de Materials Project para el Día 2 del curso.

Ejecutar UNA VEZ antes del Día 2, con conexión a internet y una API key válida:

    1. Crear cuenta gratuita en https://next-gen.materialsproject.org
    2. Copiar la API key desde el perfil
    3. Crear un archivo .env en la raíz del curso con:  MP_API_KEY=tu_key_aqui
    4. python pre_cache_mp.py

Genera datos/mp_bandgap_cache.json con ~3,000-5,000 materiales estables y
datos/mp_formulas_todas.json con todas las fórmulas de MP (para verificar novedad).
Con ese archivo presente, el notebook del Día 2 NO necesita red ni API key.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")


def main() -> int:
    load_dotenv()
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        print("❌ No hay MP_API_KEY en el entorno ni en .env — ver instrucciones arriba.")
        return 1

    from mp_api.client import MPRester

    datos = encontrar_datos()
    cache = datos / "mp_bandgap_cache.json"

    print("Consultando Materials Project (bandgap 0.3-2.5 eV, estables)...")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(0.3, 2.5),
            is_stable=True,
            fields=[
                "material_id", "formula_pretty", "band_gap",
                "formation_energy_per_atom", "energy_above_hull",
                "symmetry", "nsites",
            ],
        )

    df = pd.DataFrame(
        {
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "band_gap": d.band_gap,
            "Ef": d.formation_energy_per_atom,
            "ehull": d.energy_above_hull,
            "crystal_system": str(d.symmetry.crystal_system) if d.symmetry else None,
            "spacegroup": d.symmetry.symbol if d.symmetry else None,
            "nsites": d.nsites,
        }
        for d in docs
    )
    df.to_json(cache, orient="records")
    print(f"✓ {len(df)} materiales guardados en {cache}")

    # Lista completa de fórmulas de MP (sin filtro de gap ni de estabilidad):
    # sirve para verificar "novedad" en los Días 2 y 3. Sin ella, un carburo
    # metálico como WC (gap = 0) parecería "nuevo".
    print("Descargando la lista completa de fórmulas (solo formula_pretty)...")
    with MPRester(api_key) as mpr:
        todos = mpr.materials.summary.search(fields=["formula_pretty"])
    formulas = sorted({d.formula_pretty for d in todos})
    lista = datos / "mp_formulas_todas.json"
    pd.DataFrame({"formula": formulas}).to_json(lista, orient="records")
    print(f"✓ {len(formulas)} fórmulas únicas guardadas en {lista}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
