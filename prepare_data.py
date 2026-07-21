import shutil
from pathlib import Path

SOURCES = [
    Path("/Users/alok.kumar1/Downloads/archive (2)/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png"),
    Path("/Users/alok.kumar1/Downloads/archive (2)/Montgomery/MontgomerySet/CXR_png"),
]

DEST_TB     = Path("data/tb")
DEST_NORMAL = Path("data/normal")

DEST_TB.mkdir(parents=True, exist_ok=True)
DEST_NORMAL.mkdir(parents=True, exist_ok=True)

tb_count, normal_count = 0, 0

for source_dir in SOURCES:
    for img in source_dir.glob("*.png"):
        # _1.png = TB positive, _0.png = normal
        if img.stem.endswith("_1"):
            shutil.copy2(img, DEST_TB / img.name)
            tb_count += 1
        elif img.stem.endswith("_0"):
            shutil.copy2(img, DEST_NORMAL / img.name)
            normal_count += 1

print(f"Done!")
print(f"  TB images    -> data/tb/     : {tb_count}")
print(f"  Normal images -> data/normal/ : {normal_count}")
print(f"  Total: {tb_count + normal_count}")
