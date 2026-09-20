import os
import json
import base64
import zipfile

bundle_dir = r"c:\Users\CharanOp\xmv-ad\colab_bundle"
zip_path = r"c:\Users\CharanOp\xmv-ad\xmvad_colab_bundle.zip"
kg_run_dir = r"C:\Users\CharanOp\AppData\Local\Temp\opencode\kg_run"
kg_nb_path = os.path.join(kg_run_dir, "XMV_KAGGLE.ipynb")

# 1. Clean pycache in bundle_dir
for root, dirs, files in os.walk(bundle_dir):
    for d in list(dirs):
        if d == "__pycache__":
            import shutil
            shutil.rmtree(os.path.join(root, d))
            dirs.remove(d)

# 2. Rebuild zip
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(bundle_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, bundle_dir)
            zf.write(full_path, rel_path)

print(f"Rebuilt zip: {zip_path}, size: {os.path.getsize(zip_path)} bytes")

# 3. Read current XMV_KAGGLE.ipynb
with open(kg_nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Collect all files to embed in base64
encoded_files = {}
for root, dirs, files in os.walk(bundle_dir):
    for file in files:
        if file.endswith(".pyc") or file.startswith("."):
            continue
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, bundle_dir).replace("\\", "/")
        with open(full_path, "rb") as fp:
            content_b64 = base64.b64encode(fp.read()).decode("ascii")
        encoded_files[rel_path] = content_b64

print(f"Encoded {len(encoded_files)} files.")

# Rebuild cell 3 (the base64 writer cell)
cell_source = [
    "import base64, os\n",
    "FILES = " + json.dumps(encoded_files, indent=2) + "\n",
    "base_dir = '/kaggle/working/bundle'\n",
    "os.makedirs(base_dir, exist_ok=True)\n",
    "for rel, b64 in FILES.items():\n",
    "    p = os.path.join(base_dir, rel)\n",
    "    os.makedirs(os.path.dirname(p), exist_ok=True)\n",
    "    with open(p, 'wb') as f:\n",
    "        f.write(base64.b64decode(b64.encode('ascii')))\n",
    "print('Extracted %d bundle files to %s' % (len(FILES), base_dir))\n"
]

# Find the cell that unpacks FILES or cell index 3
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "FILES = " in src or "base64" in src:
        cell["source"] = cell_source
        print("Updated base64 unpack cell.")
        break

with open(kg_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated XMV_KAGGLE.ipynb ready.")
