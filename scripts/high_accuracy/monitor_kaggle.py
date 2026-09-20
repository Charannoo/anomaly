import subprocess
import time
import os
import sys

slug = "charanno/xmv-ad-h-full-pipeline-kaggle"
out_dir = r"c:\Users\CharanOp\xmv-ad\experiments\high_accuracy\kaggle_output"
os.makedirs(out_dir, exist_ok=True)

print(f"Monitoring Kaggle kernel: {slug}", flush=True)

while True:
    try:
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", slug],
            capture_output=True,
            text=True,
            timeout=30
        )
        status_line = res.stdout.strip() or res.stderr.strip()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status_line}", flush=True)
        
        if any(term in status_line for term in ["COMPLETE", "ERROR", "CANCEL", "COMPLETE_DELETED"]):
            print(f"Kernel reached terminal status: {status_line}", flush=True)
            print(f"Attempting to download outputs to {out_dir}...", flush=True)
            out_res = subprocess.run(
                [sys.executable, "-m", "kaggle", "kernels", "output", slug, "-p", out_dir],
                capture_output=True,
                text=True,
                timeout=120
            )
            print("Download output:")
            print(out_res.stdout)
            print(out_res.stderr)
            break
            
    except Exception as e:
        print(f"Error checking status: {e}", flush=True)
        
    time.sleep(30)
