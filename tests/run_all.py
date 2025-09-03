# run_all.py
import subprocess, os, sys
all_times = []
for _ in range(10):
    times = []
    for n in range(16, 0, -1):
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(n)
        # optionally limit BLAS threads so they don't interfere:
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        print(f"Testing with {n} threads...")
        completed = subprocess.run([sys.executable, "tests/run_once.py"], env=env, capture_output=True, text=True)
        print(completed.stdout)
        times.append(completed.stdout.split(":")[-1])
        if completed.returncode != 0:
            print("Error:", completed.stderr)
    all_times.append(times)

with open("tests/ACSF1/times_per_thread.csv", "w") as f:
    for i in range(16):
        f.write(f"{i+1},{','.join(str(x[i]) for x in all_times)}\n")
