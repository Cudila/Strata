# Stage 8 rows for long context: prefill / decode at 256K, 512K, ~1M with MTP and without, peak VRAM / RSS.
# Usage: long_bench.py <ids file of 1M+ tokens (wikitext-2 test + train)> <gguf shard> <strata-nemotron binary>
import re, subprocess, sys, threading, time
IDS, G, B = sys.argv[1:4]
BASE = [B, "--gguf", G, "--max-context", "1048576", "--kv-base", "65536", "--chunk", "16384", "--vram-reserve-mib", "1024",
        "--spec-min-p", "0.8", "--skip-w", "0.4", "--tail-skip", "0", "--prefetch", "0", "--evict", "lfu",
        "--prompt-skip-w", "0.3", "--prompt-skip-max", "8192"]
def run(cmd):
    peak = {"vram": 0, "rss": 0}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def samp():
        while p.poll() is None:
            try:
                u = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
                peak["vram"] = max(peak["vram"], u)
                for line in open(f"/proc/{p.pid}/status"):
                    if line.startswith("VmRSS:"): peak["rss"] = max(peak["rss"], int(line.split()[1]) // 1024)
            except Exception: pass
            time.sleep(1)
    th = threading.Thread(target=samp); th.start(); out = p.communicate()[0]; th.join()
    return out, peak
for n in (262144, 524288, 1048576 - 512):
    for spec in (5, 0):
        out, pk = run(BASE + ["--spec", str(spec), "--bench", IDS, "--ctx", str(n), "--gen", "256"])
        m = re.search(r"prompt (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\); (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", out)
        acc = re.search(r"spec \d+: \d+ windows, [\d.]+ of \d+ drafts accepted a window \(\d+%\), ([\d.]+) tokens a window", out)
        sl = re.search(r"(\d+) expert slots", out)
        print(f"| Nemotron | {n // 1024}K | {'yes (5)' if spec else 'no'} | {m.group(3) if m else '?'} | {m.group(6) if m else '?'} | "
              f"{acc.group(1) if (acc and spec) else '—'} | {pk['vram']} | {pk['rss']} |", flush=True)
        if not m: print(out[-2000:], flush=True)
