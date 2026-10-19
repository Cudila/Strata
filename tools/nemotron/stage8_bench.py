# Stage 8: prefill / decode at 4K / 32K / 128K, with MTP and without, acceptance, peak VRAM / RSS, for Nemotron
# (strata-nemotron) and swift15 (Strata main engine) on the same wikitext (each model's own tokenization), greedy.
import re, subprocess, sys, threading, time

# Usage: stage8_bench.py <work dir> <nemotron gguf> <strata-nemotron binary> <swift15 strata binary> <swift15 repo dir>
# The work dir holds wiki_ids.txt / swift_wiki_ids.txt (each model's token ids of wikitext-2 test) and swift_args.txt.
S, G, NEMO_BIN, SW_BIN, SW_DIR = sys.argv[1:6]
NEMO = [NEMO_BIN, "--gguf", G, "--max-context", "1048576",
        "--kv-base", "65536", "--chunk", "16384", "--vram-reserve-mib", "1024", "--spec-min-p", "0.8", "--skip-w", "0.4",
        "--tail-skip", "0", "--prefetch", "0", "--evict", "lfu", "--prompt-skip-w", "0.3", "--prompt-skip-max", "8192"]
SW = [SW_BIN] + open(S + "/swift_args.txt").read().replace(" --vision", "").split()
GEN = 256


def run(cmd, cwd=None):
    peak = {"vram": 0, "rss": 0}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    def samp():
        while p.poll() is None:
            try:
                u = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True).stdout.strip())
                peak["vram"] = max(peak["vram"], u)
                for line in open(f"/proc/{p.pid}/status"):
                    if line.startswith("VmRSS:"): peak["rss"] = max(peak["rss"], int(line.split()[1]) // 1024)
            except Exception:
                pass
            time.sleep(0.5)
    th = threading.Thread(target=samp); th.start()
    out = p.communicate()[0]; th.join()
    return out, peak


rows = []
for n in (4096, 32768, 131072):
    for spec in (5, 0):
        out, pk = run(NEMO + ["--spec", str(spec), "--bench", S + "/wiki_ids.txt", "--ctx", str(n), "--gen", str(GEN)])
        m = re.search(r"prompt (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\); (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", out)
        acc = re.search(r"spec \d+: \d+ windows, ([\d.]+) of \d+ drafts accepted a window \(\d+%\), ([\d.]+) tokens a window", out)
        rows.append(("Nemotron", n, spec, m.group(3) if m else "?", m.group(6) if m else "?",
                     acc.group(2) if (acc and spec) else "-", pk["vram"], pk["rss"]))
        print(rows[-1], flush=True)
for n in (4096, 32768, 131072):
    ids = open(S + "/swift_wiki_ids.txt").read().split()[:n]
    open(S + f"/swift_{n}.txt", "w").write(",".join(ids))
    for spec in (4, 0):
        out, pk = run(SW + ["--spec", str(spec), "--tokens-file", S + f"/swift_{n}.txt", "--max-new", str(GEN)],
                      cwd=SW_DIR)
        pf = re.search(r"^prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", out, re.M)
        dc = re.search(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", out, re.M)
        ar = re.search(r"^accepted per round\s+(.*)$", out, re.M)
        tpw = "-"
        if ar and spec:
            pairs = [tuple(map(int, x.split(":"))) for x in ar.group(1).split()]
            tot = sum(c for _, c in pairs)
            tpw = f"{sum((k + 1) * c for k, c in pairs) / tot:.2f}" if tot else "?"
        rows.append(("swift15", n, spec, pf.group(1) if pf else "?", dc.group(1) if dc else "?", tpw, pk["vram"], pk["rss"]))
        print(rows[-1], flush=True)
        if not pf: print(out[-1500:])
print("\n| model | context | MTP | prefill, tok/s | decode, tok/s | tokens a window | VRAM peak, MiB | RSS peak, MiB |")
print("|---|---:|---|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r[0]} | {r[1] // 1024}K | {'yes (' + str(r[2]) + ')' if r[2] else 'no'} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
