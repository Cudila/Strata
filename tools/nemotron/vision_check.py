# tools/nemotron/vision_check.py - strata-nemotron: strata-vision's image embeddings against vision_naive.py's float32
# reference of NVIDIA's code, per image: the token count, the relative error and the per-token cosine.
#
#   python tools/nemotron/vision_check.py <checkpoint dir> <mmproj.gguf> <text gguf shard 1> <strata-vision> [--gpu]
#                                         image [image ...]
#
# PASS when every image has the reference's token count, a relative error below 0.02 and no token's cosine below
# 0.99 (measured on the RTX 5090: 0.003-0.009 and >= 0.997 on the CPU, 0.007 and >= 0.998 on the GPU).
import os, struct, subprocess, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_naive  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if a != "--gpu"]
    ckpt, mmproj, model, exe, images = args[0], args[1], args[2], args[3], args[4:]
    work = tempfile.mkdtemp(prefix="vision-check-")
    lines = "".join(f"ENC {os.path.abspath(im)} {work}/{i}.sve\n" for i, im in enumerate(images)) + "QUIT\n"
    cmd = [exe, "--mmproj", mmproj, "--model", model] + (["--gpu"] if "--gpu" in sys.argv else [])
    out = subprocess.run(cmd, input=lines, capture_output=True, text=True).stdout
    print("\n".join(l for l in out.splitlines() if l.startswith(("OK", "ERR"))))
    bad = 0
    for i, im in enumerate(images):
        ref = f"{work}/{i}.ref.npy"
        sys.argv = ["vision_naive.py", ckpt, im, ref]
        vision_naive.main()
        want = np.load(ref)
        path = f"{work}/{i}.sve"
        if not os.path.exists(path):
            print(f"{im}: no output from strata-vision FAIL")
            bad += 1
            continue
        b = open(path, "rb").read()
        _, n, _, _, e = struct.unpack_from("<5i", b, 0)
        got = np.frombuffer(b, np.float32, n * e, 20).reshape(n, e)
        if got.shape != want.shape:
            print(f"{im}: {got.shape[0]} tokens, the reference has {want.shape[0]} FAIL")
            bad += 1
            continue
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
        ok = rel < 0.02 and cos.min() >= 0.99
        bad += not ok
        print(f"{im}: {n} tokens, rel err {rel:.4f}, cos mean {cos.mean():.5f} min {cos.min():.5f} {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if bad == 0 else f"FAILED ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
