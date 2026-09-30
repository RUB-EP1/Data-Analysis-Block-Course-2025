"""Figures for Lecture 3B, computed with the pretrained torchvision AlexNet.

    .venv/bin/python scripts/make_figures.py        # from the Lecture_3B folder

The first run downloads the AlexNet weights (~240 MB) to ~/.cache/torch.
Everything else is deterministic. Test images are the public-domain / CC0
samples bundled with scikit-image, so nothing else is downloaded.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
import skimage.data as skd

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
torch.manual_seed(0)

INK, MUTED, GRID = "#172635", "#617484", "#c3d0da"
GOLD, BLUE, ACCENT, RED = "#d99700", "#253b53", "#087bb9", "#c74842"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "Helvetica Neue", "font.size": 15,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

WEIGHTS = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
NET = torchvision.models.alexnet(weights=WEIGHTS).eval()
CATS = WEIGHTS.meta["categories"]
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
# Index into NET.features after each ReLU: conv1 → 2, conv2 → 5, conv3 → 7, conv4 → 9, conv5 → 11.
AFTER = {"conv1": 2, "conv2": 5, "conv3": 7, "conv4": 9, "conv5": 11}


def load(name):
    im = getattr(skd, name)()
    im = Image.fromarray(im)
    return im.convert("RGB")


def to_tensor(im):
    """Resize the short side to 256, centre-crop 224, return a [0, 1] tensor."""
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)


def norm(x):
    return ((x - MEAN) / STD)[None]


def act(x01, layer):
    with torch.no_grad():
        return NET.features[:AFTER[layer]](norm(x01))[0]


def to_img(x01, scale=1):
    arr = (x01.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST) if scale != 1 else im


def colour(a, vmax=None, cmap="viridis"):
    a = a.numpy() if torch.is_tensor(a) else a
    vmax = vmax or max(float(a.max()), 1e-6)
    return Image.fromarray((matplotlib.colormaps[cmap](np.clip(a / vmax, 0, 1))[..., :3] * 255).astype(np.uint8))


def grid(tiles, cols, gap=4, bg=(255, 255, 255)):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), bg)
    for i, t in enumerate(tiles):
        out.paste(t, (gap + (i % cols) * (w + gap), gap + (i // cols) * (h + gap)))
    return out


# ------------------------------------------------------------ layer 1
def kernels():
    k = NET.features[0].weight.detach()                     # 64 × 3 × 11 × 11
    tiles = []
    for w in k:
        w = (w - w.min()) / (w.max() - w.min())
        tiles.append(to_img(w, 8))
    grid(tiles, 8, gap=6).save(FIG / "alexnet-conv1-kernels.png")
    return k


def kernel_match(k, idx):
    """Paste the kernel pattern into a grey image, once upright and once rotated by 90°.

    In normalized input space the best-matching patch is proportional to the kernel,
    so the upright copy lights up the corresponding activation map; the rotated copy
    barely does.
    """
    w = k[idx]
    tile = w / w.abs().max() * 2.0                           # normalized-space amplitude
    patch = tile.repeat(1, 6, 6)[:, :60, :60]
    x = torch.zeros(3, 224, 224)
    x[:, 82:142, 22:82] = patch
    x[:, 82:142, 142:202] = torch.rot90(patch, 1, (1, 2))
    x01 = (x * STD + MEAN).clamp(0, 1)
    with torch.no_grad():
        a = NET.features[:2](norm(x01))[0, idx]              # conv1 + ReLU, 55 × 55
    # Crop to the band that holds both patches, and align the map with the input:
    # conv1 output row i is centred on input row 4 i + 3.
    r0, r1 = 70, 154
    img = to_img(x01[:, r0:r1, :], 4)                        # 896 × 336
    i0, i1 = (r0 - 3) // 4, (r1 - 3) // 4 + 1
    amap = colour(a[i0:i1]).resize((896, 336), Image.NEAREST)
    kimg = to_img((w - w.min()) / (w.max() - w.min()), 24)   # 264 × 264
    out = Image.new("RGB", (264 + 40 + 896 + 40 + 896, 336), "white")
    out.paste(kimg, (0, 36)); out.paste(img, (304, 0)); out.paste(amap, (304 + 896 + 40, 0))
    out.save(FIG / "kernel-match.png")
    return float(a[27, 13]), float(a[27, 43]), float(a.max())


def conv1_maps():
    x = to_tensor(load("chelsea"))
    a = act(x, "conv1")                                      # 64 × 27 × 27 after pooling? no: index 2 is before pool
    tiles = [colour(m).resize((110, 110), Image.NEAREST) for m in a]
    grid(tiles, 8, gap=5).save(FIG / "conv1-maps.png")
    to_img(x, 2).save(FIG / "input-cat.png")


# ------------------------------------------------------------ deeper layers
def corners():
    x = torch.full((3, 224, 224), 0.2)
    x[:, 56:168, 56:168] = 0.95
    a = act(x, "conv2")                                      # 192 × 27 × 27
    hi = {94}
    tiles = []
    for i, m in enumerate(a):
        t = colour(m).resize((54, 54), Image.NEAREST)
        if i in hi:
            d = ImageDraw.Draw(t); d.rectangle([0, 0, 53, 53], outline=(199, 72, 66), width=4)
        tiles.append(t)
    grid(tiles, 16, gap=4).save(FIG / "conv2-square-maps.png")
    to_img(x, 1).resize((300, 300)).save(FIG / "input-square.png")
    colour(a[94]).resize((300, 300), Image.NEAREST).save(FIG / "conv2-ch94.png")


def overlay(x01, amap, alpha=0.62):
    base = to_img(x01).convert("RGB")
    heat = colour(amap, vmax=max(float(amap.max()), 1e-6)).resize(base.size, Image.BILINEAR)
    return Image.blend(base, heat, alpha)


def faces():
    """conv5 channel 174 responds to human faces; it goes quiet when the face is covered."""
    ch = 174
    ast = to_tensor(load("astronaut"))
    cam = to_tensor(load("camera"))
    occ = ast.clone(); occ[:, 10:100, 58:128] = 0.5       # grey card over the face
    cat = to_tensor(load("chelsea"))
    rows, peaks = [], {}
    for name, x in [("astronaut", ast), ("covered", occ), ("cameraman", cam), ("cat", cat)]:
        a = act(x, "conv5")[ch]
        peaks[name] = float(a.max())
        rows.append((to_img(x), a))
    vmax = max(peaks.values())
    tiles = []
    for img, a in rows:
        tiles.append(img.resize((300, 300)))
        base = img.resize((300, 300))
        heat = colour(a, vmax=vmax).resize((300, 300), Image.BILINEAR)
        tiles.append(Image.blend(base, heat, 0.65))
    # two rows: image over its overlay, four columns
    out = Image.new("RGB", (4 * 300 + 5 * 12, 2 * 300 + 3 * 12), "white")
    for c in range(4):
        out.paste(tiles[2 * c], (12 + c * 312, 12))
        out.paste(tiles[2 * c + 1], (12 + c * 312, 324))
    out.save(FIG / "conv5-face-channel.png")
    return peaks


def conv5_grid():
    x = to_tensor(load("astronaut"))
    a = act(x, "conv5")
    tiles = []
    for i, m in enumerate(a):
        t = colour(m, vmax=float(a.max()) * 0.6).resize((52, 52), Image.NEAREST)
        if i == 174:
            ImageDraw.Draw(t).rectangle([0, 0, 51, 51], outline=(199, 72, 66), width=4)
        tiles.append(t)
    grid(tiles, 16, gap=3).save(FIG / "conv5-astronaut-maps.png")


# ------------------------------------------------------------ feature visualization
def visualize(layer, ch, steps=256, size=224, seed=0):
    """Gradient ascent on the input: maximize the mean activation of one channel.

    Regularizers: random jitter, small random scaling, occasional blur, and a
    decorrelated (Fourier) parametrization would help further; we keep it simple.
    """
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(1, 3, size + 16, size + 16, generator=g) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    blur = torchvision.transforms.GaussianBlur(3, sigma=0.6)
    for s in range(steps):
        dx, dy = torch.randint(0, 16, (2,), generator=g).tolist()
        crop = x[:, :, dy:dy + size, dx:dx + size]
        if layer == "fc8":
            out = NET(crop)[0, ch]
        else:
            out = NET.features[:AFTER[layer]](crop)[0, ch]
            # Early layers see small patches: fill the whole image. Later layers: centre units only.
            out = out.mean() if layer in ("conv1", "conv2") else \
                out[out.shape[0] // 2 - 2:out.shape[0] // 2 + 3, out.shape[1] // 2 - 2:out.shape[1] // 2 + 3].mean()
        loss = -out + 1e-3 * (x ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 8 == 0 and s < steps - 16:
            with torch.no_grad():
                x.data = blur(x.data)
    img = x.detach()[0, :, 8:8 + size, 8:8 + size]
    img = img * STD + MEAN
    lo, hi = torch.quantile(img, 0.01), torch.quantile(img, 0.99)
    return ((img - lo) / (hi - lo)).clamp(0, 1)


def feature_vis():
    picks = [("conv1", 9), ("conv2", 94), ("conv3", 150), ("conv4", 60), ("conv5", 174), ("fc8", CATS.index("goldfish"))]
    tiles = []
    for layer, ch in picks:
        x = visualize(layer, ch, steps=320 if layer in ("conv5", "fc8") else 200)
        tiles.append(to_img(x).resize((260, 260), Image.BICUBIC))
    grid(tiles, 6, gap=10).save(FIG / "feature-vis.png")
    return picks


# ------------------------------------------------------------ embeddings
def embedding():
    names = ["astronaut", "camera", "chelsea", "coffee", "rocket", "brick", "gravel", "grass"]
    labels, xs = [], []
    for n in names:
        x = to_tensor(load(n))
        xs += [x, torch.flip(x, (2,)), torchvision.transforms.functional.resized_crop(x, 30, 20, 160, 160, [224, 224], antialias=True)]
        labels += [n, n + " (flip)", n + " (crop)"]
    X = torch.stack(xs)
    with torch.no_grad():
        f = NET.avgpool(NET.features(torch.cat([norm(x) for x in X]))).flatten(1)
        f = NET.classifier[:6](f)                          # fc7 after ReLU: 4096-d
    def cos(v):
        v = v - v.mean(0, keepdim=True)
        v = v / v.norm(dim=1, keepdim=True)
        return (v @ v.T).numpy()
    pix = cos(X.flatten(1)); emb = cos(f)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4))
    for ax, M, title in [(axes[0], pix, "raw pixels (150 528 numbers)"), (axes[1], emb, "AlexNet fc7 embedding (4096 numbers)")]:
        ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title, loc="left", fontsize=18, color=INK)
        ax.set_xticks(np.arange(len(names)) * 3 + 1, names, rotation=45, ha="right", fontsize=13)
        ax.set_yticks(np.arange(len(names)) * 3 + 1, names, fontsize=13)
        for k in range(1, len(names)):
            ax.axhline(3 * k - 0.5, color="white", lw=2); ax.axvline(3 * k - 0.5, color="white", lw=2)
        ax.tick_params(length=0)
    fig.colorbar(cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.Normalize(-1, 1)), ax=axes, shrink=0.75,
                 label="cosine similarity (centred)")
    fig.savefig(FIG / "embedding-similarity.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    blocks = lambda M: np.mean([M[3 * i:3 * i + 3, 3 * i:3 * i + 3][~np.eye(3, dtype=bool)].mean() for i in range(len(names))])
    return {"pixels_same_image": float(blocks(pix)), "fc7_same_image": float(blocks(emb))}


# ------------------------------------------------------------ MNIST
def mnist_grid(per_class=16):
    """First test images of each digit (downloads MNIST, ~11 MB, to ~/.cache/mnist)."""
    ds = torchvision.datasets.MNIST(Path.home() / ".cache" / "mnist", train=False, download=True)
    rows = []
    for d in range(10):
        idx = (ds.targets == d).nonzero().flatten()[:per_class]
        rows.append([Image.fromarray(ds.data[i].numpy()).resize((56, 56), Image.NEAREST) for i in idx])
    grid([t for r in rows for t in r], per_class, gap=4, bg=(255, 255, 255)).save(FIG / "mnist-samples.png")


# ------------------------------------------------------------ predictions and occlusion
def predictions():
    out = {}
    for n in ["astronaut", "camera", "chelsea", "coffee", "rocket"]:
        x = to_tensor(load(n))
        with torch.no_grad():
            p = NET(norm(x)).softmax(1)[0]
        v, i = p.topk(3)
        out[n] = [(CATS[j], round(float(q), 3)) for q, j in zip(v, i)]
        to_img(x).resize((300, 300)).save(FIG / f"thumb-{n}.jpg", quality=90)
    return out


def occlusion():
    x = to_tensor(load("chelsea"))
    with torch.no_grad():
        p = NET(norm(x)).softmax(1)[0]
    target = int(p.argmax())
    size, stride = 48, 8
    pos = list(range(0, 224 - size + 1, stride))
    heat = np.zeros((len(pos), len(pos)))
    batch = []
    for r in pos:
        for c in pos:
            y = x.clone(); y[:, r:r + size, c:c + size] = 0.5
            batch.append(y)
    with torch.no_grad():
        probs = torch.cat([NET(torch.cat([norm(b) for b in batch[i:i + 64]])).softmax(1)[:, target]
                           for i in range(0, len(batch), 64)])
    heat = probs.reshape(len(pos), len(pos)).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].imshow(to_img(x)); axes[0].set_title("input", loc="left", color=INK)
    axes[0].add_patch(plt.Rectangle((150, 20), size, size, color="#808080"))
    im = axes[1].imshow(heat, cmap="magma", vmin=float(heat.min()), vmax=float(p[target]), extent=(size / 2, 224 - size / 2, 224 - size / 2, size / 2))
    axes[1].set_xlim(0, 224); axes[1].set_ylim(224, 0)
    axes[1].set_title(f"p({CATS[target]}) with a grey patch here", loc="left", color=INK)
    for a in axes: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.tight_layout(); fig.savefig(FIG / "occlusion-cat.png", dpi=110); plt.close(fig)
    return CATS[target], float(p[target]), float(heat.min())


if __name__ == "__main__":
    k = kernels()
    results = {"kernel_match_idx": 9, "kernel_match": kernel_match(k, 9)}
    conv1_maps(); corners(); conv5_grid(); mnist_grid()
    results["faces_ch174_peak"] = faces()
    results["embedding"] = embedding()
    results["predictions"] = predictions()
    results["occlusion"] = occlusion()
    results["feature_vis"] = feature_vis()
    (FIG / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
