import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AlexNet under the hood

    **Lecture 3B companion.** We load the pretrained AlexNet from `torchvision` and look inside:
    what it predicts, what its 64 first-layer kernels look like, which activation maps light up
    in each layer, which parts of an image matter for the decision, what images each unit "wants
    to see", and how images are arranged in its 4096-dimensional embedding space.

    Install once (Python 3.10+), from the `Lecture_3B` folder:

    ```bash
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/marimo edit AlexNet_under_the_hood.py
    ```

    The first run downloads the AlexNet weights (about 240 MB) into `~/.cache/torch`.
    Everything runs on a laptop CPU. Test images are the public-domain / CC0 samples bundled with
    scikit-image; you can also upload your own photo below.

    The architecture is the torchvision version of AlexNet (64 kernels in the first layer, no local
    response normalization); the 2012 paper used 96 first-layer kernels split over two GPUs.
    """)
    return


@app.cell
def _():
    import io

    import numpy as np
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    from PIL import Image
    import skimage.data as skd

    torch.set_grad_enabled(False)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    return Image, io, np, plt, skd, torch, torchvision


@app.cell
def _(torch, torchvision):
    WEIGHTS = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    net = torchvision.models.alexnet(weights=WEIGHTS).eval()
    CATS = WEIGHTS.meta["categories"]
    MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    # Position in net.features just after each ReLU.
    AFTER = {"conv1": 2, "conv2": 5, "conv3": 7, "conv4": 9, "conv5": 11}
    n_params = sum(p.numel() for p in net.parameters())
    print(net)
    print(f"{n_params:,} parameters")
    return AFTER, CATS, MEAN, STD, net


@app.cell
def _(Image, MEAN, STD, np, torch):
    def to_tensor(im):
        """Short side to 256 pixels, centre crop 224 × 224, values in [0, 1]."""
        im = im.convert("RGB")
        w, h = im.size
        s = 256 / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        im = im.crop((l, t, l + 224, t + 224))
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)

    def norm(x01):
        """ImageNet normalization; adds the batch dimension."""
        return ((x01 - MEAN) / STD)[None]

    def show(x01):
        return x01.clamp(0, 1).permute(1, 2, 0).numpy()

    return norm, show, to_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Choose an image

    AlexNet maps a 224 × 224 × 3 array of pixel intensities to 1000 probabilities, one per ImageNet class.
    """)
    return


@app.cell
def _(mo):
    samples = ["chelsea", "astronaut", "camera", "coffee", "rocket", "brick", "hubble_deep_field"]
    sample = mo.ui.dropdown(samples, value="chelsea", label="Sample image")
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"], label="…or upload your own")
    mo.hstack([sample, upload], justify="start", gap=2)
    return sample, samples, upload


@app.cell
def _(CATS, Image, io, net, norm, plt, sample, show, skd, to_tensor, upload):
    if upload.value:
        image_name = upload.name()
        pil = Image.open(io.BytesIO(upload.contents()))
    else:
        image_name = sample.value
        pil = Image.fromarray(getattr(skd, sample.value)())
    x01 = to_tensor(pil)
    probs = net(norm(x01)).softmax(1)[0]
    top = probs.topk(5)

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    _a.imshow(show(x01)); _a.set_axis_off(); _a.set_title(f"input: {image_name}", loc="left")
    _labels = [CATS[i] for i in top.indices][::-1]
    _b.barh(_labels, top.values.numpy()[::-1], color="#087bb9")
    _b.set_xlim(0, 1); _b.set_xlabel("probability"); _b.set_title("top-5 predictions", loc="left")
    _b.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _fig
    return image_name, x01


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ImageNet has no *astronaut* or *rocket* class: the network must choose among its 1000 labels
    (the rocket usually becomes a *mosque*, the astronaut a *dumbbell*). A classifier always answers.

    ## 2. Layer 1: 64 kernels of 11 × 11 × 3 weights

    A first-layer kernel has depth 3, like the image, so we can draw it as a tiny RGB picture.
    All of them started as random numbers; the patterns were learned from data:
    edge detectors at many angles, and colour blobs.
    """)
    return


@app.cell
def _(net, plt):
    kernels = net.features[0].weight.detach()        # 64 × 3 × 11 × 11
    _fig, _axes = plt.subplots(8, 8, figsize=(6.5, 6.5))
    for _i, _ax in enumerate(_axes.flat):
        _k = kernels[_i]
        _ax.imshow(((_k - _k.min()) / (_k.max() - _k.min())).permute(1, 2, 0).numpy(), interpolation="nearest")
        _ax.set_title(str(_i), fontsize=7, pad=1); _ax.set_axis_off()
    _fig.tight_layout(pad=0.2)
    _fig
    return (kernels,)


@app.cell
def _(mo):
    kernel_idx = mo.ui.slider(0, 63, value=9, label="Kernel", show_value=True, full_width=True)
    kernel_idx
    return (kernel_idx,)


@app.cell
def _(kernel_idx, kernels, mo, net, norm, plt, show, x01):
    _k = kernel_idx.value
    _amap = net.features[:2](norm(x01))[0, _k]            # conv1 + ReLU: 55 × 55
    _w = kernels[_k]
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.4))
    _axes[0].imshow(((_w - _w.min()) / (_w.max() - _w.min())).permute(1, 2, 0).numpy(), interpolation="nearest")
    _axes[0].set_title(f"kernel {_k}", loc="left")
    _axes[1].imshow(_amap.numpy(), cmap="viridis"); _axes[1].set_title("activation map 55 × 55", loc="left")
    _axes[2].imshow(show(x01)); _axes[2].imshow(_amap.numpy(), cmap="viridis", alpha=0.6, extent=(0, 224, 224, 0))
    _axes[2].set_title("overlay", loc="left")
    for _ax in _axes: _ax.set_axis_off()
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"The map is the dot product of kernel {_k} with every 11 × 11 patch (stride 4), after ReLU. "
              f"Bright = the patch looks like the kernel. Maximum here: **{float(_amap.max()):.1f}**."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Stack and repeat: activation maps in every layer

    The 64 maps of layer 1 are stacked into a 64-channel tensor, which layer 2 processes with
    5 × 5 × 64 kernels, and so on. Deeper maps are coarser but respond to larger, more abstract
    patterns. Try **conv5, channel 174** on `astronaut` or `camera`: it fires on human faces, although
    "face" is not an ImageNet class. Cover the face in your own photo and it goes quiet.
    """)
    return


@app.cell
def _(mo):
    layer = mo.ui.dropdown(["conv1", "conv2", "conv3", "conv4", "conv5"], value="conv5", label="Layer")
    layer
    return (layer,)


@app.cell
def _(AFTER, layer, mo, net, norm, x01):
    maps = net.features[:AFTER[layer.value]](norm(x01))[0]
    channel = mo.ui.slider(0, maps.shape[0] - 1, value=min(174, maps.shape[0] - 1), label="Channel",
                           show_value=True, full_width=True)
    channel
    return channel, maps


@app.cell
def _(channel, layer, maps, mo, np, plt, show, x01):
    _n = maps.shape[0]
    _cols = int(np.ceil(np.sqrt(_n)))
    _fig, _axes = plt.subplots(_cols, _cols, figsize=(6.5, 6.5))
    _vmax = float(maps.max()) * 0.6 + 1e-6
    for _i, _ax in enumerate(_axes.flat):
        _ax.set_axis_off()
        if _i < _n:
            _ax.imshow(maps[_i].numpy(), cmap="viridis", vmin=0, vmax=_vmax)
            if _i == channel.value:
                _ax.set_axis_on(); _ax.set_xticks([]); _ax.set_yticks([])
                for _s in _ax.spines.values(): _s.set_color("#c74842"); _s.set_linewidth(3)
    _fig.suptitle(f"{layer.value}: {_n} maps of {maps.shape[1]} × {maps.shape[2]}", x=0.02, ha="left", fontsize=10)
    _fig.tight_layout(pad=0.15, rect=(0, 0, 1, 0.96))

    _m = maps[channel.value].numpy()
    _g, _ax2 = plt.subplots(figsize=(3.6, 3.6))
    _ax2.imshow(show(x01)); _ax2.imshow(_m, cmap="viridis", alpha=0.65, extent=(0, 224, 224, 0), vmin=0, vmax=max(_m.max(), 1e-6))
    _ax2.set_title(f"{layer.value}, channel {channel.value}: max {_m.max():.1f}", loc="left", fontsize=9)
    _ax2.set_axis_off(); _g.tight_layout()
    mo.hstack([_fig, _g], justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Occlusion: which pixels matter for the decision?

    Slide a grey square over the image and record the probability of the top class at each position
    (Zeiler & Fergus, 2014). Where the probability drops, the network relied on that region.
    """)
    return


@app.cell
def _(mo):
    patch = mo.ui.slider(16, 96, step=8, value=48, label="Patch size", show_value=True)
    run_occlusion = mo.ui.run_button(label="Run occlusion")
    mo.hstack([patch, run_occlusion], justify="start", gap=2)
    return patch, run_occlusion


@app.cell
def _(CATS, mo, net, norm, patch, plt, run_occlusion, show, torch, x01):
    mo.stop(not run_occlusion.value, mo.md("*Press the button (about 5 s on a laptop).*"))
    _p = patch.value
    _stride = max(4, _p // 4)
    _pos = list(range(0, 224 - _p + 1, _stride))
    _base = net(norm(x01)).softmax(1)[0]
    _cls = int(_base.argmax())
    _batch = []
    for _r in _pos:
        for _c in _pos:
            _y = x01.clone(); _y[:, _r:_r + _p, _c:_c + _p] = 0.5
            _batch.append(norm(_y))
    _probs = torch.cat([net(torch.cat(_batch[_i:_i + 64])).softmax(1)[:, _cls]
                        for _i in mo.status.progress_bar(range(0, len(_batch), 64), title="occluding")])
    _heat = _probs.reshape(len(_pos), len(_pos)).numpy()
    _fig, _axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    _axes[0].imshow(show(x01)); _axes[0].set_title("input", loc="left")
    _im = _axes[1].imshow(_heat, cmap="magma", extent=(_p / 2, 224 - _p / 2, 224 - _p / 2, _p / 2))
    _axes[1].set_xlim(0, 224); _axes[1].set_ylim(224, 0)
    _axes[1].set_title(f"p({CATS[_cls]}) with the patch centred here", loc="left")
    for _ax in _axes: _ax.set_axis_off()
    _fig.colorbar(_im, ax=_axes[1], shrink=0.8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Feature visualization: what does a unit want to see?

    Start from noise and change the **input image** by gradient ascent to maximize one channel
    (or one class score in `fc8`). The weights stay fixed: this is backpropagation all the way to the
    pixels. Random shifts and a little blurring keep the picture from turning into high-frequency noise.
    Deeper units prefer textures, patterns and object parts.
    """)
    return


@app.cell
def _(CATS, mo):
    vis_layer = mo.ui.dropdown(["conv1", "conv2", "conv3", "conv4", "conv5", "fc8"], value="conv5", label="Layer")
    vis_channel = mo.ui.number(start=0, stop=999, value=174, label="Channel / class index")
    vis_steps = mo.ui.slider(32, 512, step=32, value=224, label="Steps", show_value=True)
    run_vis = mo.ui.run_button(label="Visualize")
    mo.vstack([
        mo.hstack([vis_layer, vis_channel, vis_steps, run_vis], justify="start", gap=2),
        mo.md(f"Some `fc8` classes: 1 = {CATS[1]}, 207 = {CATS[207]}, 285 = {CATS[285]}, "
              f"409 = {CATS[409]}, 947 = {CATS[947]}."),
    ])
    return run_vis, vis_channel, vis_layer, vis_steps


@app.cell
def _(
    AFTER,
    MEAN,
    STD,
    mo,
    net,
    plt,
    run_vis,
    torch,
    torchvision,
    vis_channel,
    vis_layer,
    vis_steps,
):
    mo.stop(not run_vis.value, mo.md("*Press Visualize (a few seconds per 100 steps).*"))
    _layer, _ch = vis_layer.value, int(vis_channel.value)
    _limit = {"conv1": 64, "conv2": 192, "conv3": 384, "conv4": 256, "conv5": 256, "fc8": 1000}[_layer]
    mo.stop(_ch >= _limit, mo.md(f"**{_layer}** has only {_limit} channels."))
    with torch.enable_grad():
        _g = torch.Generator().manual_seed(0)
        _x = (torch.randn(1, 3, 240, 240, generator=_g) * 0.1).requires_grad_(True)
        _opt = torch.optim.Adam([_x], lr=0.05)
        _blur = torchvision.transforms.GaussianBlur(3, sigma=0.6)
        for _s in mo.status.progress_bar(range(vis_steps.value), title="gradient ascent on the input"):
            _dx, _dy = torch.randint(0, 16, (2,), generator=_g).tolist()
            _crop = _x[:, :, _dy:_dy + 224, _dx:_dx + 224]
            if _layer == "fc8":
                _out = net(_crop)[0, _ch]
            else:
                _a = net.features[:AFTER[_layer]](_crop)[0, _ch]
                _h, _w = _a.shape
                _out = _a.mean() if _layer in ("conv1", "conv2") else _a[_h // 2 - 2:_h // 2 + 3, _w // 2 - 2:_w // 2 + 3].mean()
            _loss = -_out + 1e-3 * (_x ** 2).mean()
            _opt.zero_grad(); _loss.backward(); _opt.step()
            if _s % 8 == 0 and _s < vis_steps.value - 16:
                _x.data = _blur(_x.data)
    _img = _x.detach()[0, :, 8:232, 8:232] * STD + MEAN
    _lo, _hi = torch.quantile(_img, 0.01), torch.quantile(_img, 0.99)
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _ax.imshow(((_img - _lo) / (_hi - _lo)).clamp(0, 1).permute(1, 2, 0).numpy())
    _ax.set_title(f"{_layer}, {'class' if _layer == 'fc8' else 'channel'} {_ch}", loc="left"); _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The embedding space

    Stop one layer before the classifier and keep the 4096 numbers of `fc7`. Every image becomes a point
    in a 4096-dimensional space. Flipped and cropped copies of an image have very different pixels but
    land close together; similar concepts cluster. The AlexNet paper found that nearest neighbours in
    this space show the same kind of object.
    """)
    return


@app.cell
def _(
    Image,
    image_name,
    net,
    norm,
    np,
    plt,
    samples,
    skd,
    to_tensor,
    torch,
    torchvision,
    x01,
):
    def embed(xs):
        _f = net.avgpool(net.features(torch.cat([norm(_x) for _x in xs]))).flatten(1)
        return net.classifier[:6](_f)                   # fc7 after ReLU: 4096 numbers

    _names, _xs = [], []
    for _n in samples:
        _x = to_tensor(Image.fromarray(getattr(skd, _n)()))
        _crop = torchvision.transforms.functional.resized_crop(_x, 30, 20, 160, 160, [224, 224], antialias=True)
        _xs += [_x, torch.flip(_x, (2,)), _crop]
        _names += [_n, _n + " (flip)", _n + " (crop)"]
    _xs.append(x01); _names.append(f"YOUR IMAGE: {image_name}")
    _X = torch.stack(_xs)

    def _cos(v):
        v = v - v.mean(0, keepdim=True)
        v = v / v.norm(dim=1, keepdim=True)
        return (v @ v.T).numpy()

    _pix, _emb = _cos(_X.flatten(1)), _cos(embed(_xs))
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for _ax, _M, _t in [(_axes[0], _pix, "raw pixels"), (_axes[1], _emb, "fc7 embedding")]:
        _ax.imshow(_M, cmap="RdBu_r", vmin=-1, vmax=1)
        _ax.set_title(f"cosine similarity: {_t}", loc="left")
        _ax.set_xticks(range(len(_names)), _names, rotation=90, fontsize=6)
        _ax.set_yticks(range(len(_names)), _names, fontsize=6)
    _fig.tight_layout()
    _order = [i for i in np.argsort(-_emb[-1]) if i != len(_names) - 1][:3]
    _near = ", ".join(f"{_names[i]} ({_emb[-1, i]:.2f})" for i in _order)
    print(f"Nearest neighbours of your image in the embedding: {_near}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Find a first-layer kernel that responds to a colour rather than an edge. Which images light up its map?
    2. Photograph a printed first-layer kernel (or draw its stripes) and upload the photo. Does its map light up?
       What happens when you rotate the paper by 90°?
    3. Pick a conv5 channel and find, among your own photos, the one that activates it most. Can you name the concept?
    4. Occlude the face in `astronaut` using the occlusion experiment. Does the prediction depend on the face?
    5. Why is it easy to visualize layer-1 kernels as images, but not layer-2 kernels (5 × 5 × 64)?
    6. AlexNet has about 61 million parameters. Count how many belong to the convolutional layers and how many
       to the three dense layers. Where is most of the memory, and where most of the computation?
    """)
    return


if __name__ == "__main__":
    app.run()
