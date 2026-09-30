### A Pluto.jl notebook ###
# v0.20.28

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 97c2bab3-0102-4ac2-b860-8c32a2309272
begin
	import Pkg
	Pkg.activate(@__DIR__)   # the environment next to this notebook (Flux, Plots, PlutoUI, ...)
	using Flux, Plots, PlutoUI, Statistics
	using HypertextLiteral, AbstractPlutoDingetjes
	theme(:boxed)
end

# ╔═╡ d9ab0949-aa4f-4db1-a2ff-2c17856c0123
md"""
# Lecture 3B: train a digit classifier, then draw for it

Train a dense network or a small CNN on MNIST with the button (one epoch per click,
seconds on a laptop CPU), then draw a digit with the mouse and watch the prediction.
"""

# ╔═╡ c0ed263d-ec5b-4a8b-9d5b-d227704f91c5
md"""
## Train
"""

# ╔═╡ 4fbdcb67-a78b-4426-910c-830f39cc1cb3
md"""
Model $(@bind arch Select(["Dense 784–128–10", "CNN: 2 × (conv 3×3 + pool), dense"]))
$(@bind train_click Button("Train one epoch"))
"""

# ╔═╡ 3748bc59-f628-4233-8748-8724bf51f353
net = let
	model = if startswith(arch, "Dense")
		Chain(Flux.flatten, Dense(784 => 128, relu), Dense(128 => 10))
	else
		Chain(Conv((3, 3), 1 => 8, relu), MaxPool((2, 2)),
		      Conv((3, 3), 8 => 16, relu), MaxPool((2, 2)),
		      Flux.flatten, Dense(400 => 10))
	end
	# `started` is set on the first run of the training cell, so a new model starts untrained.
	(; model, opt = Flux.setup(Adam(1e-3), model), acc = Float64[], started = Ref(false))
end;

# ╔═╡ 71c360bb-83d5-4cc5-ae86-31b5f9859b50
md"""
## Draw a digit
"""

# ╔═╡ 275f35c1-d50e-4016-a034-5b006aeb59e8
md"""
## Shift the test digits

MNIST digits are centred. Move every test image by a few pixels to the right and
measure the accuracy again.
"""

# ╔═╡ ee9d1d1f-a4d5-40e0-a3af-1267e0e6e2bb
# One curve per model, kept across model switches, so dense and CNN can be compared.
shift_curves = Dict{String, Vector{Float64}}();

# ╔═╡ 0b5dbc8a-9977-464a-9b67-4c45f3a38684
md"""
## Appendix: helpers

MNIST from the local raw files (downloaded by torchvision for `scripts/make_figures.py`, so nothing is downloaded during the lecture), image shifting, and the drawing pad.
"""

# ╔═╡ b40476bc-8b8e-40e8-854c-ec8c3a790477
begin
	# The raw IDX files, as downloaded by torchvision for Lecture_3B/scripts/make_figures.py.
	const MNIST_RAW = expanduser("~/.cache/mnist/MNIST/raw")

	"""
	    mnist_images(split) -> Array{Float32,3}   # 28 × 28 × n, values in [0, 1]

	Indexed as `X[x, y, n]`: column `x` from the left, row `y` from the top.
	"""
	function mnist_images(split::Symbol)
	    b = read(joinpath(MNIST_RAW, split == :train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte"))
	    n = Int(ntoh(reinterpret(UInt32, b[5:8])[1]))
	    Float32.(reshape(b[17:end], 28, 28, n)) ./ 255
	end

	mnist_labels(split::Symbol) =
	    Int.(read(joinpath(MNIST_RAW, split == :train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte"))[9:end])
end

# ╔═╡ 16fdcb8c-c873-4be4-b1bc-cd029e51f887
begin
	X_train = reshape(mnist_images(:train), 28, 28, 1, :)
	y_train = mnist_labels(:train)
	X_test  = reshape(mnist_images(:test), 28, 28, 1, :)
	y_test  = mnist_labels(:test)
	loader = Flux.DataLoader((X_train, Flux.onehotbatch(y_train, 0:9)); batchsize=128, shuffle=true)
	md"MNIST: $(length(y_train)) training and $(length(y_test)) test images, read from the local raw files."
end

# ╔═╡ 39503fa6-727c-49dd-bb78-1a3c2b10fe7d
test_accuracy(m, X = X_test, y = y_test) = mean(Flux.onecold(m(X), 0:9) .== y)

# ╔═╡ bf7b58cd-7f44-4cdc-8da3-3e5d54c47dba
# Every click reruns this cell (a Button sends the same value each time, so the
# slide iframes, which are recreated on every visit, carry no click count).
epochs = let _ = train_click
	if !net.started[]
		net.started[] = true                    # first run for this model: do not train
	else
		for (xb, yb) in loader                  # one epoch: 469 mini-batches of 128
			g = Flux.gradient(m -> Flux.logitcrossentropy(m(xb), yb), net.model)
			Flux.update!(net.opt, net.model, g[1])
		end
		push!(net.acc, test_accuracy(net.model))
	end
	length(net.acc)
end;

# ╔═╡ 4262e0ff-4e84-4190-83ac-a7fe0024a1f5
let epochs = epochs   # rerun after training
	n = Flux.trainables(net.model) .|> length |> sum
	a = isempty(net.acc) ? test_accuracy(net.model) : last(net.acc)
	plot(0:epochs, [0.1; net.acc]; m=:o, lw=3, ylims=(0, 1), xlims=(0, max(epochs, 5)),
		xlabel="epoch", ylabel="test accuracy", label=false, size=(620, 260),
		title="$(split(arch, ':')[1]), $n parameters: $(round(100a, digits=1)) % after $epochs epochs",
		titlefontsize=11)
	hline!([0.1]; ls=:dash, lc=:gray, label="guessing")
end

# ╔═╡ 8d2135ae-4ce0-40ac-b663-bbf6d627ab82
"Shift images by `dx` columns (right) and `dy` rows (down), filling with zeros."
function shift_images(X::AbstractArray{T,3}, dx::Int, dy::Int) where T
    S = zeros(T, size(X))
    for x in 1:28, y in 1:28
        (1 <= x - dx <= 28 && 1 <= y - dy <= 28) && (S[x, y, :] .= @view X[x-dx, y-dy, :])
    end
    S
end

# ╔═╡ f8dd5fdd-7e21-4478-acbd-9b7cf57047b2
let epochs = epochs   # rerun after training
	shifts = 0:6
	Xs = X_test[:, :, 1, 1:2000]
	name = split(arch, ':')[1]
	if epochs > 0
		shift_curves[name] = [test_accuracy(net.model, reshape(shift_images(Xs, s, 0), 28, 28, 1, :), y_test[1:2000]) for s in shifts]
	end
	p = plot(xlabel="shift to the right (pixels)", ylabel="test accuracy", ylims=(0, 1),
		xticks=shifts, xlims=(-0.3, 6.3), size=(620, 300), legend=:topright)
	for (k, (n, acc)) in enumerate(sort(collect(shift_curves)))
		plot!(shifts, acc; m=:o, lw=3, c=k, label=n)
	end
	hline!([0.1]; ls=:dash, lc=:gray, label="guessing")
	isempty(shift_curves) && annotate!(3, 0.5, text("train a model first", 11, :gray))
	p
end

# ╔═╡ 3680db7c-76e4-4b8b-9fe9-d70246a77545
"Show one 28 × 28 image `A[x, y]` the way it looks."
digit_heatmap(A; kw...) = heatmap(permutedims(A); yflip=true, c=:grays, aspect_ratio=1,
    axis=nothing, border=:none, colorbar=false, clims=(0, 1), kw...)

# ╔═╡ 1a84b86d-95a9-43c2-bfa4-9c0d6d05a734
let epochs = epochs   # rerun after training
	idx = 1:12
	pred = Flux.onecold(net.model(X_test[:, :, :, idx]), 0:9)
	plot([digit_heatmap(X_test[:, :, 1, i]; title="$(pred[j])", titlefontsize=12,
		titlefontcolor = pred[j] == y_test[i] ? :black : :red) for (j, i) in enumerate(idx)]...;
		layout=(2, 6), size=(620, 250))
end

# ╔═╡ 22f8118c-d22a-40a4-b958-dc71725cebfb
begin
	"""
	    @bind pixels DrawPad()

	A 280 × 280 canvas. On every pen-up, the drawing is cropped, scaled into a
	20 × 20 box and centred by its centre of mass in 28 × 28, as MNIST digits are,
	and sent to Julia as 784 numbers in [0, 1] (reshape to 28 × 28, indexed [x, y]).
	"""
	struct DrawPad end

	Base.show(io::IO, m::MIME"text/html", ::DrawPad) = show(io, m, @htl("""
	<span class="drawpad" style="display:inline-flex;flex-direction:column;gap:6px;align-items:flex-start">
	<canvas width="280" height="280" style="background:#000;border-radius:6px;touch-action:none;cursor:crosshair"></canvas>
	<button>clear</button>
	<script>
	const span = currentScript.parentElement
	const canvas = span.querySelector("canvas")
	const ctx = canvas.getContext("2d")
	ctx.lineWidth = 22; ctx.lineCap = "round"; ctx.lineJoin = "round"
	ctx.strokeStyle = ctx.fillStyle = "#fff"
	let drawing = false, last = null

	const pos = (e) => {
	    const r = canvas.getBoundingClientRect()
	    return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height]
	}
	canvas.onpointerdown = (e) => {
	    drawing = true; last = pos(e)
	    try { canvas.setPointerCapture(e.pointerId) } catch (_) {}
	    ctx.beginPath(); ctx.arc(last[0], last[1], 11, 0, 2 * Math.PI); ctx.fill()
	}
	canvas.onpointermove = (e) => {
	    if (!drawing) return
	    const p = pos(e)
	    ctx.beginPath(); ctx.moveTo(...last); ctx.lineTo(...p); ctx.stroke(); last = p
	}
	canvas.onpointerup = canvas.onpointercancel = () => { if (drawing) { drawing = false; send() } }
	span.querySelector("button").onclick = () => { ctx.clearRect(0, 0, 280, 280); send() }

	// MNIST preprocessing: bounding box -> 20 × 20, then centre of mass -> (13.5, 13.5).
	function to28() {
	    const W = 280, img = ctx.getImageData(0, 0, W, W).data
	    let x0 = W, y0 = W, x1 = -1, y1 = -1
	    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if (img[4 * (y * W + x) + 3] > 20) {
	        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y)
	    }
	    if (x1 < 0) return new Array(784).fill(0)
	    const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h)
	    const t = document.createElement("canvas"); t.width = t.height = 28
	    const tc = t.getContext("2d"); tc.imageSmoothingQuality = "high"
	    tc.drawImage(canvas, x0, y0, w, h, 14 - w * s / 2, 14 - h * s / 2, w * s, h * s)
	    let d = tc.getImageData(0, 0, 28, 28).data, m = 0, cx = 0, cy = 0
	    for (let i = 0; i < 784; i++) { const v = d[4 * i + 3]; m += v; cx += v * (i % 28); cy += v * Math.floor(i / 28) }
	    const f = document.createElement("canvas"); f.width = f.height = 28
	    f.getContext("2d").drawImage(t, Math.round(13.5 - cx / m), Math.round(13.5 - cy / m))
	    d = f.getContext("2d").getImageData(0, 0, 28, 28).data
	    return Array.from({ length: 784 }, (_, i) => d[4 * i + 3] / 255)
	}
	function send() { span.value = to28(); span.dispatchEvent(new CustomEvent("input")) }
	span.value = new Array(784).fill(0)
	</script>
	</span>
	"""))

	AbstractPlutoDingetjes.Bonds.initial_value(::DrawPad) = zeros(784)

	"The drawing as a 28 × 28 Float32 image, indexed [x, y]."
	drawn_image(pixels) = reshape(Float32.(pixels), 28, 28)
end

# ╔═╡ 351a5dea-2b79-4b53-beac-154503b52c34
@bind pixels DrawPad()

# ╔═╡ d629e83e-b1cf-46cd-a045-e25dbf0b6462
let epochs = epochs   # rerun after training
	A = drawn_image(pixels)
	p = softmax(net.model(reshape(A, 28, 28, 1, 1)))[:]
	k = argmax(p)
	plot(digit_heatmap(A; title="what the network sees (28 × 28)", titlefontsize=10),
		bar(0:9, p; xticks=0:9, ylims=(0, 1), label=false, c=[i == k ? 2 : 1 for i in 1:10],
			title=sum(A) == 0 ? "draw a digit" : "prediction: $(k - 1)", titlefontsize=12);
		layout=grid(1, 2, widths=(0.4, 0.6)), size=(620, 260))
end

# ╔═╡ Cell order:
# ╟─d9ab0949-aa4f-4db1-a2ff-2c17856c0123
# ╠═97c2bab3-0102-4ac2-b860-8c32a2309272
# ╠═16fdcb8c-c873-4be4-b1bc-cd029e51f887
# ╟─c0ed263d-ec5b-4a8b-9d5b-d227704f91c5
# ╟─4fbdcb67-a78b-4426-910c-830f39cc1cb3
# ╠═3748bc59-f628-4233-8748-8724bf51f353
# ╠═39503fa6-727c-49dd-bb78-1a3c2b10fe7d
# ╠═bf7b58cd-7f44-4cdc-8da3-3e5d54c47dba
# ╠═4262e0ff-4e84-4190-83ac-a7fe0024a1f5
# ╟─71c360bb-83d5-4cc5-ae86-31b5f9859b50
# ╠═351a5dea-2b79-4b53-beac-154503b52c34
# ╠═d629e83e-b1cf-46cd-a045-e25dbf0b6462
# ╟─275f35c1-d50e-4016-a034-5b006aeb59e8
# ╠═ee9d1d1f-a4d5-40e0-a3af-1267e0e6e2bb
# ╠═f8dd5fdd-7e21-4478-acbd-9b7cf57047b2
# ╠═1a84b86d-95a9-43c2-bfa4-9c0d6d05a734
# ╟─0b5dbc8a-9977-464a-9b67-4c45f3a38684
# ╠═b40476bc-8b8e-40e8-854c-ec8c3a790477
# ╠═8d2135ae-4ce0-40ac-b663-bbf6d627ab82
# ╠═3680db7c-76e4-4b8b-9fe9-d70246a77545
# ╠═22f8118c-d22a-40a4-b958-dc71725cebfb
