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

# ╔═╡ 2cc7df0e-90e4-4908-8cef-2b13150441f1
begin
	import Pkg
	Pkg.activate(@__DIR__)   # the environment next to this notebook (JLBoost from JuliaHEP, ...)
	using JLBoost, DataFrames, Plots, PlutoUI, Statistics, Serialization
	using HypertextLiteral, AbstractPlutoDingetjes
	theme(:boxed)
end

# ╔═╡ c0a1e893-aadb-4cef-b1e3-84fbed907a45
md"""
# Lecture 3B: can trees read digits?

Gradient-boosted trees (Lecture 2B) on raw MNIST pixels, with
[JLBoost.jl](https://github.com/JuliaHEP/JLBoost.jl). Every pixel is one feature
`x…_y…`; a tree asks questions like *"is pixel (15, 12) brighter than 0.3?"*.
JLBoost fits one binary classifier per digit ("is it a 3?"), so ten boosters vote.
"""

# ╔═╡ 63ad7575-ce23-4544-988c-bc1a00b55e13
md"""
## Train ten boosters

Settings: the first **N = 10 000** training images, **20 rounds**, trees of **depth 4**,
learning rate 0.3. The ten boosters train in parallel (about 2 minutes on 8 cores);
the result is cached in `cache/`, so the notebook opens instantly.
"""

# ╔═╡ 1921c297-82f9-4a49-81f6-f5f28bd99774
const N, ROUNDS, DEPTH = 10_000, 20, 4

# ╔═╡ 68e78339-cc61-442a-a074-7cd850da1dcd
@bind retrain_clicks CounterButton("Retrain (about 2 minutes)")

# ╔═╡ 5b3f0c7e-2c3e-4b8e-9d0a-4f1e2a6c9d02
md"""
## Where do the trees look?

Total split gain per pixel, summed over all trees: bright pixels are the ones the
trees ask about. Choose a digit to see its booster only.
"""

# ╔═╡ 7a2c9e4d-1b5f-4c8a-8e3d-6f0b9a1c2d03
@bind which_digit Select(["all ten", string.(0:9)...])

# ╔═╡ 8e4d0b6f-3c7a-4d9b-a1f2-7b3c5d8e9f04
begin
	pixel_xy(f) = parse.(Int, match(r"x(\d+)_y(\d+)", string(f)).captures)
	function add_gain!(G, node)
		if !isempty(node.children) && !ismissing(node.splitfeature)
			x, y = pixel_xy(node.splitfeature)
			G[x, y] += node.gain
			foreach(c -> add_gain!(G, c), node.children)
		end
		G
	end
	gain_map(ms) = (G = zeros(28, 28); for m in ms, t in trees(m); add_gain!(G, t); end; G)
end;

# ╔═╡ a06f2d8b-5e9c-4fbd-83b4-9d5e7fa0b106
md"""
## The first tree of the "is it a 0?" booster
"""

# ╔═╡ c28b4fad-7b0e-4b1f-a5d6-bf7091c2d308
md"""
## Draw a digit
"""

# ╔═╡ f5be7cd0-ae3b-4e4c-98a9-e2a3c4f5a60b
md"""
## Shift the test digits

A tree asks about fixed pixel positions. Move every test image a few pixels to the right:
"""

# ╔═╡ edd98d54-4d7d-44ba-966f-84a031bffc32
md"""
## Appendix: helpers

MNIST from the local raw files (downloaded by torchvision for `scripts/make_figures.py`, so nothing is downloaded during the lecture), image shifting, and the drawing pad.
"""

# ╔═╡ b703919c-ea62-4df3-b5c8-6a3afe2ea814
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

# ╔═╡ da73ce72-ee13-48a2-8dc8-4ac48912b66a
begin
	X_train, y_train = mnist_images(:train), mnist_labels(:train)
	X_test,  y_test  = mnist_images(:test),  mnist_labels(:test)
	# Column k = x + 28(y - 1) of the table is pixel (x, y), as in vec(image).
	const PIXELS = [Symbol("x$(x)_y$(y)") for y in 1:28 for x in 1:28]
	to_table(X) = DataFrame(permutedims(reshape(X, 784, :)), PIXELS)
	md"MNIST: $(length(y_train)) training and $(length(y_test)) test images, read from the local raw files."
end

# ╔═╡ 9dc0a96b-deaa-47ee-8429-895723deec20
boosters = let clicks = retrain_clicks
	file = joinpath(@__DIR__, "cache", "jlboost-mnist-n$N-r$ROUNDS-d$DEPTH.jls")
	if clicks == 0 && isfile(file)
		deserialize(file)
	else
		df = to_table(X_train[:, :, 1:N])
		ms = fetch.([Threads.@spawn begin
			dk = copy(df)
			dk.y = Int.(y_train[1:N] .== digit)          # one versus rest
			jlboost(dk, :y; nrounds=ROUNDS, max_depth=DEPTH, eta=0.3)
		end for digit in 0:9])
		mkpath(dirname(file)); serialize(file, ms)
		ms
	end
end;

# ╔═╡ 0c0d66e2-8002-4c6d-8132-b8115e1fb946
"Margins of the ten boosters, one column per digit; the prediction is the largest."
margins(tbl) = reduce(hcat, [predict(m, tbl) for m in boosters])

# ╔═╡ b17a3e9c-6fad-4a0e-94c5-ae6f80b1c207
let
	function lines(node, depth=0)
		pad = "    "^depth
		if isempty(node.children) || ismissing(node.splitfeature)
			return ["$(pad)→ score $(round(node.weight, digits=2))"]
		end
		x, y = pixel_xy(node.splitfeature)
		["$(pad)pixel ($x, $y) ≤ $(round(node.split, digits=2)) ?";
		 lines(node.children[1], depth + 1);
		 "$(pad)else";
		 lines(node.children[2], depth + 1)]
	end
	Text(join(lines(trees(boosters[1])[1].tree), "\n"))
end

# ╔═╡ d9d1a1f0-6d3c-4b3a-9a57-0f6a1b7c2e01
begin
	test_table = to_table(X_test)
	test_pred = getindex.(argmax(margins(test_table); dims=2)[:], 2) .- 1
	md"""
	### Test accuracy: **$(round(100mean(test_pred .== y_test), digits=1)) %**

	In total $(sum(length ∘ trees, boosters)) trees with at most $(2^DEPTH) leaves each, trained on $N images.
	Compare the MNIST table on the slide: random forest 2.8 % error, a dense network 1.6 %.
	"""
end

# ╔═╡ 10d29be6-75bf-4629-9f2a-fa89f71b7c2b
"Shift images by `dx` columns (right) and `dy` rows (down), filling with zeros."
function shift_images(X::AbstractArray{T,3}, dx::Int, dy::Int) where T
    S = zeros(T, size(X))
    for x in 1:28, y in 1:28
        (1 <= x - dx <= 28 && 1 <= y - dy <= 28) && (S[x, y, :] .= @view X[x-dx, y-dy, :])
    end
    S
end

# ╔═╡ a6cf8de1-bf4c-4f5d-a9ba-f3b4d5a6b70c
let
	shifts = 0:6
	acc = map(shifts) do s
		pred = getindex.(argmax(margins(to_table(shift_images(X_test[:, :, 1:2000], s, 0))); dims=2)[:], 2) .- 1
		mean(pred .== y_test[1:2000])
	end
	plot(shifts, acc; m=:o, lw=3, ylims=(0, 1), xlabel="shift to the right (pixels)",
		ylabel="test accuracy", label="boosted trees", size=(620, 260))
	hline!([0.1]; ls=:dash, lc=:gray, label="guessing")
end

# ╔═╡ 36037215-e940-436d-936e-482b1daa4d02
"Show one 28 × 28 image `A[x, y]` the way it looks."
digit_heatmap(A; kw...) = heatmap(permutedims(A); yflip=true, c=:grays, aspect_ratio=1,
    axis=nothing, border=:none, colorbar=false, clims=(0, 1), kw...)

# ╔═╡ 9f5e1c7a-4d8b-4eac-b2a3-8c4d6e9fa005
let
	ms = which_digit == "all ten" ? boosters : [boosters[parse(Int, which_digit) + 1]]
	G = gain_map(ms)
	avg = which_digit == "all ten" ? dropdims(mean(X_train[:, :, 1:N]; dims=3); dims=3) :
		dropdims(mean(X_train[:, :, findall(==(parse(Int, which_digit)), y_train[1:N])]; dims=3); dims=3)
	plot(digit_heatmap(avg; title="average training image", titlefontsize=10),
		heatmap(permutedims(G); yflip=true, c=:inferno, aspect_ratio=1, axis=nothing,
			border=:none, colorbar=false, title="split gain per pixel", titlefontsize=10);
		layout=(1, 2), size=(620, 300))
end

# ╔═╡ b05bca55-02f7-4b55-b2b5-9dae252b5db4
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

# ╔═╡ d39c5abe-8c1f-4c2a-b6e7-c081a2d3e409
@bind pixels DrawPad()

# ╔═╡ e4ad6bcf-9d2a-4d3b-87f8-d192b3e4f50a
let
	A = drawn_image(pixels)
	s = margins(to_table(reshape(A, 28, 28, 1)))[:]
	p = exp.(s .- maximum(s)); p ./= sum(p)
	k = argmax(p)
	plot(digit_heatmap(A; title="what the trees see (28 × 28)", titlefontsize=10),
		bar(0:9, p; xticks=0:9, ylims=(0, 1), label=false, c=[i == k ? 2 : 1 for i in 1:10],
			title=sum(A) == 0 ? "draw a digit" : "prediction: $(k - 1)", titlefontsize=12);
		layout=grid(1, 2, widths=(0.4, 0.6)), size=(620, 260))
end

# ╔═╡ Cell order:
# ╟─c0a1e893-aadb-4cef-b1e3-84fbed907a45
# ╠═2cc7df0e-90e4-4908-8cef-2b13150441f1
# ╠═da73ce72-ee13-48a2-8dc8-4ac48912b66a
# ╟─63ad7575-ce23-4544-988c-bc1a00b55e13
# ╠═1921c297-82f9-4a49-81f6-f5f28bd99774
# ╟─68e78339-cc61-442a-a074-7cd850da1dcd
# ╠═9dc0a96b-deaa-47ee-8429-895723deec20
# ╠═0c0d66e2-8002-4c6d-8132-b8115e1fb946
# ╟─d9d1a1f0-6d3c-4b3a-9a57-0f6a1b7c2e01
# ╟─5b3f0c7e-2c3e-4b8e-9d0a-4f1e2a6c9d02
# ╟─7a2c9e4d-1b5f-4c8a-8e3d-6f0b9a1c2d03
# ╠═8e4d0b6f-3c7a-4d9b-a1f2-7b3c5d8e9f04
# ╠═9f5e1c7a-4d8b-4eac-b2a3-8c4d6e9fa005
# ╟─a06f2d8b-5e9c-4fbd-83b4-9d5e7fa0b106
# ╠═b17a3e9c-6fad-4a0e-94c5-ae6f80b1c207
# ╟─c28b4fad-7b0e-4b1f-a5d6-bf7091c2d308
# ╠═d39c5abe-8c1f-4c2a-b6e7-c081a2d3e409
# ╠═e4ad6bcf-9d2a-4d3b-87f8-d192b3e4f50a
# ╟─f5be7cd0-ae3b-4e4c-98a9-e2a3c4f5a60b
# ╠═a6cf8de1-bf4c-4f5d-a9ba-f3b4d5a6b70c
# ╟─edd98d54-4d7d-44ba-966f-84a031bffc32
# ╠═b703919c-ea62-4df3-b5c8-6a3afe2ea814
# ╠═10d29be6-75bf-4629-9f2a-fa89f71b7c2b
# ╠═36037215-e940-436d-936e-482b1daa4d02
# ╠═b05bca55-02f7-4b55-b2b5-9dae252b5db4
