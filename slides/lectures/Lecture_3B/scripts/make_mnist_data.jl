# Train the CNN of pluto/mnist-draw.jl for three epochs and export what the browser
# widgets widgets/mnist.html and widgets/mnist-shift.html need for inference:
#   - the weights after 0, 1, 2 and 3 epochs,
#   - test accuracy and accuracy against a shift to the right, per checkpoint,
#   - the same shift curve for the dense 784–128–10 network (3 epochs), for comparison,
#   - a few test digits for the shift widget.
# Writes widgets/data_mnist.js, plus scripts/mnist_reference.json for the JS check
# (scripts/check_mnist_widget.mjs).
#
#   julia --project=slides/lectures/Lecture_3B/pluto slides/lectures/Lecture_3B/scripts/make_mnist_data.jl
#
# MNIST is read from ~/.cache/mnist/MNIST/raw, as in the notebook.
using Flux, Random, Statistics, Base64

# Minimal JSON writer, so the script needs nothing beyond the notebook environment.
json(io, x::Union{NamedTuple,AbstractDict}) = (print(io, '{'); join_json(io, pairs(x)) do io, (k, v)
    print(io, '"', k, "\":"); json(io, v) end; print(io, '}'))
json(io, x::Union{AbstractVector,Tuple}) = (print(io, '['); join_json(io, x) do io, v; json(io, v) end; print(io, ']'))
json(io, x::AbstractString) = print(io, '"', escape_string(x), '"')
json(io, x::Integer) = print(io, x)
json(io, x::Real) = print(io, round(Float64(x); sigdigits=7))
function join_json(f, io, xs)
    for (i, x) in enumerate(xs)
        i > 1 && print(io, ',')
        f(io, x)
    end
end

const MNIST_RAW = expanduser("~/.cache/mnist/MNIST/raw")
const HERE = @__DIR__
const WIDGETS = joinpath(HERE, "..", "widgets")

# X[x, y, 1, n]: column x from the left, row y from the top (as in the notebook).
function mnist_images(split)
    b = read(joinpath(MNIST_RAW, split == :train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte"))
    n = Int(ntoh(reinterpret(UInt32, b[5:8])[1]))
    reshape(Float32.(reshape(b[17:end], 28, 28, n)) ./ 255, 28, 28, 1, n)
end
mnist_labels(split) =
    Int.(read(joinpath(MNIST_RAW, split == :train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte"))[9:end])

function shift_images(X, dx)
    S = zero(X)
    for x in 1:28
        1 <= x - dx <= 28 && (S[x, :, :, :] .= @view X[x-dx, :, :, :])
    end
    S
end

X_train, y_train = mnist_images(:train), mnist_labels(:train)
X_test, y_test = mnist_images(:test), mnist_labels(:test)
accuracy(m, X, y) = mean(Flux.onecold(m(X), 0:9) .== y)
const SHIFTS = 0:6
const NSHIFT = 2000
shift_curve(m) = [round(accuracy(m, shift_images(X_test[:, :, :, 1:NSHIFT], s), y_test[1:NSHIFT]); digits=4) for s in SHIFTS]

function train_epoch!(model, opt, rng)
    loader = Flux.DataLoader((X_train, Flux.onehotbatch(y_train, 0:9)); batchsize=128, shuffle=true, rng)
    for (xb, yb) in loader
        g = gradient(m -> Flux.logitcrossentropy(m(xb), yb), model)
        Flux.update!(opt, model, g[1])
    end
end

# Weights in the order the JS forward pass reads them. Flux's Conv is a true
# convolution (flipped kernel); flipping here lets JS use a plain sliding window.
#   conv1 w[co][ky][kx] (8×3×3), b[8]; conv2 w[co][ci][ky][kx] (16×8×3×3), b[16];
#   dense w[out][in] (10×400) with in = c·25 + y·5 + x, b[10].
function export_weights(model)
    c1, c2, d = model[1], model[3], model[6]
    flipk(W) = W[end:-1:1, end:-1:1, :, :]
    order(W) = vec(flipk(W))  # [kx, ky, ci, co] column-major = [co][ci][ky][kx] row-major
    v = Float32[order(c1.weight); c1.bias; order(c2.weight); c2.bias; vec(permutedims(d.weight)); d.bias]
    base64encode(reinterpret(UInt8, v))
end

Random.seed!(2026)
rng = Xoshiro(2026)
cnn = Chain(Conv((3, 3), 1 => 8, relu), MaxPool((2, 2)),
            Conv((3, 3), 8 => 16, relu), MaxPool((2, 2)),
            Flux.flatten, Dense(400 => 10))
opt = Flux.setup(Adam(1e-3), cnn)

checkpoints = []
reference = []
ref_idx = 1:5
for epoch in 0:3
    epoch > 0 && train_epoch!(cnn, opt, rng)
    acc = round(accuracy(cnn, X_test, y_test); digits=4)
    curve = shift_curve(cnn)
    println("CNN epoch $epoch: test accuracy $acc, shift curve $curve")
    push!(checkpoints, (epoch=epoch, acc=acc, shift=curve, w=export_weights(cnn)))
    push!(reference, cnn(X_test[:, :, :, ref_idx]) |> permutedims |> eachrow .|> collect)
end

dense = Chain(Flux.flatten, Dense(784 => 128, relu), Dense(128 => 10))
opt_d = Flux.setup(Adam(1e-3), dense)
for _ in 1:3
    train_epoch!(dense, opt_d, rng)
end
dense_acc = round(accuracy(dense, X_test, y_test); digits=4)
dense_curve = shift_curve(dense)
println("Dense after 3 epochs: test accuracy $dense_acc, shift curve $dense_curve")

# Test digits for the shift widget: the first three of each class, row-major bytes (y·28 + x).
picks = vcat([findall(==(d), y_test)[1:3] for d in 0:9]...)
sort!(picks)
digits_bytes = UInt8[]
for i in picks
    append!(digits_bytes, round.(UInt8, vec(X_test[:, :, 1, i]) .* 255))  # column-major [x, y] = row-major (y, x)
end

data = (
    arch = "conv 3×3 (8) + ReLU → max-pool 2×2 → conv 3×3 (16) + ReLU → max-pool 2×2 → dense 400 → 10",
    params = sum(length, Flux.trainables(cnn)),
    shifts = collect(SHIFTS), nShift = NSHIFT, nTest = length(y_test),
    checkpoints = checkpoints,
    dense = (acc=dense_acc, shift=dense_curve, params=sum(length, Flux.trainables(dense))),
    digits = base64encode(digits_bytes), labels = y_test[picks],
)
open(joinpath(WIDGETS, "data_mnist.js"), "w") do io
    print(io, "// Generated by scripts/make_mnist_data.jl — do not edit.\nwindow.DATA_MNIST=")
    json(io, data)
    println(io, ";")
end
ref_pixels = [vec(X_test[:, :, 1, i]) for i in ref_idx]  # row-major (y, x)
open(joinpath(HERE, "mnist_reference.json"), "w") do io
    json(io, (pixels=ref_pixels, logits=reference))
end
println("wrote widgets/data_mnist.js ($(filesize(joinpath(WIDGETS, "data_mnist.js"))) bytes)")
