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

# ╔═╡ b89c9e0a-6d28-11f0-0e0c-b723e78bfeca
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true
	# 
	using Flux, MLDatasets, FileIO # CUDA
	using Flux: train!, onehotbatch
	using PlutoUI
	using Plots
	using PlutoTeachingTools
	"import packages"
end

# ╔═╡ 7bb1811b-8671-4528-beaa-7f51705065aa
md"""
# Lecture 3: MNIST Classification
"""

# ╔═╡ bf13d5a2-35a5-4ce4-a664-2c9bec5800d0
begin
	theme(:boxed)
	"color settings"
end

# ╔═╡ 40a4a3f7-fe5d-4200-9862-3630a0ae876a
begin
	x_train, y_train = MNIST(split=:train)[:]
	x_test, y_test = MNIST(split=:test)[:]
	# 
	x_train = Float32.(x_train)
	y_train = Flux.onehotbatch(y_train, 0:9)
	# 
	train_data = [(Flux.flatten(x_train), y_train)]
	test_data = [(Flux.flatten(x_test), y_test)]
	# 
	"download the data"
end

# ╔═╡ bb1b1a04-75da-4dcb-a2e5-2a21524771e5
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10, relu), softmax
)

# ╔═╡ 8cd2f2ad-e636-4c6d-8f41-abdc5f8fc881
md"""
## Training loop
"""

# ╔═╡ f72101b5-b486-4850-9fcd-1261925d141b
md"""
1. Give permission to run the train loop $(@bind permit_training CheckBox(default=false)) -- nEpoch=400 will take about 10'
2. Press $(@bind start_train Button("Train")) to rerun the trainig loop
"""

# ╔═╡ b34c5cd0-20bc-40ef-bba7-5bd8b132bc60
const nEpochs = 400 

# ╔═╡ e19e619d-eb34-4244-b258-5548c5059ab6
if permit_training
	start_train
	# 
	opt = Flux.setup(Adam(0.0001), model)
	loss(m,x, y) = Flux.Losses.logitcrossentropy(m(x), y)
	for i in 1:nTrain
	    Flux.train!(loss, model, train_data, opt)
	end
end

# ╔═╡ 7ed89ca8-ac55-4e91-b8f1-7ad2daf0783e
let
	accuracy = 0
	for i in 1:length(y_test)
	    if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
	        accuracy = accuracy + 1
	    end
	end
	accuracy_fraction = accuracy / length(y_test)
	md"#### fraction of correctly classified is $(round(accuracy_fraction; digits=2))"
end

# ╔═╡ 4d44ac3b-920a-453f-a5d1-a4de6963b5ab
let
	selected_indices = rand(1:size(x_test,3),5)
	plot(layout=grid(5,1), size=(600,1000),	
		map(selected_indices) do i_test
			scores = model(test_data[1][1][:, i_test])
			v, ind = findmax(scores)
			plot(size=(600,200), layout=grid(1,2,widths=(0.3,0.7)),
				heatmap(x_test[:,end:-1:1,i_test] |> transpose, aspect_ratio=1,
						xaxis=nothing, yaxis=nothing, colorbar=false),
				plot(0:9, scores; seriestype=:bar,
					xticks=0:9, ylims=(0,1), c=2,
					ann=(ind-1, v+0, text(round(v; digits=2), 7,:top)))
				)
		end...)
end

# ╔═╡ Cell order:
# ╟─7bb1811b-8671-4528-beaa-7f51705065aa
# ╟─b89c9e0a-6d28-11f0-0e0c-b723e78bfeca
# ╟─bf13d5a2-35a5-4ce4-a664-2c9bec5800d0
# ╟─40a4a3f7-fe5d-4200-9862-3630a0ae876a
# ╠═bb1b1a04-75da-4dcb-a2e5-2a21524771e5
# ╟─8cd2f2ad-e636-4c6d-8f41-abdc5f8fc881
# ╟─f72101b5-b486-4850-9fcd-1261925d141b
# ╟─b34c5cd0-20bc-40ef-bba7-5bd8b132bc60
# ╠═e19e619d-eb34-4244-b258-5548c5059ab6
# ╟─7ed89ca8-ac55-4e91-b8f1-7ad2daf0783e
# ╟─4d44ac3b-920a-453f-a5d1-a4de6963b5ab
