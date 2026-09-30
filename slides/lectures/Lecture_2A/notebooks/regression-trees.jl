### A Pluto.jl notebook ###
# v0.20.13

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

# ╔═╡ 2cd53efe-a9f5-4770-8ca3-67f8e0405d30
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using Plots, DataFrames, PlutoUI, QuadGK, LaTeXStrings, DecisionTree, Statistics, Random

end

# ╔═╡ b8382884-5293-4227-8b11-ef7c09e295cc
TableOfContents()

# ╔═╡ 0fe9550b-c0b4-44f6-9213-8916608dde7a
md"""
In this exercise session, you will learn:
1) How to train regression trees for 1D and multivariate functions
2) Training a random forest for optimization of your functions
"""

# ╔═╡ 1a25487d-96e9-4de5-9b4e-71bf8b510684
md"""
## Exercise 1 : Regression Trees
Regression trees involves splitting data into two halves at a node value and using the average of training points in each category as the prediction. This process is recursively repeated until maxinum number of splitting is achieved for the tree. As first exercise, we will play along with some 1D and 2D functions and use DecisionTree.jl to train a regression tree.
"""

# ╔═╡ 5d525bcf-87a5-424c-b0cf-2d1acb5de9cb
md"""
Simulated data of some complex functions with some gaussian noise added to it. A part of this data will be used for training a regression tree. Some functions are listed below, you can also write your own function
"""

# ╔═╡ f714e0b7-df14-478c-aa90-1d93bec5901c
f1_1D(x) = sin.(2x) 

# ╔═╡ e98a668d-397b-46ed-a355-339841e737df
f2_1D(x) = sin.(exp.(.- x .^ 2)) 

# ╔═╡ 944286d6-f0d4-41c6-a70d-971daa0d5524
f3_1D(x) = 2 .* exp.(-2 .* x)

# ╔═╡ 2f6652d5-dc86-4567-b665-5756d1686e50
@bind n_sample Slider(100:100:5000, show_value=true,default=1000) #Number of data points to simulate

# ╔═╡ 74b640b8-369d-466c-bc90-349b3a7a769d
@bind train_fraction Slider(0.1:0.1:0.9, show_value=true,default=0.5) # Fraction of samples to consider for training, rest will be automatically considered for test.

# ╔═╡ 9cf9795b-e717-4399-9719-cb10497cc727
@bind tree_depth Slider(0:1:20, show_value=true,default=5) #Depth set for a regression tree, equivalent to number of splits allowed in the tree

# ╔═╡ bbd876c4-a86c-4f79-8270-5c96cffc2932
md"""
Lets create our sample which will be used to create a decision tree model. We first randomly sample input features based on the number of data points (n_sample). Use one of the functions (f1_1D,f2_1D,f3_1D) for labels 'y'. Add random noise to it (you can change the scaling factor). We store this data in a dataframe because it would be easier to handle it. Complete the below code block and run it in the cell below.
```
begin
	x = 2π * rand(MersenneTwister(12), n_sample)
	y = <your_1Dfunction>(x) .+ 0.01 .* randn(MersenneTwister(14), n_sample)
	Data = DataFrame()
	Data[!,"x_data"] = x
	Data[!,"y_data"] = y
	shuffle!(MersenneTwister(123), Data) #Randomly shuffles the datapoints
	n_train_1D = Int(floor(n_sample * train_fraction))
	n_test_1D = Int(floor(n_sample * (1-train_fraction)))
end
```
"""

# ╔═╡ f7a972c1-bd8a-48a1-8c56-29adcc80f335
begin
	x = 2π * rand(MersenneTwister(12), n_sample)
	y = f2_1D(x) .+ 0.01 .* randn(MersenneTwister(14), n_sample)
	Data = DataFrame()
	Data[!,"x_data"] = x
	Data[!,"y_data"] = y
	shuffle!(MersenneTwister(123), Data) #Randomly shuffles the datapoints
	n_train_1D = Int(floor(n_sample * train_fraction))
	n_test_1D = Int(floor(n_sample * (1-train_fraction)))
end

# ╔═╡ 919b70e2-5762-4f39-878c-7f0bc71b6fa8
begin
	#TODO Create training and test datasets from 'Data' dataframe. You can consider selecting some number of rows for training and rest for test.
	
	Ex1D_training = Data[1:n_train_1D,:]
	Ex1D_test = Data[n_train_1D+1:end,:]
end

# ╔═╡ d054433b-bacb-40ab-84ac-ba8c5229a245
md"""
Alright, lets try to fit a simple regression tree!
"""

# ╔═╡ 49bbcc37-b256-4fd5-9bf9-04e1f85851a6
md"""
Features need to be in a matrix NxM where M is the number of features considered and N corresponds to the number of values for corresponding features
"""

# ╔═╡ 22a9de71-aab6-4d80-9f6e-5f83c23a3002
features_1D = reshape(Ex1D_training.x_data, :, 1)  #Reshape method can be used to convert our array into required format.

# ╔═╡ dec696bf-e524-4670-8bf2-3991d2c4362d
ytrue_1D = Ex1D_training.y_data #These are the y-values for your features. Uncomment this after defining your training dataset.

# ╔═╡ 4d94395d-6b08-4e98-8b6b-be9ef662e2eb
model_1D = let model = DecisionTreeRegressor(max_depth=tree_depth, rng=123)
    fit!(model,features_1D,ytrue_1D)
    model
end

# ╔═╡ 85fcceb2-7d28-439a-b1b2-fda9a36427a3
# Model fitting is part of the model cell above.

# ╔═╡ d1f629f7-7d87-41ad-ace3-851f42985a44
md"""
Lets see the mean squared error for this tree
"""

# ╔═╡ 15e46291-40a1-4fe9-b7e9-cb503cdfa5e3
md"""
We will see how the model performs on the test sample. Copy paste the below code once you complete training of your model
```
let
	#TODO test_features_1D = reshape(...)
	y_pred = predict.(Ref(model_1D), eachrow(test_features_1D))
	test_mse = mean((y_pred .- Ex1D_test.y_data).^2)
	println("Test MSE:$(test_mse)")
	plot(title="Testing regression tree on test data")
	scatter!(Ex1D_test.x_data,Ex1D_test.y_data,label="test data")
	scatter!(Ex1D_test.x_data,y_pred,label="model predictions")
	xlabel!("x_input")
	ylabel!("y_output")
end
```
"""

# ╔═╡ 728f800f-586a-40d1-9c51-bd49585d56d4
let
	test_features_1D = reshape(Ex1D_test.x_data, :, 1)
	y_pred = predict.(Ref(model_1D), eachrow(test_features_1D))
	test_mse = mean((y_pred .- Ex1D_test.y_data).^2)
	println("Test MSE:$(test_mse)")
	plot(title="Testing regression tree on test data")
	scatter!(Ex1D_test.x_data,Ex1D_test.y_data,label="test data")
	scatter!(Ex1D_test.x_data,y_pred,label="model predictions")
	xlabel!("x_input")
	ylabel!("y_output")
end

# ╔═╡ 5b59aa42-3c1c-47e4-8b9f-fd9b3d3a1008
md"""
#### Notice the step like output of the model. While training the tree, it averages the points available after a split and is the output value of the model. If the splits are less, it flattens the output for a range of inputs. This can be smoothened by increasing the dept of the tree or by training a random forest (more on this later in the notebook)
"""

# ╔═╡ 71b5fd7f-7518-4023-a457-23f4500391bb
print_tree(model_1D) #You can view your trained tree by printing it out. Notice the splits observed at the leaf nodes based on the features

# ╔═╡ 54e3a807-ed8f-49c7-a49a-c9c918805885
md"""
#### Awesome, lets go to a multivariate function now. 

The procedure is exactly similar as 1D case, we just now introduce a third variable "z" as function of inputs x2 and y2.
"""

# ╔═╡ d085bfca-5bea-44a0-9518-aaec03e308f9
f1_2D(x,y) = sin.(x).* cos.(y) 

# ╔═╡ 39cda66e-a43e-44bf-8ad4-d89a93f06a52
f2_2D(x,y) = sin.(x).* cos.(y .^ 2)

# ╔═╡ 5193c54b-da80-43a9-9321-b06b6dc73881
f3_2D(x,y) = log.(abs.(x .* sin.(y)))

# ╔═╡ 842c77c5-4481-43c3-9106-2de564eb3799
md"""
Complete the code block below to start
```
begin
	x2 = 2π * rand(MersenneTwister(12), n_sample) .+ 1.0
	y2 = rand(MersenneTwister(13), n_sample) .* 2π .+ 1.0
	z = <your_2Dfunction>(x2,y2) .+ 0.005 .* randn(MersenneTwister(14), n_sample)
	#Just store this data in a dataframe to handle things easily
	Data2D = DataFrame()
	Data2D[!,"x_data"] = x2
	Data2D[!,"y_data"] = y2
	Data2D[!,"z_data"] = z
	shuffle!(MersenneTwister(123), Data2D)
	n_train_2D = Int(floor(n_sample * train_fraction))
	n_test_2D = Int(floor(n_sample * (1-train_fraction)))
end
```
"""

# ╔═╡ 23297db7-6961-46ef-a19c-d72c5f27f7d2
begin
	x2 = 2π * rand(MersenneTwister(12), n_sample) .+ 1.0
	y2 = rand(MersenneTwister(13), n_sample) .* 2π .+ 1.0 
	z = f2_2D(x2,y2) .+ 0.005 .* randn(MersenneTwister(14), n_sample)
	#Just store this data in a dataframe to handle things easily
	Data2D = DataFrame()
	Data2D[!,"x_data"] = x2
	Data2D[!,"y_data"] = y2
	Data2D[!,"z_data"] = z
	shuffle!(MersenneTwister(123), Data2D)
	n_train_2D = Int(floor(n_sample * train_fraction))
	n_test_2D = Int(floor(n_sample * (1-train_fraction)))
end

# ╔═╡ 6c9e1490-8d91-4356-a1e3-fae3beb9a148
md"""
Lets plot this function. Copy paste the code below.
```
scatter3d(Data2D.x_data, Data2D.y_data, Data2D.z_data; xlabel="x", ylabel="y", zlabel="F(x,y)", label="sampled function", markersize=2)
```
"""

# ╔═╡ b8238aac-03b2-4189-878e-67b94935d744
scatter3d(Data2D.x_data, Data2D.y_data, Data2D.z_data; xlabel="x", ylabel="y", zlabel="F(x,y)", label="sampled function", markersize=2)

# ╔═╡ 7a215db3-8dad-4e35-97e6-4237379bac77
begin
	#TODO Create datasets for your 2D decision tree as defined above for 1D example
	Ex2D_training = Data2D[1:n_train_2D,:]
	Ex2D_test = Data2D[n_train_2D+1:end,:]
end

# ╔═╡ cd36ecc7-ef48-461c-b2e1-c9edc8364718
features_2D = hcat(Ex2D_training.x_data, Ex2D_training.y_data)

# ╔═╡ 56a2e6e9-dd06-4b17-8ca5-01c3a5d9274d
model_2D = let model = DecisionTreeRegressor(max_depth=tree_depth, rng=123)
    fit!(model,features_2D,Ex2D_training.z_data)
    model
end

# ╔═╡ de3456e2-362e-4a30-afcc-3e7b0b53f646
# Model fitting is part of the model cell above.

# ╔═╡ c5f7fa3f-3db8-4271-89b4-94dc9a9c92d5
md"""
Do you think you need more depth to fit a multivariate function? How does a splitting happens in multivariate functions? Uncomment the cell below to see
"""

# ╔═╡ 7a1cd02b-0f9c-4cba-a0a3-6f7c392318a1
print_tree(model_2D)

# ╔═╡ d41ed8ab-71f3-4e43-91ff-e476fe991e83
md"""
We test the predictions of our tree on the test data
```
let
	test_features_2D = hcat(...) #You need test data here
	z_pred = predict.(Ref(model_2D), eachrow(test_features_2D))
	println("Test MSE from single tree = $(mean((z_pred .- Ex2D_test.z_data).^2))")
	scatter3d(Ex2D_test.x_data, Ex2D_test.y_data, Ex2D_test.z_data; xlabel="x", ylabel="y", zlabel="F(x,y)", label="validation data", markersize=2)
	scatter3d!(Ex2D_test.x_data, Ex2D_test.y_data,z_pred,label=" Single Tree")
end
```
"""

# ╔═╡ 238e2994-7251-4055-85f9-4afff001b98e
let
	test_features_2D = hcat(Ex2D_test.x_data,Ex2D_test.y_data) #You need test data here
	z_pred = predict.(Ref(model_2D), eachrow(test_features_2D))
	println("Test MSE from single tree = $(mean((z_pred .- Ex2D_test.z_data).^2))")
	scatter3d(Ex2D_test.x_data, Ex2D_test.y_data, Ex2D_test.z_data; xlabel="x", ylabel="y", zlabel="F(x,y)", label="validation data", markersize=2)
	scatter3d!(Ex2D_test.x_data, Ex2D_test.y_data,z_pred,label=" Single Tree")
end

# ╔═╡ 81897c7a-312f-4b75-a38e-61edec39e6bb
md"""
At each internal node, the tree searches the available features and candidate thresholds. Each leaf predicts the mean training target in its region. Restricting the candidate features at each split is an additional random-forest option.
"""

# ╔═╡ 56fd86cb-8ed4-4bb0-b2dd-86be08c5660a
md"""
## Exercise 2: Random Forest. 
You might have noticed how a single tree with lower depth results in coarser representations of our functions which can produce high bias. In addition, if the depth is increased, the tree has a chance to be overfitted. Hence, we can use bagging techniques like random forest in order to optimize the models. A forest is 
1) Group of trees with different bootstrap samples (their errors may remain correlated)
2) Trained on different subsets of data
So unlike a single tree, a test input passes through all the trees in the forest and the average of the individual tree outputs is considered as the final prediction of the forest.

Lets revisit our 1D example and this time we fit a forest model for it.
"""

# ╔═╡ 5246067a-770d-455a-817d-766cca7eead5
@bind Ntrees Slider(10:10:100, show_value=true,default=20) #Controls the number of trees in your forest

# ╔═╡ 6c4bfbdf-3710-47ac-80e8-babaf439c372
forest1D = let model = RandomForestRegressor(n_trees=Ntrees,max_depth=tree_depth, rng=123)
    fit!(model, features_1D, ytrue_1D)
    model
end

# ╔═╡ ff2d18e1-eac2-4f68-90c4-bc38f8529ff5
# Model fitting is part of the model cell above.

# ╔═╡ 780a7e9a-806e-40fa-a087-ffa13a5910be
md"""
### General performance of the models with the function. It can be seen how the forest smoothens out the steps. 
"""

# ╔═╡ ce5ea121-65f7-47ad-b522-74ded0ac79b6
md"""
Lets compare the behavior of our decision tree and forest models. Complete the code below and run it in the cell below.
```
let
	x_range = collect(range(0,stop=6,step=0.05))
	x_features = reshape(...) #You need x_range here
	y_output = <your_function>(x_range)
	y_pred_tree = predict.(Ref(model_1D), eachrow(x_features))
	mse_tree = mean((y_pred_tree .- y_output).^2) #
	println("MSE from single tree:$(mse_tree)")
	y_pred_forest = predict.(Ref(forest1D), eachrow(x_features))
	mse_forest = mean((y_pred_forest .- y_output).^2)
	println("MSE from forest:$(mse_forest)")
	plot(title="Max_Depth_per_tree = $(tree_depth),Ntrees=$(Ntrees) for forest")
	plot!(...,label="Function")
	scatter!(...,label="Forest")
	scatter!(...,label="Single tree")
	xlabel!("x_input")
	ylabel!("y_output")
	#xlims!(1,1)
end
```
"""

# ╔═╡ dc39faf3-bbb3-4d12-aaad-15b16f817f37
let
	x_range = collect(range(0,stop=6,step=0.05))
	x_features = reshape(x_range,:,1) #You need x_range here
	y_output = f2_1D(x_range)
	y_pred_tree = predict.(Ref(model_1D), eachrow(x_features))
	mse_tree = mean((y_pred_tree .- y_output).^2) #
	println("MSE from single tree:$(mse_tree)")
	y_pred_forest = predict.(Ref(forest1D), eachrow(x_features))
	mse_forest = mean((y_pred_forest .- y_output).^2)
	println("MSE from forest:$(mse_forest)")
	plot(title="Max_Depth_per_tree = $(tree_depth),Ntrees=$(Ntrees) for forest")
	plot!(x_range,y_output,label="Function")
	scatter!(x_range,y_pred_forest,label="Forest")
	scatter!(x_range,y_pred_tree,label="Single tree")
	xlabel!("x_input")
	ylabel!("y_output")
	#xlims!(1,1)
end



# ╔═╡ c9053139-73a1-4fb8-82ff-05f921eb761f
md"""
#### Lets come back to our 2D example and try fitting a random forest
"""

# ╔═╡ afd1c291-924c-4b43-9ba7-6ca7c0cee740
forest2D = let model = RandomForestRegressor(n_trees=Ntrees, max_depth=tree_depth, rng=123)
    fit!(model, features_2D, Ex2D_training.z_data)
    model
end

# ╔═╡ c4d30248-0e98-442e-8e37-1d6803d58b8c
# Model fitting is part of the model cell above.

# ╔═╡ a4d77437-5502-4115-bf39-45de9a11a269
md"""
#### Do you think its performing better than a single tree? Discuss.
"""

# ╔═╡ 521ad368-febd-4db3-b74a-ae02b4b03119
md"""
As a last step, lets again compare the general performance of single tree and random forest for our 2D functions. Do you agree with the MSE evaluated?
```
let
	x_range = collect(range(1,stop=6,step=0.05))
	y_range = collect(range(1,stop=6,step=0.05))
	xy_features = hcat(...) #You know what you need here :)
	z_output = <your_2Dfunction>(x_range,y_range)
	z_pred_tree = predict.(Ref(model_2D), eachrow(xy_features))
	mse_tree = mean(<expression to get mean squared error>)
	println("MSE from single tree:$(mse_tree)")
	z_pred_forest = predict.(Ref(forest2D), eachrow(xy_features))
	mse_forest = mean(<expression to get mean squared error>)
	println("MSE from forest:$(mse_forest)")
	plot(title="Max_Depth_per_tree = $(tree_depth),Ntrees=$(Ntrees) for forest")
	plot!(...,label="Function")
	scatter3d!(...,label="Forest")
	scatter3d!(...,label="Single tree")
	xlabel!("x_input")
	ylabel!("y_input")
	zlabel!("z_output")
	#xlims!(1,1)
end
```
"""

# ╔═╡ a7338484-3c3c-426f-a97e-555c0e76acc5
let
	x_range = collect(range(1,stop=6,step=0.05))
	y_range = collect(range(1,stop=6,step=0.05))
	xy_features = reduce(vcat, [permutedims([a,b]) for a in x_range for b in y_range])
	x_range, y_range = xy_features[:,1], xy_features[:,2]
	z_output = f2_2D(x_range,y_range)
	z_pred_tree = predict.(Ref(model_2D), eachrow(xy_features))
	mse_tree = mean((z_output .- z_pred_tree).^2)
	println("MSE from single tree:$(mse_tree)")
	z_pred_forest = predict.(Ref(forest2D), eachrow(xy_features))
	mse_forest = mean((z_output .- z_pred_forest).^2)
	println("MSE from forest:$(mse_forest)")
	plot(title="Max_Depth_per_tree = $(tree_depth),Ntrees=$(Ntrees) for forest")
	plot!(x_range,y_range,z_output,label="Function")
	scatter3d!(x_range,y_range,z_pred_forest,label="Forest")
	scatter3d!(x_range,y_range,z_pred_tree,label="Single tree")
	xlabel!("x_input")
	ylabel!("y_input")
	zlabel!("z_output")
	#xlims!(1,1)
end

# ╔═╡ 1d6e8436-47f7-4726-9461-14c99ab6c8f3
md"""
### Notes:
1) Averaging can reduce variance, especially when tree errors are weakly correlated.
2) Performance on peaks and discontinuities depends on depth, data density, and the function. Compare validation errors rather than assuming one method always wins.
"""

# ╔═╡ 601aa31d-c815-4246-a44f-298f477810d9
md"""
## Exercise 3 : Loss functions
In Julia, DecisionTree.jl by default trains a tree or forest using MeanSquaredError (MSE) as the cost function. The robustness of splitting at each node is decided by your cost function. Hence, MSE is hardcoded in DecisionTree.jl and cannot be modified.
Other loss functions which can be useful while training boosted trees.
1) Huber loss function
2) Logistic loss function
As a short exercise, lets visualize [Huber](https://en.wikipedia.org/wiki/Huber_loss) loss function
"""

# ╔═╡ d50c607f-9bc6-4ea9-87e0-5563ded4a39e
md"""
From the definition, write a simple function to evaluate Huber loss
```
function HuberLoss(δy;δ=0.5)
	#Your function here
end
```
"""

# ╔═╡ fbdd3967-b14c-4431-93c6-0f9829a7eb6b
function HuberLoss(δy;δ=0.5)
	if abs(δy) <= δ #δy = y_true - y_prediction
		return 0.5 * (δy^2)
	else
		return δ * (abs(δy) - 0.5 * δ)
	end
end

# ╔═╡ c561c0f9-0e5c-4c33-bfb4-17b9c3b5cbfd
@bind δ Slider(0:0.001:0.05,show_value=true,default=0.001)

# ╔═╡ 69af712e-d4e8-489a-8db9-21500c6270ca
md"""
Lets plot the two loss functions (MSE and Huber) for our 1D forest model.
```
let
	#δ=0.05
	x_range = collect(range(0,stop=8,step=0.01))
	x_features = reshape(...)
	y_output = <your_1Dfunction>(x_range)
	y_pred = predict.(Ref(forest1D), eachrow(x_features))
	Δy  = sort(y_output .- y_pred)
	mse = Δy .^ 2
	hub = [HuberLoss(Δy[i];δ) for i in 1:length(Δy)]
	plot(title="Comparison of MSE vs HuberLoss")
	xlabel!(L"y_{true} - y_{prediction}")
	ylabel!("Loss value")
	plot!(Δy,mse,label="MeanSquaredError")
	plot!(Δy,hub,label="Huber Loss(δ=$(δ))")
end
```
"""

# ╔═╡ f2d61909-0de6-48d6-b387-858689e5e06b
let
	#δ=0.05
	x_range = collect(range(0,stop=8,step=0.01))
	x_features = reshape(x_range,:,1)
	y_output = f2_1D(x_range)
	y_pred = predict.(Ref(forest1D), eachrow(x_features))
	Δy  = sort(y_output .- y_pred)
	mse = Δy .^ 2
	hub = [HuberLoss(Δy[i];δ) for i in 1:length(Δy)]
	plot(title="Comparison of MSE vs HuberLoss")
	xlabel!(L"y_{true} - y_{prediction}")
	ylabel!("Loss value")
	plot!(Δy,mse,label="MeanSquaredError")
	plot!(Δy,hub,label="Huber Loss(δ=$(δ))")
end

# ╔═╡ 529480e9-83a8-4fee-ab5b-71f7a19471f3
md"""
### For large errors, Huber loss linearizes the cost function which results in a lower penalty.
"""

# ╔═╡ Cell order:
# ╠═b8382884-5293-4227-8b11-ef7c09e295cc
# ╠═2cd53efe-a9f5-4770-8ca3-67f8e0405d30
# ╟─0fe9550b-c0b4-44f6-9213-8916608dde7a
# ╟─1a25487d-96e9-4de5-9b4e-71bf8b510684
# ╟─5d525bcf-87a5-424c-b0cf-2d1acb5de9cb
# ╠═f714e0b7-df14-478c-aa90-1d93bec5901c
# ╠═e98a668d-397b-46ed-a355-339841e737df
# ╠═944286d6-f0d4-41c6-a70d-971daa0d5524
# ╠═2f6652d5-dc86-4567-b665-5756d1686e50
# ╠═74b640b8-369d-466c-bc90-349b3a7a769d
# ╠═9cf9795b-e717-4399-9719-cb10497cc727
# ╠═bbd876c4-a86c-4f79-8270-5c96cffc2932
# ╠═f7a972c1-bd8a-48a1-8c56-29adcc80f335
# ╠═919b70e2-5762-4f39-878c-7f0bc71b6fa8
# ╠═d054433b-bacb-40ab-84ac-ba8c5229a245
# ╠═4d94395d-6b08-4e98-8b6b-be9ef662e2eb
# ╟─49bbcc37-b256-4fd5-9bf9-04e1f85851a6
# ╠═22a9de71-aab6-4d80-9f6e-5f83c23a3002
# ╠═dec696bf-e524-4670-8bf2-3991d2c4362d
# ╠═85fcceb2-7d28-439a-b1b2-fda9a36427a3
# ╠═d1f629f7-7d87-41ad-ace3-851f42985a44
# ╟─15e46291-40a1-4fe9-b7e9-cb503cdfa5e3
# ╠═728f800f-586a-40d1-9c51-bd49585d56d4
# ╟─5b59aa42-3c1c-47e4-8b9f-fd9b3d3a1008
# ╠═71b5fd7f-7518-4023-a457-23f4500391bb
# ╟─54e3a807-ed8f-49c7-a49a-c9c918805885
# ╠═d085bfca-5bea-44a0-9518-aaec03e308f9
# ╠═39cda66e-a43e-44bf-8ad4-d89a93f06a52
# ╠═5193c54b-da80-43a9-9321-b06b6dc73881
# ╠═842c77c5-4481-43c3-9106-2de564eb3799
# ╠═23297db7-6961-46ef-a19c-d72c5f27f7d2
# ╟─6c9e1490-8d91-4356-a1e3-fae3beb9a148
# ╠═b8238aac-03b2-4189-878e-67b94935d744
# ╠═7a215db3-8dad-4e35-97e6-4237379bac77
# ╠═56a2e6e9-dd06-4b17-8ca5-01c3a5d9274d
# ╠═cd36ecc7-ef48-461c-b2e1-c9edc8364718
# ╠═de3456e2-362e-4a30-afcc-3e7b0b53f646
# ╠═c5f7fa3f-3db8-4271-89b4-94dc9a9c92d5
# ╠═7a1cd02b-0f9c-4cba-a0a3-6f7c392318a1
# ╠═d41ed8ab-71f3-4e43-91ff-e476fe991e83
# ╠═238e2994-7251-4055-85f9-4afff001b98e
# ╟─81897c7a-312f-4b75-a38e-61edec39e6bb
# ╟─56fd86cb-8ed4-4bb0-b2dd-86be08c5660a
# ╠═5246067a-770d-455a-817d-766cca7eead5
# ╠═6c4bfbdf-3710-47ac-80e8-babaf439c372
# ╠═ff2d18e1-eac2-4f68-90c4-bc38f8529ff5
# ╠═780a7e9a-806e-40fa-a087-ffa13a5910be
# ╟─ce5ea121-65f7-47ad-b522-74ded0ac79b6
# ╠═dc39faf3-bbb3-4d12-aaad-15b16f817f37
# ╠═c9053139-73a1-4fb8-82ff-05f921eb761f
# ╠═afd1c291-924c-4b43-9ba7-6ca7c0cee740
# ╠═c4d30248-0e98-442e-8e37-1d6803d58b8c
# ╠═a4d77437-5502-4115-bf39-45de9a11a269
# ╟─521ad368-febd-4db3-b74a-ae02b4b03119
# ╠═a7338484-3c3c-426f-a97e-555c0e76acc5
# ╠═1d6e8436-47f7-4726-9461-14c99ab6c8f3
# ╠═601aa31d-c815-4246-a44f-298f477810d9
# ╟─d50c607f-9bc6-4ea9-87e0-5563ded4a39e
# ╠═fbdd3967-b14c-4431-93c6-0f9829a7eb6b
# ╠═c561c0f9-0e5c-4c33-bfb4-17b9c3b5cbfd
# ╟─69af712e-d4e8-489a-8db9-21500c6270ca
# ╠═f2d61909-0de6-48d6-b387-858689e5e06b
# ╟─529480e9-83a8-4fee-ab5b-71f7a19471f3
