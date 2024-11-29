include("./LightconeMLP.jl")

past_depth = 8
c = 1

model = lcMLP(past_depth, c)
opt = Flux.Optimise.Optimiser(Flux.Optimise.WeightDecay(1e-6), Flux.ADAM(1e-4))
scores = [];
training = circle_map_local_data(200, 200, 600, past_depth, c; 
                               transient=100, coupling=1.0, nonlinearity=1.0);
data = Flux.Data.DataLoader(training, batchsize=10_000,shuffle=true);
@JLD2.save "cml_local_train" data

train_local(model, data, 200, opt, scores)

@JLD2.save "cml_local_8-1_scores" scores
BSON.@save "cml_local_model_8-1.bson" model