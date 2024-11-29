"""
1+1 D only, expects spacetime fields with dimensions (space, time)
"""

import Plots
import Flux
import NNlib
import BSON
import JLD2
using ProgressMeter: Progress, next!
using Zygote: pullback

include("./MapLattice.jl")

function lightcone_size(depth::Int, c::Int)
    size = 0
    for d = 0:depth
        size += (2*c*d + 1)
    end
    return size
end

function grab_plc(field, X::Int, T::Int, past_depth::Int, c::Int)
    D_space, D_time = size(field)
    lc_size = lightcone_size(past_depth, c)
    plc = Array{eltype(field)}(undef, lc_size)
    p = 1
    for d = 0:past_depth
        window_size = 2*c*d 
        for w = 0:(window_size)
            a = -d*c + w - 1
            plc[p] = field[1+mod(X+a, D_space), T-d]
            p += 1
        end
    end
    return plc
end

function extract_plcs(field, past_depth, c)
    X, T = size(field)
    adjusted_T = T - past_depth
    lc_size = lightcone_size(past_depth, c)
    plcs = Array{eltype(field)}(undef, X*adjusted_T, lc_size)
    wrap = past_depth * c
    # implement periodic b.c. by wrapping the edges
    wrapped_field = vcat(field[end-(wrap-1):end,:], field, field[1:wrap,:])
    X_0 = wrap # space reference
    T_0 = past_depth # time reference 
    
    i = 1
    for t = 1:adjusted_T
        for x = 1:X
            p = 1
            for d = 0:past_depth
                window_size = 2*c*d 
                for w = 0:(window_size)
                    a = -d*c + w 
                    plcs[i,p] = wrapped_field[X_0+x+a, T_0+t-d]
                    p += 1
                end
            end
            i += 1
        end
    end
    return(plcs)
end

function stfield_local_data(field, past_depth, c)
    plcs = permutedims(extract_plcs(field, past_depth, c), (2,1))
    inputs = plcs[2:end, :]
    outputs = plcs[1:1, :]
    return (inputs, outputs)
end

function circle_map_local_data(N_ensemble, space, time, past_depth, c; 
                               transient=100, coupling=1.0, nonlinearity=1.0)
    adjusted_T = time - past_depth
    lc_size = lightcone_size(past_depth, c)
    field_size = space*adjusted_T
    all_plcs = Array{Float64}(undef, lc_size, field_size*N_ensemble)
    i = 0
    for n in 1:N_ensemble
        initial = rand(space)
        field = map_lattice(initial, transient+time, circle_map; coupling=coupling, nonlinearity=nonlinearity)
        train_field = permutedims(field[transient+2:end, :], (2,1))
        plcs = permutedims(extract_plcs(train_field, past_depth, c), (2,1))
        all_plcs[:, (n-1)*field_size+1 : n*field_size] = plcs
    end
    inputs = all_plcs[2:end, :]
    outputs = all_plcs[1:1, :]
    return(inputs, outputs)
end

function lcMLP(past_depth, c)
    lc_size = lightcone_size(past_depth, c)
    input_size = lc_size - 1
    model = Flux.Chain(
                       Flux.Dense(input_size, input_size, Flux.σ),
                       Flux.Dense(input_size, Int(floor(input_size/2)), Flux.σ),
                       Flux.Dense(Int(floor(input_size/2)), 1, Flux.σ)
                      )
    return model
end

function train_local(m, dataloader, num_epochs, optimiser, scores)
    trainable_params = Flux.params(m)

    for epoch_num = 1:num_epochs
        acc_loss = 0.0
        progress_tracker = Progress(length(dataloader), 1, "Training epoch $epoch_num: ")
        for (input, output) in dataloader
            loss, back = pullback(trainable_params) do
                local_loss(m, input,output)
            end
            gradients = back(one(loss))
            
            acc_loss += loss

            Flux.Optimise.update!(optimiser, trainable_params, gradients)
            
            next!(progress_tracker; showvalues=[(:loss, loss)])
        end

        avg_loss = acc_loss / length(dataloader)
        push!(scores, avg_loss)
    end
end

local_loss(m, input, output) = Flux.mse(m(input), output)

function local_mlp_forecast(field, n_steps, model, past_depth, c)
    X, T = size(field)
    addition = Array{eltype(field)}(undef, X, n_steps)
    full_field = hcat(field, addition)
    
    for t = 1:n_steps
        for x = 1:X
            plc = grab_plc(full_field, x, T+t, past_depth, c)
            input = plc[2:end]
            output = model(input)[1]
            full_field[x, T+t] = output
        end
    end
    return full_field[:, T+1:end]
end   

"""
functions for second stage of training
using MSE of full (single) spatial fields

*** Need to re-think this, won't work as initially thought***
"""

# function circle_map_markov_data(N_ensemble, space, time, past_depth, c; 
#                                transient=100, coupling=1.0, nonlinearity=1.0)
#     adjusted_T = time - past_depth
#     lc_size = lightcone_size(past_depth, c)

#     inputs = Array{Float64}(undef, space, past_depth, adjusted_T*N_ensemble)
#     outputs = Array{Float64}(undef, space, 1, adjusted_T*N_ensemble)
    
#     i = 1
#     for n in 1:N_ensemble
#         initial = rand(space)
#         field = map_lattice(initial, transient+time, circle_map; coupling=coupling, nonlinearity=nonlinearity)
#         train_field = permutedims(field[transient+2:end, :], (2,1))
#         for t in 1:adjusted_T
#             inputs[:,:, i] = train_field[:, t:t+past_depth-1]
#             outputs[:,:, i] = train_field[:, t+past_depth:t+past_depth]
#             i += 1
#         end
#     end
#     return(inputs, outputs)
# end

# function markov_loss_single(model, input, output, past_depth, c)
#     pred = local_mlp_forecast(input, 1, model, past_depth, c)
#     loss = Flux.mse(pred, output)
#     return loss
# end

# function markov_loss_batch(model, inputs, outputs, past_depth, c)
#     n_batch = size(inputs)[end]
#     agg_loss = 0
#     for i = 1:n_batch
#         input = inputs[:,:, i]
#         output = outputs[:,:, i]
#         loss = markov_loss_single(model, input, output, past_depth, c)
#         agg_loss += loss
#     end
#     return agg_loss/n_batch
# end

# function train_markov(m, dataloader, num_epochs, optimiser, scores, past_depth, c)
#     trainable_params = Flux.params(m)

#     for epoch_num = 1:num_epochs
#         acc_loss = 0.0
#         progress_tracker = Progress(length(dataloader), 1, "Training epoch $epoch_num: ")
#         for (input, output) in dataloader
#             loss, back = pullback(trainable_params) do
#                 markov_loss_batch(m, input,output, past_depth,c)
#             end
#             gradients = back(one(loss))
            
#             acc_loss += loss

#             Flux.Optimise.update!(optimiser, trainable_params, gradients)
            
#             next!(progress_tracker; showvalues=[(:loss, loss)])
#         end

#         avg_loss = acc_loss / length(dataloader)
#         push!(scores, avg_loss)
#     end
# end