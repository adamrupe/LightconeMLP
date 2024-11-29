function circle_map(x::Float64; nonlinearity::AbstractFloat, phase_shift::AbstractFloat=0.5)
    r = nonlinearity
    ω = phase_shift
    
    f = x + ω - ((r/(2.0*pi)) * sin(2.0*pi*x))
    mod(f, 1.0)
end

"""
Returns a spacetime field of the specified map lattice
for the given map function, with periodic boundary conditions.
"""
function map_lattice(initial_condition::Vector{Float64}, 
                     time_steps::Int, 
                     map_function; 
                     coupling::AbstractFloat, params...)
                     
    c = coupling
    f = map_function
    lattice_size = length(initial_condition)
    
    spacetime = Array{Float64}(undef, lattice_size, time_steps+1)
    spacetime[:,1] = initial_condition # set first column of spacetime to initial configuration
    
    for t = 1:time_steps
        current_state = spacetime[:,t]
        next_state = @view spacetime[:,t+1]
        for s = 1:lattice_size
            mapped_site = f(current_state[s]; params...)
            # get left and right neighbors, using periodic boundaries
            if s == 1
                mapped_left = f(current_state[end]; params...)
                mapped_right = f(current_state[s+1]; params...)
            elseif s == lattice_size
                mapped_left = f(current_state[s-1]; params...)
                mapped_right = f(current_state[1]; params...)
            else
                mapped_left = f(current_state[s-1]; params...)
                mapped_right = f(current_state[s+1]; params...)
            end
            
            next_state[s] = mod((1-c)*mapped_site + (c/2)*(mapped_left + mapped_right), 1.0)
            
        end
    end
    return permutedims(spacetime, (2,1))
end

"""
Simple function for making spacetime diagrams
"""
function diagram(field; color=:Greys, scale=400)
    T, X = size(field)
    S = (floor(X/T*scale), scale)
    Plots.heatmap(field; color=color, colorbar=false, yflip=true, size=S)
end

function circle_map_ensemble(N_ensemble, space, time; 
                                  transient=300, coupling=1.0, nonlinearity=1.0)
    ensemble = Array{Float32}(undef, space, time*N_ensemble)
    for n in 1:N_ensemble
        initial = rand(space)
        field = map_lattice(initial, transient+time, circle_map; coupling=coupling, nonlinearity=nonlinearity)
        train_field = convert(Array{Float32}, field)[transient+2:end, :]
        T, X = size(train_field)
        ens_offset = (n-1)*T
        for t in 1:T
            ensemble[:, ens_offset+t] = train_field[t, :]
        end
    end
    return ensemble
end