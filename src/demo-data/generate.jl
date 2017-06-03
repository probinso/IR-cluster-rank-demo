#!/usr/bin/env Julia

using Distributions

size, subsize, depth = parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3])


function generator(μ, subsize, size, depth)
    N = MvNormal(μ, eye(2) * ((depth * 0.5) ^ 4))
    if depth == 1
        for _ in 1:size
            X = rand(N)
            produce([X' N.μ'])
        end
    else
        for seed in 1:subsize
            generator(rand(N), subsize, size, depth-1)
        end
    end
end

p = Task(() -> generator(zeros(2), subsize, size, depth))

Data = vcat([X for X in p]...)

using DataFrames

df = DataFrame(x=Data[:, 1], y=Data[:, 2], mux=Data[:, 3], muy=Data[:, 4])

writetable("jam.csv", df)

using Plots

df[:group] = (df[:mux] .* df[:muy])

plt = scatter(df[:x], df[:y], color=df[:group])

png("plot")
