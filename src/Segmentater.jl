module Segmentater

using Flux
using Images

using Flux.Data:DataLoader

include("imageloader.jl")
include("model.jl")

end # module
