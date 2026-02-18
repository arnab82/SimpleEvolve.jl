using SimpleEvolve
using Test

@testset "SimpleEvolve.jl" begin
include("./test_slepian_H2.jl")
include("./Lih_15.jl")
include("./test_H2_excited.jl")
end

