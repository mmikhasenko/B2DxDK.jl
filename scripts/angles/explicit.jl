using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.add(Pkg.PackageSpec(url="https://github.com/mmikhasenko/FourVectors.jl.git"))

using FourVectors

pD_vec = FourVector(-0.1467, 0.2235, -0.7847; E=2.0452)
pK_vec = FourVector(-0.0873, 0.1803, -0.5584; E=0.7718)
ppi_vec = FourVector(0.0056, -0.0349, 0.1413; E=0.2017)
pD0_vec = FourVector(0.2284, -0.3689, 1.2019; E=2.2606)

set_of_vectors = [pD_vec, pK_vec, ppi_vec, pD0_vec]

let
    # 1. Transform to DDx rest frame
    DDx_vec = pD0_vec + ppi_vec + pD_vec
    set_of_vectors_in_DDx_rest_frame = transform_to_cmf.(set_of_vectors, DDx_vec |> Ref)

    @assert spatial_magnitude(set_of_vectors_in_DDx_rest_frame[[1,3,4]] |> sum) < 1e-14

    # 2. Transform to Dx rest frame
    Dx_vec = set_of_vectors_in_DDx_rest_frame[[3,4]] |> sum
    set_of_vectors_in_Dx_rest_frame = transform_to_cmf.(set_of_vectors_in_DDx_rest_frame, Dx_vec |> Ref)

    @assert spatial_magnitude(set_of_vectors_in_Dx_rest_frame[[3,4]] |> sum) < 1e-15

    # 3. Calculate angles of D0
    D0_vec = set_of_vectors_in_Dx_rest_frame[4]

    polar_angle(D0_vec), cos_theta(D0_vec)
end
## Other decay chain via (DK) + Dx


let 
    # 1. orient to go to DK rest frame
    DK_vec = set_of_vectors[[1,2]]|> sum
    set_of_vectors_DK_plane = rotate_to_plane.(set_of_vectors, DK_vec |> Ref, pD_vec |> Ref)
    @assert set_of_vectors_DK_plane[[1,2]] |> sum |> transverse_momentum < 1e-14
    @assert set_of_vectors_DK_plane[1][2] < 1e-14

    set_of_vectors_DK_plane_z2Dx = set_of_vectors_DK_plane .|> Ry(π|>Float64)
    Dx_vec = set_of_vectors_DK_plane_z2Dx[[3,4]] |> sum
    Dx_vec |> transverse_momentum# < 1e-14

    # 2. go to Dx rest frame
    γ_Dx = boost_gamma(Dx_vec)
    set_of_vectors_DK_plane_z2Dx_in_Dx_rest_frame = set_of_vectors_DK_plane_z2Dx .|> Bz(-γ_Dx)

    set_of_vectors_DK_plane_z2Dx_in_Dx_rest_frame[[3,4]] |> sum |> spatial_magnitude
    D0_vec = set_of_vectors_DK_plane_z2Dx_in_Dx_rest_frame[4]

    polar_angle(D0_vec), cos_theta(D0_vec)
end