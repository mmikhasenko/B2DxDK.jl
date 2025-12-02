using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using ThreeBodyDecays.PartialWaveFunctions
using HadronicLineshapes
using OrderedCollections
using ThreeBodyDecays
using LinearAlgebra
using Parameters
using DataFrames
using JSON
using Random
using ThreeBodyDecays.StaticArrays

# --- Configuration ---
Random.seed!(42)

begin
    const mB = 5.27934 # B+
    const mD = 1.86965 # D+
    const mDx = 2.01026 # Dx+: m(D) + Δm(D*,D) from PDG
    const mK = 0.493677 # K+
end;

(two_js, pc), (_, pv) = map(["0+", "0-"]) do jp0
    ThreeBodySpinParities("0-", "0-", "1-"; jp0)
end;

tbs = let
    ms = ThreeBodyMasses(mD, mK, mDx; m0=mB)
    ThreeBodySystem(ms, two_js)
end;

# --- Lineshapes ---
const EFF = BreitWigner(3.87165, 0.00119);

begin
    @with_kw struct NRexp <: HadronicLineshapes.AbstractFlexFunc
        αβ::ComplexF64
        m0::Float64
    end
    (f::NRexp)(σ::Float64) = exp(f.αβ * (σ - f.m0^2))

    const ConstantLineshape = WrapFlexFunction(x -> 1.0)
end

# --- Decay Chains ---
# Definitions will be built using parameters loaded from JSON later.

# --- Amplitude Calculation Helper ---
struct DalitzAndDecay{T}
    σs::MandelstamTuple{T}
    cosθ::T
    ϕ::T
end

function ThreeBodyDecays.amplitude(three_body_model::ThreeBodyDecay, dd::DalitzAndDecay)
    @unpack σs, cosθ, ϕ = dd
    total_amp = 0.0
    jDx = 1
    _O = amplitude(three_body_model, σs) # order: -1,0,1
    _D = [wignerD(jDx, λ, 0, ϕ, cosθ, 0.0) for λ in -1:1] .|> conj # order: -1,0,1
    total_amp = sum(reshape(_O, 3) .* _D)
    return total_amp
end

# --- Kinematics (Hardcoded from AutomationTest.py) ---
println("\n--- Calculation with AutomationTest.py vectors ---")

# 4-vectors (E, Px, Py, Pz) -> [E, px, py, pz]
pD_vec = [2.0452, -0.1467, 0.2235, -0.7847]
pD0_vec = [2.2606, 0.2284, -0.3689, 1.2019]
pK_vec = [0.7718, -0.0873, 0.1803, -0.5584]
ppi_vec = [0.2017, 0.0056, -0.0349, 0.1413]

# Compute D* 4-vector
pDx_vec = pD0_vec + ppi_vec

# Helper functions
function m2_vec(p)
    return p[1]^2 - p[2]^2 - p[3]^2 - p[4]^2
end

function boost_vec(p, beta)
    # p is [E, px, py, pz]
    # beta is [bx, by, bz]
    b2 = sum(beta .^ 2)
    if b2 >= 1.0
        error("Beta >= 1")
    end
    gamma = 1.0 / sqrt(1.0 - b2)
    bp = sum(p[2:4] .* beta)
    gamma2 = (gamma - 1.0) / b2

    # If b2 is very small, gamma2 -> 0.5
    if b2 < 1e-10
        gamma2 = 0.5
    end

    E_new = gamma * (p[1] - bp)
    p_new = p[2:4] .+ (gamma2 * bp - gamma * p[1]) .* beta
    return [E_new; p_new]
end

# Compute Mandelstam variables
# Particle mapping in pure_model: 1=D, 2=K, 3=Dx
# σ1 = m23^2 = (pK + pDx)^2
# σ2 = m31^2 = (pDx + pD)^2
# σ3 = m12^2 = (pD + pK)^2

s1 = m2_vec(pK_vec + pDx_vec)
s2 = m2_vec(pDx_vec + pD_vec)
s3 = m2_vec(pD_vec + pK_vec)

σs_new = (σ1=s1, σ2=s2, σ3=s3)
println("Calculated σs: ", σs_new)

# Compute angles (Helicity angles of D* -> D0 pi)
# Frame: D* Rest Frame
# Analyzer: D0
# Axes (defined in B Rest Frame):
# z: Direction of D*
# y: Normal to production plane (pD x pK)
# x: y x z

# Boost D0 to D* rest frame
beta_Dx = pDx_vec[2:4] ./ pDx_vec[1]
pD0_DxRF = boost_vec(pD0_vec, beta_Dx)

# Define axes in B frame
z_axis = pDx_vec[2:4]
z_axis = z_axis / norm(z_axis)

pD_3 = pD_vec[2:4]
pK_3 = pK_vec[2:4]
y_axis = cross(pD_3, pK_3)
y_axis = y_axis / norm(y_axis)

x_axis = cross(y_axis, z_axis)

# Project D0 momentum (in D* RF) onto these axes
pD0_3_DxRF = pD0_DxRF[2:4]

px_new = dot(pD0_3_DxRF, x_axis)
py_new = dot(pD0_3_DxRF, y_axis)
pz_new = dot(pD0_3_DxRF, z_axis)

# Calculate angles
p_mag = sqrt(px_new^2 + py_new^2 + pz_new^2)
cos_theta_new = pz_new / p_mag
phi_new = atan(py_new, px_new)

println("Calculated angles: cosθ = $cos_theta_new, ϕ = $phi_new")

# Create test_point
test_point_new = DalitzAndDecay(σs_new, cos_theta_new, phi_new)

# --- Update parameters from JSON (Masses/Widths only) ---
println("\n--- Updating parameters from final_params_full.json ---")

# Load JSON
json_path = joinpath(@__DIR__, "..", "Analysis", "final_params_full.json")
println("Loading parameters from: ", json_path)
params_json = JSON.parsefile(json_path)
val = params_json["value"]

# Helper to get mass/width
get_mass(name) = val["$(name)_mass"]
get_width(name) = val["$(name)_width"]

# Update resonances with loaded parameters
resonances_updated = [
    (; jp=jp"1+", name="EFF(1++)", lineshape=BreitWigner(get_mass("X(3872)"), get_width("X(3872)"))),
    (; jp=jp"0-", name="ηc(3945)", lineshape=BreitWigner(get_mass("X(3915)(0-)"), get_width("X(3915)(0-)"))),
    (; jp=jp"2+", name="χc2(3930)", lineshape=BreitWigner(get_mass("chi(c2)(3930)"), get_width("chi(c2)(3930)"))),
    (; jp=jp"1+", name="hc(4000)", lineshape=BreitWigner(get_mass("X(3940)(1.)"), get_width("X(3940)(1.)"))),
    (; jp=jp"1+", name="χc1(4010)", lineshape=BreitWigner(get_mass("X(3993)"), get_width("X(3993)"))),
    (; jp=jp"1-", name="ψ(4040)", lineshape=BreitWigner(get_mass("Psi(4040)"), get_width("Psi(4040)"))),
    (; jp=jp"1+", name="hc(4300)", lineshape=BreitWigner(get_mass("X(4300)"), get_width("X(4300)"))),
    (; jp=jp"0+", name="Tcs0(2870)", lineshape=BreitWigner(m=get_mass("X0(2900)"), Γ=get_width("X0(2900)"), ma=mD, mb=mK, l=0, d=3.0)),
    (; jp=jp"1-", name="Tcs1(2900)", lineshape=BreitWigner(m=get_mass("X1(2900)"), Γ=get_width("X1(2900)"), ma=mD, mb=mK, l=1, d=3.0)),
    (; jp=jp"1-", name="NR(1--)", lineshape=ConstantLineshape),
    (; jp=jp"0-", name="NR(0--)", lineshape=ConstantLineshape),
    (; jp=jp"1+", name="NR(1++)", lineshape=ConstantLineshape),
    (; jp=jp"0-", name="NR(0-+)", lineshape=NRexp(αβ=val["NR(0-)SPp_alpha"] + val["NR(0-)SPp_beta"] * 1im, m0=4.35))
] |> DataFrame

# Define decay chains structure
decay_chains = [
    (k=2, resonance_name="EFF(1++)", l=0),
    (k=2, resonance_name="EFF(1++)", l=2),
    (k=2, resonance_name="ηc(3945)"),
    (k=2, resonance_name="χc2(3930)"),
    (k=2, resonance_name="hc(4000)", l=0),
    (k=2, resonance_name="hc(4000)", l=2),
    (k=2, resonance_name="χc1(4010)", l=0),
    (k=2, resonance_name="χc1(4010)", l=2),
    (k=2, resonance_name="ψ(4040)"),
    (k=2, resonance_name="hc(4300)", l=0),
    (k=2, resonance_name="hc(4300)", l=2),
    (k=2, resonance_name="NR(1--)"),
    (k=2, resonance_name="NR(0--)"),
    (k=2, resonance_name="NR(1++)", l=0),
    (k=2, resonance_name="NR(0-+)"),
    (k=3, resonance_name="Tcs0(2870)"),
    (k=3, resonance_name="Tcs1(2900)", L=0),
    (k=3, resonance_name="Tcs1(2900)", L=1),
    (k=3, resonance_name="Tcs1(2900)", L=2),
];

# Rebuild chains with updated resonances
chains_updated = let
    resonance_dict = LittleDict(
        resonances_updated.name .=> NamedTuple.(eachrow(resonances_updated)))

    map(decay_chains) do dc
        @unpack k, resonance_name = dc
        _jp = resonance_dict[resonance_name].jp
        comprete_data = complete_l_s_L_S(_jp, tbs.two_js, [pc, pv], dc; k)
        @unpack L, S, l, s = comprete_data
        two_j = _jp.two_j
        d = 3.0
        Xlineshape = resonance_dict[resonance_name].lineshape
        HRk = VertexFunction(RecouplingLS((L, S) .|> x2), BlattWeisskopf{div(x2(L), 2)}(d))
        Hij = VertexFunction(RecouplingLS((l, s) .|> x2), BlattWeisskopf{div(x2(l), 2)}(d))
        DecayChain(; k, two_j, Xlineshape, Hij, HRk, tbs)
    end
end;

# Define the model with updated chains
const model_pure = let
    names = getproperty.(decay_chains, :resonance_name) .*
            "_l" .* [ch.Hij.h.two_ls[1] |> d2 for ch in chains_updated]
    names .*= [(ch.k == 3) ? "_L$(ch.HRk.h.two_ls[1] |> d2)" : "" for ch in chains_updated]
    ThreeBodyDecay(names .=> zip(fill(1.0 + 0.0im, length(chains_updated)), chains_updated))
end;

# --- Output Groups ---
# Define resonance groups (indices in chains)
# Split by LS where applicable
# 1: EFF(1++) l=0 -> X(3872) LS=0
# 2: EFF(1++) l=2 -> X(3872) LS=1
# 3: ηc(3945) -> X(3915) LS=0
# 4: χc2(3930) -> chi(c2)(3930) LS=0
# 5: hc(4000) l=0 -> X(3940) LS=0
# 6: hc(4000) l=2 -> X(3940) LS=1
# 7: χc1(4010) l=0 -> X(3993) LS=0
# 8: χc1(4010) l=2 -> X(3993) LS=1
# 9: ψ(4040) -> Psi(4040) LS=0
# 10: hc(4300) l=0 -> X(4300) LS=0
# 11: hc(4300) l=2 -> X(4300) LS=1
# 12: NR(1--) -> NR(1-)PPm LS=0
# 13: NR(0--) -> NR(0-)SPm LS=0
# 14: NR(1++) -> NR(1+)PSp LS=0
# 15: NR(0-+) -> NR(0-)SPp LS=0
# 16: Tcs0 -> X0(2900) LS=0
# 17: Tcs1 L=0 -> X1(2900) LS=0
# 18: Tcs1 L=1 -> X1(2900) LS=1
# 19: Tcs1 L=2 -> X1(2900) LS=2

# Mapping logic:
# - If splitting by Decay LS (l): Select specific chain.
# - If splitting by Prod LS (L): Select specific chain.
# - If asking for Prod LS but resonance splits by Decay: Sum all Decay chains.
# - If asking for Decay LS but resonance splits by Prod: Sum all Prod chains.

resonance_groups = [
    ("X(3872) Prod LS=0", [1, 2]),
    ("X(3872) Decay LS=0", [1]),
    ("X(3872) Decay LS=1", [2]), ("X(3915)(0-) Prod LS=0", [3]),
    ("X(3915)(0-) Decay LS=0", [3]), ("chi(c2)(3930) Prod LS=0", [4]),
    ("chi(c2)(3930) Decay LS=0", [4]), ("X(3940)(1.) Prod LS=0", [5, 6]),
    ("X(3940)(1.) Decay LS=0", [5]),
    ("X(3940)(1.) Decay LS=1", [6]), ("X(3993) Prod LS=0", [7, 8]),
    ("X(3993) Decay LS=0", [7]),
    ("X(3993) Decay LS=1", [8]), ("Psi(4040) Prod LS=0", [9]),
    ("Psi(4040) Decay LS=0", [9]), ("X(4300) Prod LS=0", [10, 11]),
    ("X(4300) Decay LS=0", [10]),
    ("X(4300) Decay LS=1", [11]), ("NR(0-)SPp Prod LS=0", [15]),
    ("NR(0-)SPp Decay LS=0", [15]), ("NR(1.)PSp Prod LS=0", [14]),
    ("NR(1.)PSp Decay LS=0", [14]), ("NR(0-)SPm Prod LS=0", [13]),
    ("NR(0-)SPm Decay LS=0", [13]), ("NR(1-)PPm Prod LS=0", [12]),
    ("NR(1-)PPm Decay LS=0", [12]), ("X0(2900) Prod LS=0", [16]),
    ("X0(2900) Decay LS=0", [16]), ("X1(2900) Prod LS=0", [17]),
    ("X1(2900) Prod LS=1", [18]),
    ("X1(2900) Prod LS=2", [19]),
    ("X1(2900) Decay LS=0", [17, 18, 19])
]

output_file = joinpath(@__DIR__, "..", "Analysis", "amplitudes.txt")
println("Writing results to: ", output_file)

open(output_file, "a") do io
    println(io, "\nJulia pure_model Results (Coupling 1+0i, LS separated)")

    # Calculate total amplitude first (sum of all unique chains)
    couplings_all = fill(1.0 + 0.0im, length(decay_chains))
    model_all = ThreeBodyDecay(model_pure.names .=> zip(SVector{19}(couplings_all), chains_updated))
    total_amp = amplitude(model_all, test_point_new)

    for (name, indices) in resonance_groups
        # Set couplings: 1.0 for indices in group, 0.0 otherwise
        couplings_list = zeros(ComplexF64, length(decay_chains))
        for idx in indices
            couplings_list[idx] = 1.0 + 0.0im
        end
        new_couplings_updated = SVector{19}(couplings_list)

        # Create updated model
        model_updated = ThreeBodyDecay(model_pure.names .=> zip(new_couplings_updated, chains_updated))

        # Calculate amplitude
        amp = amplitude(model_updated, test_point_new)

        # Format output
        r = real(amp)
        i = imag(amp)
        sign = i >= 0 ? "+" : "-"
        out_str = "Resonance ($name): $r $sign $(abs(i))im"

        println(out_str)
        println(io, out_str)
    end

    println("-"^70)

    r_tot = real(total_amp)
    i_tot = imag(total_amp)
    sign_tot = i_tot >= 0 ? "+" : "-"

    println("Total complex amplitude (Sum): $r_tot $sign_tot $(abs(i_tot))im")
    println(io, "-"^70)
    println(io, "Total complex amplitude (Sum): $r_tot $sign_tot $(abs(i_tot))im")
end