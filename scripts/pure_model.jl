
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
# 
using ThreeBodyDecays.PartialWaveFunctions
using HadronicLineshapes
using OrderedCollections
using ThreeBodyDecays
using LinearAlgebra
using Measurements
using Statistics
using Parameters
using DataFrames.PrettyTables
using DataFrames
using Setfield
using QuadGK
using Plots
using Optim
using JSON
using Cuba
# 
using Random
Random.seed!(1234)





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
    ms = ThreeBodyMasses(mD, mK, mDx; m0 = mB)
    ThreeBodySystem(ms, two_js)
end;


# Various lineshapes

function BreitWignerSDwaves(; m, Γ, γS, d)
    fr = [γS, 1 - γS]
    channels = map(zip([0, 2], fr)) do (l, x)
        ma, mb = mD, mDx
        # 
        p = HadronicLineshapes.breakup(m, ma, mb)
        gsq = m * Γ / (2 * p / m) * x / BlattWeisskopf{l}(d)(p)^2
        # 
        (; gsq, ma, mb, l, d)
    end
    MultichannelBreitWigner(; m, channels)
end

const EFF = BreitWigner(3.85, 0.001);


begin
    @with_kw struct NRexp <: HadronicLineshapes.AbstractFlexFunc
        αβ::ComplexF64
        m0::Float64
    end
    (f::NRexp)(σ::Float64) = exp(f.αβ * (σ - f.m0^2))
    # 
    const ConstantLineshape = WrapFlexFunction(x -> 1.0)
end


# Decay chains

resonances =
    [
        (; jp = jp"1+", name = "EFF(1++)", lineshape = EFF),
        # 
        (; jp = jp"0-", name = "ηc(3945)", lineshape = BreitWigner(3.945, 0.13)),
        (; jp = jp"2+", name = "χc2(3930)", lineshape = BreitWigner(3.922, 0.0352)),
        (; jp = jp"1+", name = "hc(4000)", lineshape = BreitWignerSDwaves(m = 4.0, Γ = 0.184, γS = cos(0.8)^2, d = 3.0)),
        (; jp = jp"1+", name = "χc1(4010)", lineshape = BreitWignerSDwaves(m = 4.0125, Γ = 0.0627, γS = cos(0.47)^2, d = 3.0)),
        (; jp = jp"1-", name = "ψ(4040)", lineshape = BreitWigner(4.04, 0.084)),
        (; jp = jp"1+", name = "hc(4300)", lineshape = BreitWignerSDwaves(m = 4.3073, Γ = 0.058, γS = cos(1.02)^2, d = 3.0)),
        # Tcbarsbar
        (; jp = jp"0+", name = "Tcs0(2870)", lineshape = BreitWigner(m = 2.914, Γ = 0.128, ma = mD, mb = mK, l = 0, d = 3.0)),
        (; jp = jp"1-", name = "Tcs1(2900)", lineshape = BreitWigner(m = 2.887, Γ = 0.092, ma = mD, mb = mK, l = 1, d = 3.0)),
        # NR
        (; jp = jp"1-", name = "NR(1--)", lineshape = ConstantLineshape),
        (; jp = jp"0-", name = "NR(0--)", lineshape = ConstantLineshape),
        (; jp = jp"1+", name = "NR(1++)", lineshape = ConstantLineshape),
        (; jp = jp"0-", name = "NR(0-+)", lineshape = NRexp(αβ = 0.11 - 0.34im, m0 = 4.35)),
    ] |> DataFrame

decay_chains = [
    (k = 2, resonance_name = "EFF(1++)", l = 0),
    (k = 2, resonance_name = "EFF(1++)", l = 2),
    # 
    (k = 2, resonance_name = "ηc(3945)"),
    (k = 2, resonance_name = "χc2(3930)"),
    (k = 2, resonance_name = "hc(4000)", l = 0),
    (k = 2, resonance_name = "hc(4000)", l = 2),
    (k = 2, resonance_name = "χc1(4010)", l = 0),
    (k = 2, resonance_name = "χc1(4010)", l = 2),
    (k = 2, resonance_name = "ψ(4040)"),
    (k = 2, resonance_name = "hc(4300)", l = 0),
    (k = 2, resonance_name = "hc(4300)", l = 2),
    # 
    (k = 2, resonance_name = "NR(1--)"),
    (k = 2, resonance_name = "NR(0--)"),
    (k = 2, resonance_name = "NR(1++)", l = 0),
    (k = 2, resonance_name = "NR(0-+)"),
    # 
    (k = 3, resonance_name = "Tcs0(2870)"),
    (k = 3, resonance_name = "Tcs1(2900)", L = 0),
    (k = 3, resonance_name = "Tcs1(2900)", L = 1),
    (k = 3, resonance_name = "Tcs1(2900)", L = 2),
];

chains = let
    resonance_dict = LittleDict(
        resonances.name .=> NamedTuple.(eachrow(resonances)))
    #
    map(decay_chains) do dc
        @unpack k, resonance_name = dc
        _jp = resonance_dict[resonance_name].jp
        comprete_data = complete_l_s_L_S(_jp, tbs.two_js, [pc, pv], dc; k) # takes into account l and L from named tuple
        @unpack L, S, l, s = comprete_data
        # 
        two_j = _jp.two_j
        # 
        d = 3.0
        Xlineshape = resonance_dict[resonance_name].lineshape
        # 
        HRk = VertexFunction(RecouplingLS((L, S) .|> x2), BlattWeisskopf{div(x2(L), 2)}(d))
        Hij = VertexFunction(RecouplingLS((l, s) .|> x2), BlattWeisskopf{div(x2(l), 2)}(d))
        # 
        DecayChain(; k, two_j, Xlineshape, Hij, HRk, tbs)
    end
end;

const model_pure = let
    names = getproperty.(decay_chains, :resonance_name) .*
            "_l" .* [ch.Hij.h.two_ls[1] |> d2 for ch in chains]
    names .*= [(ch.k == 3) ? "_L$(ch.HRk.h.two_ls[1] |> d2)" : "" for ch in chains]
    ThreeBodyDecay(names .=> zip(fill(1.0 + 0.0im, length(chains)), chains))
end;




# compute full amplitude

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



# testing: use the DPD configuration from data/crosscheck_event.json

json_path = joinpath(@__DIR__, "..", "data", "crosscheck_event.json")
event = JSON.parsefile(json_path)
dpd = event["dpd_kinematics"]

# Map DPD invariants to the Mandelstam variables for (D, K, Dx)
# Convention in ThreeBodyDecays: σ₁ = m²(23), σ₂ = m²(13), σ₃ = m²(12)
# Here: 1 ≡ D, 2 ≡ K, 3 ≡ Dx
σs_dpd = (
    σ1 = dpd["msq_KDx"],  # m²(K, Dx)  → pair (2,3)
    σ2 = dpd["msq_DxD"],  # m²(D, Dx)  → pair (1,3)
    σ3 = dpd["msq_DK"],   # m²(D, K)   → pair (1,2)
)

dalitz_dpd = DalitzAndDecay(
    σs_dpd,
    dpd["cos_theta_D_in_Dx"],
    dpd["phi_D_in_Dx"],
)

full_amplitude = amplitude(model_pure, dalitz_dpd)

println("## Amplitude at DPD cross-check event")
println("DalitzAndDecay = ", dalitz_dpd)
println("amplitude(model_pure, dalitz_dpd) = ", full_amplitude)

# amplitudes for individual decay chains
println("\n## Amplitudes per decay chain at DPD cross-check event")
chain_amps = [
    (chain_name = name,
     amplitude  = amplitude(model_pure[i], dalitz_dpd))
    for (i, name) in enumerate(model_pure.names)
]

df_chain_amps = DataFrame(chain_amps)
println(df_chain_amps)

@assert df_chain_amps.amplitude |> sum ≈ full_amplitude atol=1e-10

