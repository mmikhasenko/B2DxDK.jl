import Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using FourVectors
using LazyDecayAngles

let
    # Input vectors (in Lab frame)
    pD  = FourVector(-0.1467, 0.2235, -0.7847; E=2.0452) # 1
    pK  = FourVector(-0.0873, 0.1803, -0.5584; E=0.7718) # 2
    ppi = FourVector( 0.0056,-0.0349,  0.1413; E=0.2017) # 3
    pD0 = FourVector( 0.2284,-0.3689,  1.2019; E=2.2606) # 4
    
    # Tuple for type stability: indices are 1=D, 2=K, 3=pi, 4=D0
    objs = (pD, pK, ppi, pD0) 
    
    # --- Program: Analyze Branch (D0, pi) ---
    # Topology: Total -> (D0, pi) -> D0
    # Here we treat the (D0, pi) system as "Particle 2" relative to the first branch,
    # potentially using ToHelicityFrameParticle2 if we want the second-particle convention.
    program_D0pi = (
        # 1. Go to rest frame of Bp
        ToHelicityFrame((1, 2, 3, 4)),
        
        # 2. Measure (D0, pi) system properties
        MeasureMassCosThetaPhi(:D0pi_vars, (4, 3)),
        
        # 3. Go to (D0, pi) rest frame using Particle 2 convention
        #    (Assuming (D0, pi) is the recoil system against (D, K))
        ToHelicityFrame((4, 3)),
        
        # 4. Measure D0 angles in (D0, pi) frame
        MeasureCosThetaPhi(:vars_D0, 4)
    )
    
    # Execute
    (_, res_D0pi) = execute_decay_program(objs, program_D0pi)
    
    cosθ_python = -0.8649158171627784
    ϕ_python = 0.6942087211091432
    
    abs(res_D0pi.vars_D0.cosθ - cosθ_python) < 1e-10
    abs(res_D0pi.vars_D0.ϕ - ϕ_python) < 1e-10
end
