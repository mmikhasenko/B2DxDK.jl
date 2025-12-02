# B2DxDK Decay Model Analysis

This repository contains a Julia implementation of the analysis for the three-body decay B+ → D- D*+ K+ using Pluto.jl notebooks.
The project implements the amplitude model for studying this decay channel,
including various resonance contributions and interference effects.

## References

- **Paper**: [arXiv:2406.03156](https://arxiv.org/pdf/2406.03156)
- **InspireHEP**: [2794793](https://inspirehep.net/literature/2794793)
- **Internal Documentation**: [TWiki](https://twiki.cern.ch/twiki/bin/viewauth/LHCbPhysics/Bm2DstmDpKm)
- **Internal Code**: [GitLab@CERN](https://gitlab.cern.ch/lhcb-b2oc/analyses/b2oc-aman-bu2dstdk-run12/-/issues/1), [GitLab@EP1](https://gitlab.ep1.rub.de/lhcb/b2oc-aman-bu2dstdk-run12)
- **Full TF2 code**: [fork by Alexander](https://github.com/AlexanderKazatsky/B2DxDK/tree/main)

## Overview

The B+ → D- D*+ K+ decay is a complex three-body decay that involves multiple resonance contributions and interference effects.

## Physics Background

The decay B+ → D- D*+ K+ involves several resonance contributions:

### Resonances Included:
- Charmonium states in $D^*D$ system: `EFF(1++)`, `ηc(3945)`, `χc2(3930)`, `hc(4000)`, `χc1(4010)`, `ψ(4040)`, `hc(4300)`
- Tetraquark candidate in $D^*K$ and $DK$ system: `Tcs0(2870)`, `Tcs1(2900)`

## Project Structure

```
B2DxDK/
├── notebooks/
│   └── completion.jl          # Main Pluto.jl notebook
├── data/
│   ├── interference_paper.json    # Paper results for comparison
│   ├── interference_tf.json      # TensorFlow results
│   ├── paper_couplings.json      # Coupling parameters
│   ├── backup_400001.json        # Precomputed integrals
│   └── ...                      # Additional data files
├── scripts/
│   └── cal_pw_fraction.py       # Python script for partial wave analysis
└── README.md
```

## Installation and Usage

### Prerequisites
- Julia 1.10. The package manager will have to resolve the dependencies for any julia version rather than 1.11.5.
- Pluto.jl

For testing the setup in terminal from the project folder, you can run:
```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```
Any problems at this step, should be reported in the project issue tracker.

### Run the analysis

1. **Install Pluto.jl**:
   ```julia
   julia> ] add Pluto
   julia> using Pluto; Pluto.run()
   ```

2. **Open the notebook**:
   - Navigate to the `notebooks/` directory
   - Open `completion.jl` in Pluto

3. **Run the analysis**:
   - The notebook will automatically install required dependencies
   - Execute cells sequentially to perform the analysis


### Using the amplitude extraction

This repository includes a slightly modified version of tf_pwa (https://github.com/jiangyi15/tf-pwa).

Steps to make the analysis code operational:
- Conda has to be installed on the system
- Clone this repository
- In console (inside the repo folder):
  - `chmod +x setup_tf_pwa.sh`
  - `./setup_tf_pwa.sh`

 The current analysis can be found in `Analysis/Amplitude.ipynb`.


 ## Automation of complex amplitude calculation and comparison

 The simulation comparison between tf_pwa and the ThreeBodyDecay framework has been recreated and expanded by linking all given information about tf_pwa (config files, GitHub repositoriy, etc.) to a NotebookLM notebook and generating a python code to calculate the complex amplitudes for individual decay chains and LS couplings. Afterwards the generated file and `the pure_model.jl` file were given to the Antigravity AI agent (Gemini 3 Pro (High)) to implove the generated code and align the hard coded parameters inside `pure_model.jl` to `config_a.yml` and `final_params_full.json` (a renamed but otherwise identical version of `final_params.json`).

 The resulting Python and Julia codes are stored in `Analysis/tf_pwa_analysis_Gemini.py` and `notebook/ThreeBodyDecay_Gemini.jl` respectively. To execute the scripts, run `tf_pwa_analysis_Gemini.py` inside the conda environment created by `setup_tf_pwa.sh` and `ThreeBodyDecay_Gemini.jl` afterwards. This will generate (or regenerate) the `amplitudes.txt` file in the `Analysis` directory.
