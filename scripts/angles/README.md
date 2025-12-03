

1. mention explicit.jl # setting manually the rotations
2. mention with_LDA.jl # using the LazyDecayAngles library

## Installing the dependencies

Dependencies are prescrined in `Manifest.toml`. Make sure the file is there, and you run the same julia version as the one used to create the file.
The project relies on two non-registered packages:
- [FourVectors.jl](https://github.com/mmikhasenko/FourVectors.jl.git)
- [LazyDecayAngles.jl](https://github.com/mmikhasenko/LazyDecayAngles.jl.git)

## Setting up the environment

```bash
] activate scripts/angles
] instantiate
```

