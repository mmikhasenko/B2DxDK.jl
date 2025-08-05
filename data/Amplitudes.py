from tf_pwa.config_loader import ConfigLoader
import numpy as np
import tensorflow as tf
import json

config = ConfigLoader("config_a.yml")
config.set_params("final_params.json")

phsp = config.get_phsp_noeff()
def convert_complex(d):
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.astype(np.complex128)
        elif isinstance(v, dict):
            convert_complex(v)
convert_complex(phsp)

amp_model = config.get_amplitude()
decay_group = amp_model.decay_group

A_tensor = decay_group.get_amp(phsp)

A = A_tensor.numpy().reshape(-1)

data = {
    'real': A.real.flatten().tolist(),
    'imag': A.imag.flatten().tolist()
}

# In JSON-Datei speichern
with open('amplitudes.json', 'w') as f:
    json.dump(data, f, indent=2)