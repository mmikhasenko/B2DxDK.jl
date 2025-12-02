import numpy as np
import tensorflow as tf
from tf_pwa.config_loader import ConfigLoader 
import json
import extra_amp

# --- Configuration ---
CONFIG_FILE = "config_a.yml"
PARAMS_FILE = "final_params_full.json"

# --- Main Logic ---
try:
    # 1. Load Configuration
    config = ConfigLoader(CONFIG_FILE)
    
    # 2. Load Parameters
    with open(PARAMS_FILE, 'r') as f:
        params_dict = json.load(f)['value']
    
    # Set all couplings to 1+0i for comparison
    params_ones = params_dict.copy()
    for k in params_ones:
        if "total" in k or "g_ls" in k:
            if k.endswith("r"):
                params_ones[k] = 1.0
            elif k.endswith("i"):
                params_ones[k] = 0.0
                
    config.set_params(params_ones) 

    # 3. Define Kinematics (Hardcoded for consistency with Julia)
    # Using the same phase space point as in pure_model.jl
    particles = list(config.get_decay().outs)
    particle_map = {p.name: p for p in particles}
    
    p4_dict = {
        particle_map["D"]: tf.constant([[2.0452, -0.1467, 0.2235, -0.7847]], dtype=tf.float64),
        particle_map["D0"]: tf.constant([[2.2606, 0.2284, -0.3689, 1.2019]], dtype=tf.float64),
        particle_map["K"]: tf.constant([[0.7718, -0.0873, 0.1803, -0.5584]], dtype=tf.float64),
        particle_map["pi"]: tf.constant([[0.2017, 0.0056, -0.0349, 0.1413]], dtype=tf.float64)
    }

    phsp_variables = config.data.cal_angle(p4_dict)
    phsp_variables["c"] = np.array([-1.0]) # Extra variable required by config
    
    # 4. Calculate Amplitudes
    amp_model = config.get_amplitude()
    dg = amp_model.decay_group
    all_chains = dg.chains
    
    with open("amplitudes.txt", "w") as outfile:
        outfile.write("Python tf-pwa Results (Coupling 1+0i, LS separated)\n")
        
        for i, chain in enumerate(all_chains):
            dg.set_used_chains([i])
            
            # Identify resonance name from chain string
            chain_str = str(chain)
            res_name = ""
            try:
                # Parse chain string: "Bp->X(3872)+K, ..."
                decays = chain_str.split(",")
                first_decay = decays[0].strip() 
                products = first_decay.split("->")[1]
                
                # Split products handling parentheses
                parts = []
                current_part = ""
                paren_level = 0
                for char in products:
                    if char == "+" and paren_level == 0:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        if char == "(": paren_level += 1
                        if char == ")": paren_level -= 1
                        current_part += char
                parts.append(current_part.strip())
                
                for p in parts:
                    if p not in ["K", "D", "Dst", "D0", "pi"]:
                        res_name = p
                        break
                
                if not res_name:
                     res_name = parts[0] if parts[0] != "K" else parts[1]
                         
            except Exception as e:
                print(f"Warning: Could not parse resonance name from {chain_str}: {e}")
                res_name = "Unknown"

            # Handle naming mismatch for X(3940)
            if "(1+)" in res_name:
                temp_name = res_name.replace("(1+)", "(1.)")
                found = any(temp_name in k for k in params_ones)
                if found:
                    res_name = temp_name

            # Find LS keys
            prod_keys = []
            decay_keys = []
            
            for k in params_ones:
                # Decay LS keys
                if f"{res_name}->Dst.D_g_ls_" in k or f"{res_name}->D.K_g_ls_" in k:
                    if k.endswith("r"):
                        base_key = k[:-1]
                        if base_key not in decay_keys:
                            decay_keys.append(base_key)
                # Production LS keys
                if f"Bp->{res_name}" in k and "g_ls_" in k:
                     if k.endswith("r"):
                        base_key = k[:-1]
                        if base_key not in prod_keys:
                            prod_keys.append(base_key)
            
            all_ls_keys = prod_keys + decay_keys
            
            if not all_ls_keys:
                # No LS structure, calculate as is
                amplitude_tensor = dg.get_amp(phsp_variables)
                val = amplitude_tensor.numpy().flatten()[0]
                
                sign = "+" if val.imag >= 0 else "-"
                out_str = f"Resonance ({res_name}): {val.real} {sign} {abs(val.imag)}im"
                print(out_str)
                outfile.write(out_str + "\n")
            else:
                # Iterate over Production keys
                prod_keys.sort()
                for ls_key in prod_keys:
                    current_params = params_ones.copy()
                    # Isolate this Production LS
                    for k in prod_keys:
                        current_params[k+"r"] = 0.0
                        current_params[k+"i"] = 0.0
                    current_params[ls_key+"r"] = 1.0
                    
                    config.set_params(current_params)
                    val = dg.get_amp(phsp_variables).numpy().flatten()[0]
                    
                    ls_idx = ls_key.split("g_ls_")[-1]
                    sign = "+" if val.imag >= 0 else "-"
                    out_str = f"Resonance ({res_name} Prod LS={ls_idx}): {val.real} {sign} {abs(val.imag)}im"
                    print(out_str)
                    outfile.write(out_str + "\n")

                # Iterate over Decay keys
                decay_keys.sort()
                for ls_key in decay_keys:
                    current_params = params_ones.copy()
                    # Isolate this Decay LS
                    for k in decay_keys:
                        current_params[k+"r"] = 0.0
                        current_params[k+"i"] = 0.0
                    current_params[ls_key+"r"] = 1.0
                    
                    config.set_params(current_params)
                    val = dg.get_amp(phsp_variables).numpy().flatten()[0]
                    
                    ls_idx = ls_key.split("g_ls_")[-1]
                    sign = "+" if val.imag >= 0 else "-"
                    out_str = f"Resonance ({res_name} Decay LS={ls_idx}): {val.real} {sign} {abs(val.imag)}im"
                    print(out_str)
                    outfile.write(out_str + "\n")
                    
                # Restore params
                config.set_params(params_ones)

        # Calculate Total Amplitude
        dg.set_used_chains(list(range(len(all_chains))))
        total_val = dg.get_amp(phsp_variables).numpy().flatten()[0]
        
        sign = "+" if total_val.imag >= 0 else "-"
        print("-" * 70)
        print(f"Total complex amplitude (Sum): {total_val.real} {sign} {abs(total_val.imag)}im")
        outfile.write("-" * 70 + "\n")
        outfile.write(f"Total complex amplitude (Sum): {total_val.real} {sign} {abs(total_val.imag)}im\n")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nError: {e}")