import numpy as np
from scipy.fft import fft, fft2, fftshift

# Configuration Parameters
c = 3e8
fc = 77e9
B = 4e9
T_chirp = 60e-6
slope = B / T_chirp
fs = 10e6
N_samples = int(T_chirp * fs)
N_chirps = 128

def generate_radar_cube(num_chirps, num_samples, targets, noise_level=0.1):
    t = np.linspace(0, T_chirp, num_samples)
    radar_cube = np.zeros((num_chirps, num_samples), dtype=complex)
    
    for target in targets:
        r0 = target['range']
        v = target['velocity']
        amp = target['rcs']
        
        for chirp_idx in range(num_chirps):
            t_c = chirp_idx * T_chirp 
            r_t = r0 + v * t_c
            fb = slope * 2 * r_t / c
            fd = 2 * v * fc / c
            phase_fast = 2 * np.pi * fb * t
            phase_slow = 2 * np.pi * fd * t_c
            signal = amp * np.exp(1j * (phase_fast + phase_slow))
            radar_cube[chirp_idx, :] += signal
            
    noise = noise_level * (np.random.randn(num_chirps, num_samples) + 1j * np.random.randn(num_chirps, num_samples))
    radar_cube += noise
    return radar_cube

def process_radar_cube(radar_cube):
    range_doppler = fftshift(fft2(radar_cube), axes=0)
    return range_doppler

def get_random_scenario(scenario_type):
    targets = []
    if scenario_type == 'non_metal': # Non-Metal (Empty/Clutter)
        # 50% chance of empty, 50% clutter
        if np.random.rand() > 0.5:
            # Clutter
            num_clutter = np.random.randint(5, 15)
            for _ in range(num_clutter):
                r = np.random.uniform(1, 20)
                v = np.random.uniform(-2, 2)
                rcs = np.random.uniform(0.05, 0.3) # Low RCS for non-metal/clutter
                targets.append({'range': r, 'velocity': v, 'rcs': rcs})
        else:
            # Empty (just noise, handled by generate_radar_cube default noise)
            pass
            
    elif scenario_type == 'metal': # Metal
        # Strong target
        r = np.random.uniform(2, 18)
        v = np.random.uniform(-1, 1)
        rcs = np.random.uniform(2.0, 10.0) # High RCS for metal
        targets.append({'range': r, 'velocity': v, 'rcs': rcs})
        
        # Optional: Add some clutter around it
        if np.random.rand() > 0.7:
            num_clutter = np.random.randint(2, 5)
            for _ in range(num_clutter):
                r_c = np.random.uniform(1, 20)
                v_c = np.random.uniform(-2, 2)
                rcs_c = np.random.uniform(0.05, 0.3)
                targets.append({'range': r_c, 'velocity': v_c, 'rcs': rcs_c})
                
    elif scenario_type == 'metal_in_clutter':
        # Strong target + clutter
        targets.append({'range': 10.0, 'velocity': 0.0, 'rcs': 5.0})
        for _ in range(10):
            r = np.random.uniform(1, 20)
            v = np.random.uniform(-2, 2)
            rcs = np.random.uniform(0.1, 0.5)
            targets.append({'range': r, 'velocity': v, 'rcs': rcs})
            
    elif scenario_type == 'hidden_metal':
        # Weak metal target (hidden) + Strong Clutter
        # Metal target at specific range
        targets.append({'range': 12.0, 'velocity': 0.0, 'rcs': 0.8}) # Weaker RCS
        
        # Strong Clutter masking it
        for _ in range(15):
            r = np.random.uniform(1, 20)
            # Ensure some clutter is near the target to make it "hidden"
            if np.random.rand() > 0.7:
                 r = np.random.uniform(11, 13)
            v = np.random.uniform(-2, 2)
            rcs = np.random.uniform(0.5, 2.0) # Stronger clutter
            targets.append({'range': r, 'velocity': v, 'rcs': rcs})
            
    return targets
