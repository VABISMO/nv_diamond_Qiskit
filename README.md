# NVQauntum - Qiskit NV Diamond Simulator

🚀 **NVQuantum** is a Python-based simulation of a Nitrogen-Vacancy (NV) center diamond quantum computer, designed to run Shor's algorithm for factoring composite numbers. It models a quantum computing environment with a 532 nm laser, galvo scanning, a 200 fps camera (using OpenCV), automatic NV center alignment, ODMR (Optically Detected Magnetic Resonance) with hyperfine interaction for 14N, and XY8 dynamical decoupling. The simulation operates at liquid nitrogen temperature (77.35 K) and supports affordable hardware components, such as a €50 NV diamond and a €30 microwave chip from Alibaba. The project is split into a reusable library (`nvquantum.py`) for core functionality and a CLI script (`nv.py`) with a user-friendly interface featuring colorful rich tables, progress bars, and robust error handling.

<img width="1114" height="798" alt="image" src="https://github.com/user-attachments/assets/d2de2bc2-86c8-4935-a81a-1b6b7760fe5a" />


## Features

- **Shor's Algorithm**: Factorizes composite numbers (default N=15) with configurable input and automatic coprime selection for `a`.
- **NV Center Simulation**: Simulates NV center detection with realistic, temperature-dependent noise models (T1=10-100 ms, T2=10-100 µs, fluorescence-based readout error 5-15%).
- **ODMR Simulation**: Models ODMR with hyperfine interaction for 14N, with a configurable frequency range (default 2.8-2.9 GHz) and matplotlib plotting.
- **Dynamic Galvo Scanning**: Configurable scan area (default 10x10 µm) for NV center alignment.
- **XY8 Decoupling**: Implements dynamical decoupling to mitigate noise in quantum circuits.
- **Rich CLI Interface**: Features colorful tables, progress bars, and an interactive menu using the `rich` library.
- **Progress Bars**: Visual feedback for time-consuming processes (NV alignment, Shor's execution, ODMR scanning).
- **Reusable Library**: `nvquantum.py` allows custom simulations (e.g., arbitrary circuits) by exposing core functions and classes.
- **Realistic Timing**: Enforces physical component speeds (e.g., laser pulses, galvo steps, camera frames) with `time.sleep`.
- **Robust Error Handling**: Validates inputs (e.g., composite `N`, shots 1-10000) and includes unit tests with `pytest`.
- **Visualization**: Saves ODMR contrast vs. frequency plots as `odmr_spectrum.png`.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`:

```text
qiskit==1.2.4
qiskit-aer==0.15.1
qutip==5.0.3
numpy==1.26.4
opencv-python==4.10.0
pyserial==3.5
rich==13.9.2
pyfiglet==1.0.2
matplotlib==3.9.2
pytest==8.3.3
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VABISMO/nv_diamond_Qiskit.git
   cd nv_diamond_Qiskit
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Python 3.8+ is installed:
   ```bash
   python --version
   ```

## Usage

### CLI Usage (`nv.py`)
Run the CLI script for an interactive experience or with command-line arguments.

#### Interactive Mode
```bash
python nv.py
```
The menu prompts for:
- Simulation mode (only simulation supported).
- Number of qubits (default: 8).
- Number of shots (1-10000, default: 1024).
- Galvo scan area (e.g., "10 10" for 10x10 µm).
- ODMR frequency range (e.g., "2.8 2.9" for 2.8-2.9 GHz).
- Number to factor for Shor's algorithm (e.g., 21, default: 15).

#### Command-Line Mode
```bash
python nv.py --num-qubits 20 --shots 1024 --galvo-area 15 15 --freq-range 2.8 2.9 --N 21
```

#### Example Output
The CLI outputs rich-formatted tables and progress bars:
- **NV Memory Table**: Displays NV centers (e.g., ID, X, Y, Intensity, T1, T2, Readout Err).
- **Shor's Results Table**: Shows measurement outcomes with "State Binary", "State Decimal", and "Counts".
- **ODMR Results**: Reports peak resonance frequency (e.g., ~2.870 GHz) and saves a plot.
- **Progress Bars**: For NV alignment, Shor's execution, and ODMR scanning.

Example NV Memory Table:
```
╭─────────────────────────────── NV Centers (8 Qubits) 📊 ───────────────────────────────╮
│ ID   X (µm)   Y (µm)   Intensity   T1 (ms)   T2 (µs)   Readout Err (%)              │
├──────┼───────┼───────┼───────────┼─────────┼─────────┼────────────────────────────┤
│ 5    │ 6.04  │ 11.43 │ 106.67    │ 193.92  │ 98.47   │ 6.3                        │
│ 6    │ 9.43  │ 11.34 │ 112.50    │ 193.92  │ 98.47   │ 5.2                        │
│ 7    │ 8.20  │ 10.61 │ 104.17    │ 193.92  │ 98.47   │ 6.7                        │
│ ...  │ ...   │ ...   │ ...       │ ...     │ ...     │ ...                        │
╰───────────────────────────────────────────────────────────────────────────────────────╯
```

### Library Usage (`nvquantum.py`)
The library can be used for custom simulations, such as running arbitrary quantum circuits or custom ODMR experiments.

#### Example: Custom Circuit
```python
from nvquantum import NVBackend, auto_align
import qiskit

# Align NV centers
nv_memory = auto_align(galvo_area=(10, 10), num_qubits=4)

# Create a custom circuit
qc = qiskit.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run on NV backend
backend = NVBackend(nv_memory)
result = backend.run(qc, shots=1000)
print(result.get_counts())
```

#### Example: Custom ODMR Simulation
```python
from nvquantum import simulate_odmr
import numpy as np

# Simulate ODMR with custom frequency range
freq, contrast = simulate_odmr(np.linspace(2.85, 2.95, 100))
print(f"Peak frequency: {freq[np.argmax(contrast)]:.3f} GHz")
```

### CLI Options
Run `python nv.py --help` to see options in a rich table:

| Option          | Description                                      | Default   |
|-----------------|--------------------------------------------------|-----------|
| --num-qubits    | Number of qubits (e.g., 20)                      | 8         |
| --shots         | Number of circuit runs (1-10000)                | 1024      |
| --galvo-area    | Galvo area in µm (e.g., 10 10)                  | 10 10     |
| --freq-range    | ODMR frequency range in GHz (e.g., 2.8 2.9)     | 2.8 2.9   |
| --N             | Number to factor for Shor's algorithm (e.g., 21) | 15        |

## Simulated Machine Architecture

The simulated NVQuantum machine consists of interconnected components modeled after a real NV center quantum computer:

```mermaid
graph TD
    A[Laser (532 nm)] -->|Pulse| B[NV Centers]
    B -->|Fluorescence| C[Camera (200 fps)]
    D[Galvo Scanner] -->|Position| B
    E[Microwave Chip] -->|ODMR Pulses| B
    B -->|Qubits| F[Quantum Circuit]
    F -->|Shor's Algorithm| G[NVBackend (Qiskit)]
    G -->|Measurements| H[Results]
    F -->|XY8 Decoupling| I[Noise Mitigation]
    C -->|Frame| J[NV Detection]
    J -->|NV Memory| G
    E -->|Frequency Sweep| K[ODMR Simulation (QuTiP)]
    K -->|Contrast Plot| L[Matplotlib Output]
```

- **Laser**: Generates 532 nm pulses (~1 µs) to excite NV centers.
- **Galvo Scanner**: Moves beam over area (e.g., 10x10 µm, 0.5 µm steps, ~5 ms/point).
- **Camera**: Captures fluorescence at 200 fps (512x512 frames).
- **Microwave Chip**: Simulates ODMR pulses for spin manipulation.
- **NV Centers**: Simulated qubits with T1/T2 noise and fluorescence-based readout.
- **NVBackend**: Qiskit-based backend with noise model for circuit execution.
- **ODMR Simulation**: QuTiP-based simulation of hyperfine interactions.
- **XY8 Decoupling**: Mitigates noise in quantum circuits.

## File Structure

```
nv_diamond_Qiskit/
│
├── nvquantum.py         # Core library for NV center simulation
├── nv.py               # CLI script with rich interface
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── LICENSE.md           # Apache License 2.0 with non-commercial restriction
└── odmr_spectrum.png    # Output ODMR plot (generated)
```

## How It Works

1. **Initialization**: Sets up a rich console with ASCII banner and logging tables (init, hardware, quantum, sim, transpile, ODMR).
2. **NV Center Alignment**: Simulates detection using camera frames, galvo scanning, and laser pulses, with temperature-dependent T1/T2 and fluorescence-based readout.
3. **Shor's Algorithm**: Constructs a quantum circuit to factor `N` (default 15) with a random coprime `a`, using Qiskit with progress bars.
4. **ODMR Simulation**: Models hyperfine interaction for 14N, sweeps frequencies, and plots contrast vs. frequency.
5. **XY8 Decoupling**: Applies dynamical decoupling to mitigate noise.
6. **Output**: Displays rich tables (NV memory, Shor's results), logs, and ODMR plot.

## Technical Details

- **Physical Constants**:
  - Zero-field splitting (D): 2.87 GHz
  - Electron gyromagnetic ratio: 28 GHz/T
  - Hyperfine coupling (14N): -2.16 MHz
  - Quadrupole splitting (14N): -4.95 MHz
  - Temperature: 77.35 K (liquid nitrogen)
  - Fluorescence factor: 1.2 (20% intensity increase)
- **Noise Model**: T1 (10-100 ms, ∝ 300/T), T2 (10-100 µs, ∝ sqrt(300/T)), fluorescence-based readout error (5-15%).
- **Qubit Count**: Dynamic, based on detected NV centers (min 1).
- **Shor's Algorithm**: Uses random coprime `a`, modular exponentiation, and QFT.
- **Timing**: Enforces laser (~1 µs), galvo (~5 ms/point), camera (~5 ms/frame), and gate durations (~100 ns).

## Limitations

- Only simulation mode is fully implemented; real hardware mode is stubbed.
- ODMR simulation is computationally intensive (~5s for 200 points).
- Shor's algorithm assumes even period `r`; may miss some factors.
- Progress bars for Qiskit jobs are simulated (no real-time feedback).

## Testing

Run unit tests to verify core functionality:
```bash
pytest nvquantum.py -v
```
Tests cover `is_prime`, `mod_mult_gate`, and `postprocess_shors`.

## Contributing

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description.

Follow PEP 8 and include docstrings for new functions.

## Issues

Report bugs or feature requests on the [GitHub Issues page](https://github.com/VABISMO/nv_diamond_Qiskit/issues) with:
- Description of the issue/feature.
- Steps to reproduce (if applicable).
- Expected vs. actual behavior.

## License

Licensed under the [Apache License 2.0](LICENSE.md) with a non-commercial use restriction. See [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- [Qiskit](https://qiskit.org/) for quantum circuit simulation.
- [QuTiP](http://qutip.org/) for ODMR modeling.
- [Rich](https://github.com/Textualize/rich) and [Pyfiglet](https://github.com/pwaller/pyfiglet) for CLI interface.
- Inspired by NV center quantum computing research.

---

Happy quantum computing! 🚀