# NVQauntum - Qiskit NV Diamond Simulator

🚀 **NVQuantum** is a Python-based simulation of an NV (Nitrogen-Vacancy) center diamond quantum computer, designed to run Shor's algorithm for factoring composite numbers. It simulates a quantum computing environment with a 532 nm laser, galvo scanning, a 200 fps camera (using OpenCV), automatic NV center alignment, ODMR (Optically Detected Magnetic Resonance) with hyperfine interaction for 14N, and XY8 dynamical decoupling. The simulation operates at liquid nitrogen temperature (77.35 K) and supports affordable hardware components, such as a €50 NV diamond and a €30 microwave chip from Alibaba. The project features a user-friendly command-line interface with colorful rich tables, progress bars for time-consuming processes, and robust error handling.

## Features

- **Shor's Algorithm**: Factorizes composite numbers (default N=15) with configurable input via CLI or interactive menu.
- **NV Center Simulation**: Simulates NV center detection with realistic noise models (T1=10-100 ms, T2=10-100 µs, readout error 5-15%).
- **ODMR Simulation**: Models ODMR with hyperfine interaction for 14N, with a frequency range of 2.8-2.9 GHz.
- **Dynamic Galvo Scanning**: Configurable scan area (default 10x10 µm) for NV center alignment.
- **XY8 Decoupling**: Implements dynamical decoupling to mitigate noise in the quantum circuit.
- **Rich CLI Interface**: Features colorful tables, progress bars, and a sleek menu using the `rich` library.
- **Progress Bars**: Visual feedback for time-consuming processes (NV alignment, Shor's execution, ODMR scanning).
- **Flexible Configuration**: Supports command-line arguments for qubits, shots, galvo area, ODMR range, and the number to factor (N).
- **Error Handling**: Robust validation for inputs, ensuring composite numbers for Shor's algorithm and valid qubit/shot counts.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - `qiskit`: Quantum computing framework for circuit construction and simulation.
  - `qiskit-aer`: Aer simulator for quantum circuit execution.
  - `qutip`: Quantum mechanics simulation for ODMR modeling.
  - `opencv-python`: Image processing for simulated camera frames.
  - `pyserial`: Serial communication (used in simulation mode).
  - `rich`: Rich text and table formatting for CLI interface.
  - `pyfiglet`: ASCII art for the NVQuantum banner.
  - `numpy`: Numerical computations for noise models and matrix operations.

Install dependencies using:

```bash
pip install qiskit qiskit-aer qutip opencv-python pyserial rich pyfiglet numpy
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nvquantum.git
   cd nvquantum
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Python 3.8+ installed:
   ```bash
   python --version
   ```

## Usage

Run the NVQuantum simulator using the command-line interface. You can either use the interactive menu or specify parameters directly.

### Interactive Mode
Run the script without arguments to access the interactive menu:
```bash
python qnv.py
```

The menu will prompt you to:
- Select simulation mode (only simulation is supported).
- Enter the number of qubits (default: 8).
- Enter the number of shots (1-10000, default: 1024).
- Specify the galvo scan area (e.g., "10 10" for 10x10 µm).
- Set the ODMR frequency range (e.g., "2.8 2.9" for 2.8-2.9 GHz).
- Choose the number to factor for Shor's algorithm (e.g., 21, default: 15).

### Command-Line Mode
Specify parameters directly via command-line arguments:
```bash
python qnv.py --num-qubits 20 --shots 1024 --galvo-area 15 15 --freq-range 2.8 2.9 --N 21
```

### Example Output
The program outputs several rich-formatted tables and progress bars, including:
- **NV Memory Table**: Displays detected NV centers with coordinates, intensity, T1, T2, and readout error.
- **Shor's Results Table**: Shows measurement outcomes with "State Binary", "State Decimal", and "Counts".
- **ODMR Results**: Reports the peak resonance frequency (e.g., ~2.870 GHz).
- **Progress Bars**: Visual feedback for NV center alignment, Shor's algorithm execution, and ODMR scanning.

Example command:
```bash
python qnv.py --num-qubits 12 --shots 2048 --N 21
```

### CLI Options
Run `python qnv.py --help` to see the CLI options in a rich table:

| Option          | Description                                      | Default   |
|-----------------|--------------------------------------------------|-----------|
| --num-qubits    | Number of qubits (e.g., 20)                      | 8         |
| --shots         | Number of circuit runs (1-10000)                | 1024      |
| --galvo-area    | Galvo area in µm (e.g., 10 10)                  | 10 10     |
| --freq-range    | ODMR frequency range in GHz (e.g., 2.8 2.9)     | 2.8 2.9   |
| --N             | Number to factor for Shor's algorithm (e.g., 21) | 15        |

## File Structure

```
nvquantum/
│
├── qnv.py               # Main script for NVQuantum simulation
├── requirements.txt     # List of Python dependencies
├── README.md            # This file
```

## How It Works

1. **Initialization**: The script initializes a rich console with an ASCII banner and sets up logging tables for initialization, hardware, quantum circuits, simulation, transpilation, and ODMR.
2. **NV Center Alignment**: Simulates NV center detection using a virtual camera frame, galvo scanning, and laser pulses, with a progress bar for alignment attempts.
3. **Shor's Algorithm**: Constructs a quantum circuit to factor a composite number `N` (default 15) using Qiskit, with a progress bar during execution.
4. **ODMR Simulation**: Models ODMR with hyperfine interaction for 14N, sweeping frequencies (default 2.8-2.9 GHz) and displaying a progress bar.
5. **XY8 Decoupling**: Applies dynamical decoupling to mitigate noise effects.
6. **Output**: Displays results in rich tables, including NV center properties, Shor's measurement counts, and ODMR peak frequency.

## Technical Details

- **Physical Constants**:
  - Zero-field splitting (D): 2.87 GHz
  - Electron gyromagnetic ratio: 28 GHz/T
  - Hyperfine coupling (14N): -2.16 MHz
  - Quadrupole splitting (14N): -4.95 MHz
  - Temperature: 77.35 K (liquid nitrogen)
  - Fluorescence factor: 1.2 (20% intensity increase at 77 K)
- **Noise Model**: Incorporates T1 (10-100 ms), T2 (10-100 µs), and readout errors (5-15%) for realistic simulation.
- **Qubit Count**: Supports up to 12 qubits by default, adjustable via `--num-qubits`.
- **Shor's Algorithm**: Uses a hardcoded `a=7` for the modular exponentiation, with configurable `N` for factoring.

## Limitations

- The simulation assumes idealized hardware behavior for laser pulses and galvo scanning.
- Only simulation mode is supported; real hardware mode is not implemented.
- Shor's algorithm uses a fixed `a=7`, which may not be coprime with all valid `N`. Future updates could add configurable `a`.
- The progress bars simulate progress for processes without real-time feedback (e.g., Qiskit job execution).

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description of your changes.

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments.

## Issues

If you encounter bugs or have feature requests, please open an issue on the GitHub repository with:
- A clear description of the issue or feature.
- Steps to reproduce (if applicable).
- Expected behavior and actual behavior.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Qiskit](https://qiskit.org/) for quantum circuit simulation.
- Uses [QuTiP](http://qutip.org/) for ODMR modeling.
- Enhanced CLI with [Rich](https://github.com/Textualize/rich) and [Pyfiglet](https://github.com/pwaller/pyfiglet).
- Inspired by research on NV center quantum computing.

---

Happy quantum computing! 🚀
