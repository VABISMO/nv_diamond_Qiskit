# File: nvquantum.py
# Description: Core library for NVQuantum, a simulated NV (Nitrogen-Vacancy) center diamond quantum computer.
# Provides functionality for NV center alignment, ODMR simulation, Shor's algorithm, and XY8 decoupling.
# Notes:
# - Requires qiskit, qiskit-aer, qutip, opencv-python, numpy, matplotlib.
# - Simulates at 77.35 K with realistic noise (T1=10-100 ms, T2=10-100 µs, fluorescence-based readout).
# - Features: Temperature-dependent noise, realistic timing, coprime selection for Shor, ODMR plotting.
import numpy as np
import time
import cv2
from qutip import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit import Gate
from fractions import Fraction
from math import gcd
import matplotlib.pyplot as plt
import random

# Suppress QuTiP FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")

# Physical constants
D = 2.87e9  # Zero-field splitting (Hz)
gamma_e = 28e9  # Electron gyromagnetic ratio (Hz/T)
A_hf = -2.16e6  # Hyperfine coupling for 14N (Hz)
Q_n = -4.95e6  # Quadrupole splitting for 14N (Hz)
TEMPERATURE = 77.35  # Liquid nitrogen temperature in Kelvin
FLUORESCENCE_FACTOR = 1.2  # 20% intensity increase at 77 K
T1_BASE = 50e-3  # Baseline T1 at 300 K (s)
T2_BASE = 50e-6  # Baseline T2 at 300 K (s)

def is_prime(n, k=5):
    """Check if a number is prime using the Rabin-Miller primality test.

    Args:
        n (int): Number to check.
        k (int): Number of iterations for probabilistic testing (default: 5).

    Returns:
        bool: True if n is likely prime, False if composite.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness test
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)  # a^d mod n
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)  # x^2 mod n
            if x == n - 1:
                break
        else:
            return False
    return True

def select_coprime(N):
    """Select a random number coprime to N (gcd(a, N) == 1).

    Args:
        N (int): Number to find a coprime for.

    Returns:
        int: A random number a such that gcd(a, N) == 1.
    """
    candidates = [a for a in range(2, N) if gcd(a, N) == 1]
    if not candidates:
        raise ValueError(f"No coprime found for N={N}")
    return int(np.random.choice(candidates))  # Convert to Python int

def simulate_laser_pulse(intensity=1.0, duration=1e-6, hardware_logs=None):
    """Simulate a 532 nm laser pulse with noise and physical timing.

    Args:
        intensity (float): Pulse intensity (default: 1.0).
        duration (float): Pulse duration in seconds (default: 1e-6).
        hardware_logs (rich.table.Table, optional): Table to log messages.

    Returns:
        dict: Pulse data with intensity, duration, pulse value, and timestamp.
    """
    start_time = time.perf_counter()
    intensity_noise = np.random.normal(1.0, 0.05) * intensity
    jitter = np.random.normal(0, 0.1e-6)
    duration += jitter
    time.sleep(duration)  # Enforce physical timing
    pulse_data = {'pulse': intensity_noise * 4095, 'timestamp': time.time(), 'intensity': intensity_noise, 'duration': duration*1e6}
    if hardware_logs:
        hardware_logs.add_row(f"[yellow]Simulating laser pulse (532 nm, intensity={intensity_noise:.2f}, duration={duration*1e6:.2f} µs) 🔆[/]")
        hardware_logs.add_row(f"[yellow]Laser pulse took {(time.perf_counter() - start_time)*1e3:.2f} ms[/]")
    return pulse_data

def simulate_galvo_scan(area=(10, 10), step=0.5, hardware_logs=None):
    """Simulate galvo scanning over a specified area.

    Args:
        area (tuple): Scan area in µm (x, y).
        step (float): Step size in µm (default: 0.5).
        hardware_logs (rich.table.Table, optional): Table to log messages.

    Returns:
        list: List of (x, y) coordinates scanned.
    """
    start_time = time.perf_counter()
    x_steps = int(area[0] / step)
    y_steps = int(area[1] / step)
    coords = [(x * step, y * step) for x in range(x_steps) for y in range(y_steps)]
    for _ in coords:
        time.sleep(0.005)  # Simulate galvo move time (5 ms per point)
    if hardware_logs:
        hardware_logs.add_row(f"[yellow]Simulating galvo scan over {area[0]}x{area[1]} µm with {step} µm step at {TEMPERATURE:.1f} K 🔍[/]")
        hardware_logs.add_row(f"[yellow]Galvo scan took {(time.perf_counter() - start_time)*1e3:.2f} ms[/]")
    return coords

def simulate_camera_frame(nv_centers, shape=(512, 512), sigma=5, background_mean=20, background_std=5, hardware_logs=None):
    """Simulate a camera frame with NV centers and noise.

    Args:
        nv_centers (list): List of NV center dictionaries with x, y, intensity.
        shape (tuple): Frame dimensions (height, width).
        sigma (float): Gaussian spread for NV centers.
        background_mean (float): Background noise mean.
        background_std (float): Background noise standard deviation.
        hardware_logs (rich.table.Table, optional): Table to log messages.

    Returns:
        np.ndarray: Simulated grayscale frame (uint8).
    """
    start_time = time.perf_counter()
    frame = np.random.normal(background_mean, background_std, shape)
    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for nv in nv_centers:
        x, y, intensity = nv['x'], nv['y'], nv['intensity'] * FLUORESCENCE_FACTOR
        frame += intensity * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    frame = np.random.poisson(np.clip(frame, 0, None))
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    time.sleep(1/200)  # Simulate 200 fps camera (5 ms per frame)
    if hardware_logs:
        hardware_logs.add_row(f"[yellow]Camera frame simulation took {(time.perf_counter() - start_time)*1e3:.2f} ms[/]")
    return frame

def detect_nv_centers(frame, threshold=50):
    """Detect NV centers in a simulated camera frame using contour detection.

    Args:
        frame (np.ndarray): Grayscale image frame.
        threshold (int): Intensity threshold for contour detection (default: 50).

    Returns:
        list: List of detected NV centers with x, y, intensity.
    """
    _, thresh = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nv_centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            intensity = frame[cy, cx] / FLUORESCENCE_FACTOR
            nv_centers.append({'x': cx, 'y': cy, 'intensity': intensity})
    return nv_centers

def auto_align(galvo_area=(10, 10), camera_shape=(512, 512), max_retries=5, num_qubits=8, hardware_logs=None, progress=None):
    """Automatically align and detect NV centers with a progress bar.

    Args:
        galvo_area (tuple): Scan area in µm (x, y).
        camera_shape (tuple): Camera frame dimensions (height, width).
        max_retries (int): Maximum alignment attempts (default: 5).
        num_qubits (int): Required number of qubits.
        hardware_logs (rich.table.Table, optional): Table to log messages.
        progress (rich.progress.Progress, optional): Progress bar for alignment.

    Returns:
        list: List of NV center dictionaries with x, y, intensity, qubit_id, T1, T2, readout_err.
    """
    scan_time = 5 + num_qubits * 0.5
    task = None
    if progress:
        task = progress.add_task("Aligning NV centers...", total=max_retries)
    for attempt in range(max_retries):
        start_time = time.perf_counter()
        if hardware_logs:
            hardware_logs.add_row(f"[bold yellow]Aligning NV centers, attempt {attempt + 1}/{max_retries}, scan time {scan_time:.1f} s 🔍[/]")
        area_um2 = galvo_area[0] * galvo_area[1]
        density = np.random.uniform(1, 10)
        num_nv = int(np.random.uniform(5, min(30, density * area_um2)))
        nv_centers = [{'x': np.random.uniform(50, camera_shape[0]-50), 'y': np.random.uniform(50, camera_shape[1]-50), 'intensity': np.random.uniform(50, 200)} for _ in range(num_nv)]
        frame = simulate_camera_frame(nv_centers, camera_shape, hardware_logs=hardware_logs)
        coords = simulate_galvo_scan(galvo_area, step=0.5, hardware_logs=hardware_logs)
        simulate_laser_pulse(hardware_logs=hardware_logs)
        detected = detect_nv_centers(frame, threshold=50)
        density = len(detected) / (galvo_area[0] * galvo_area[1]) if detected else 0
        if hardware_logs:
            hardware_logs.add_row(f"[yellow]{len(detected)} NV centers detected (density ~{density:.2f}/µm²)[/]")
        nv_memory = []
        min_distance = 10
        for nv in detected:
            duplicate = False
            for existing in nv_memory:
                dist = np.sqrt((nv['x'] - existing['x'])**2 + (nv['y'] - existing['y'])**2)
                if dist < min_distance:
                    duplicate = True
                    break
            if not duplicate:
                x_um = nv['x'] * galvo_area[0] / camera_shape[0]
                y_um = nv['y'] * galvo_area[1] / camera_shape[1]
                t1 = T1_BASE * (300 / TEMPERATURE)
                t2 = T2_BASE * np.sqrt(300 / TEMPERATURE)
                readout_err = 0.15 / (1 + np.exp((nv['intensity'] - 100) / 20))
                nv_memory.append({
                    'x': x_um,
                    'y': y_um,
                    'intensity': nv['intensity'],
                    'qubit_id': len(nv_memory),
                    'T1': t1,
                    'T2': t2,
                    'readout_err': readout_err
                })
        usable_qubits = [nv for nv in nv_memory if nv['intensity'] > 100 and nv['T2'] > 20e-6]
        if progress:
            progress.update(task, advance=1)
        if hardware_logs:
            hardware_logs.add_row(f"[yellow]Alignment attempt took {(time.perf_counter() - start_time)*1e3:.2f} ms[/]")
        if len(usable_qubits) >= num_qubits:
            if hardware_logs:
                hardware_logs.add_row(f"[green]{len(nv_memory)} NV centers, {len(usable_qubits)} usable qubits ✅[/]")
            break
        if hardware_logs:
            hardware_logs.add_row(f"[yellow]Need {num_qubits} qubits, got {len(usable_qubits)}, retrying...[/]")
    else:
        if hardware_logs:
            hardware_logs.add_row(f"[red]Failed to find {num_qubits} qubits 🚨[/]")
        raise ValueError(f"Insufficient NV centers: {len(usable_qubits)} found")
    return nv_memory

def mod_mult_gate(b, N):
    """Create a modular multiplication gate for Shor's algorithm.

    Args:
        b (int): Multiplier.
        N (int): Modulus.

    Returns:
        UnitaryGate: Quantum gate for modular multiplication.
    """
    if gcd(b, N) != 1:
        raise ValueError(f"gcd({b}, {N}) must be 1")
    n = int(np.ceil(np.log2(N)))
    U = np.zeros((2**n, 2**n), dtype=complex)
    for x in range(2**n):
        y = (b * x) % N if x < N else x
        U[y, x] = 1
    return UnitaryGate(U, label=f"M_{b} mod {N}")

class NVBackend:
    """Simulated NV center quantum backend with noise model.

    Args:
        nv_memory (list): List of NV center dictionaries.
        max_qubits (int): Maximum qubits supported (default: None, uses len(nv_memory)).
        gate_time (float): Single-qubit gate duration (s, default: 100e-9).
        two_qubit_gate_time (float): Two-qubit gate duration (s, default: 1e-6).
        init_logs (rich.table.Table, optional): Table to log initialization messages.
    """
    def __init__(self, nv_memory, max_qubits=None, gate_time=100e-9, two_qubit_gate_time=1e-6, init_logs=None):
        self.num_qubits = min(len(nv_memory), max_qubits) if max_qubits else len(nv_memory)
        if self.num_qubits < 1:
            raise ValueError(f"At least 1 qubit required, got {self.num_qubits}")
        if init_logs:
            init_logs.add_row(f"[bold cyan]Initializing NV backend with {self.num_qubits} qubits at {TEMPERATURE:.1f} K ⚙️[/]")
        self.name = 'nv_diamond'
        self.target = Target(num_qubits=self.num_qubits)
        for gate in ['h', 'x', 'rz', 'rx', 'ry']:
            self.target.add_instruction(
                Gate(name=gate, num_qubits=1, params=[1] if gate in ['rz', 'rx', 'ry'] else []),
                {(i,): InstructionProperties(duration=gate_time, error=0.001) for i in range(self.num_qubits)}
            )
        for gate in ['cx', 'cz', 'cp']:
            params = [1] if gate == 'cp' else []
            self.target.add_instruction(
                Gate(name=gate, num_qubits=2, params=params),
                {(i, j): InstructionProperties(duration=two_qubit_gate_time, error=0.01)
                 for i in range(self.num_qubits) for j in range(self.num_qubits) if i != j}
            )
        self.target.add_instruction(
            Gate(name='measure', num_qubits=1, params=[]),
            {(i,): InstructionProperties(duration=1e-6, error=0.05) for i in range(self.num_qubits)}
        )
        noise_model = NoiseModel()
        for q in range(self.num_qubits):
            t1 = nv_memory[q]['T1']
            t2 = nv_memory[q]['T2']
            readout_err = nv_memory[q]['readout_err']
            noise_model.add_quantum_error(
                thermal_relaxation_error(t1, t2, gate_time), ['rx', 'ry', 'h', 'x', 'rz'], [q]
            )
            for j in range(self.num_qubits):
                if j != q:
                    error_q = thermal_relaxation_error(t1, t2, two_qubit_gate_time)
                    error_j = thermal_relaxation_error(nv_memory[j]['T1'], nv_memory[j]['T2'], two_qubit_gate_time)
                    for gate in ['cx', 'cz', 'cp']:
                        two_qubit_error = error_q.tensor(error_j)
                        noise_model.add_quantum_error(two_qubit_error, gate, [q, j])
            noise_model.add_readout_error(
                ReadoutError([[1 - readout_err, readout_err], [readout_err, 1 - readout_err]]), [q]
            )
        self.noise_model = noise_model
        self.backend = AerSimulator()
        if init_logs:
            init_logs.add_row("[green]NV backend initialized successfully ✅[/]")

    def run(self, circuit, shots=1024, sim_logs=None, transpile_logs=None, progress=None):
        """Run a quantum circuit with a progress bar.

        Args:
            circuit (QuantumCircuit): Circuit to execute.
            shots (int): Number of shots (default: 1024).
            sim_logs (rich.table.Table, optional): Table to log simulation messages.
            transpile_logs (rich.table.Table, optional): Table to log transpilation messages.
            progress (rich.progress.Progress, optional): Progress bar for execution.

        Returns:
            Result: Qiskit simulation result.
        """
        start_time = time.perf_counter()
        if sim_logs:
            sim_logs.add_row(f"[bold magenta]Running circuit with {shots} shots on NV backend 🔬[/]")
        if transpile_logs:
            transpile_logs.add_row(f"[green]Initial circuit: {circuit.num_qubits} qubits, depth={circuit.depth()}, gates={circuit.count_ops()}[/]")
        t_circ = transpile(circuit, self.backend, optimization_level=2, target=self.target)
        if transpile_logs:
            transpile_logs.add_row(f"[green]Circuit transpiled with optimization level 2[/]")
            transpile_logs.add_row(f"[green]Transpiled circuit: {t_circ.num_qubits} qubits, depth={t_circ.depth()}, gates={t_circ.count_ops()}[/]")
        task = None
        if progress:
            task = progress.add_task("Running simulation...", total=100)
        job = self.backend.run(t_circ, shots=shots, noise_model=self.noise_model)
        if sim_logs:
            sim_logs.add_row("[magenta]Simulation job submitted[/]")
        if progress:
            for _ in range(100):
                progress.update(task, advance=1)
                time.sleep(0.02)
        result = job.result()
        if sim_logs:
            sim_logs.add_row("[green]Simulation completed ✅[/]")
        pulse_data = simulate_laser_pulse(hardware_logs=sim_logs)
        if sim_logs:
            sim_logs.add_row(f"[green]Simulated laser pulse: {pulse_data['pulse']} (ADC), timestamp {pulse_data['timestamp']} 🔆[/]")
            sim_logs.add_row(f"[green]Simulated fluorescence: {np.random.poisson(1e4 * self.num_qubits * FLUORESCENCE_FACTOR)} photons/s at {TEMPERATURE:.1f} K[/]")
            sim_logs.add_row(f"[green]Circuit execution took {(time.perf_counter() - start_time)*1e3:.2f} ms[/]")
        return result

def simulate_odmr(frequency_sweep=np.linspace(2.8, 2.9, 200), B_field=1e-3, odmr_logs=None, progress=None):
    """Simulate ODMR with hyperfine interaction and plot results.

    Args:
        frequency_sweep (np.ndarray): Array of frequencies in GHz (default: 2.8-2.9 GHz, 200 points).
        B_field (float): Magnetic field strength in Tesla (default: 1e-3).
        odmr_logs (rich.table.Table, optional): Table to log messages.
        progress (rich.progress.Progress, optional): Progress bar for frequency sweep.

    Returns:
        tuple: (frequency_sweep, contrast) - Frequencies and ODMR contrast values.
    """
    start_time = time.perf_counter()
    if odmr_logs:
        odmr_logs.add_row(f"[bold cyan]Simulating ODMR at {TEMPERATURE:.1f} K 📡[/]")
    D_ang = D * 2 * np.pi
    gamma_e_ang = gamma_e * 2 * np.pi
    A_ang = A_hf * 2 * np.pi
    Q_ang = Q_n * 2 * np.pi
    Sz_e = tensor(jmat(1, 'z'), qeye(3))
    Sx_e = tensor(jmat(1, 'x'), qeye(3))
    Iz_n = tensor(qeye(3), jmat(1, 'z'))
    H0 = D_ang * Sz_e**2 + gamma_e_ang * B_field * Sz_e + A_ang * Sz_e * Iz_n + Q_ang * Iz_n**2
    times = np.linspace(0, 1e-6, 100)
    rho0_e = ket2dm(basis(3, 1))
    rho0_n = (fock_dm(3, 0) + fock_dm(3, 1) + fock_dm(3, 2)) / 3
    rho0 = tensor(rho0_e, rho0_n)
    proj_ms0 = tensor(ket2dm(basis(3, 1)), qeye(3))
    contrast = []
    task = None
    if progress:
        task = progress.add_task("Scanning ODMR frequencies...", total=len(frequency_sweep))
    for f_ghz in frequency_sweep:
        omega = f_ghz * 1e9 * 2 * np.pi
        drive_strength = 0.01 * omega
        H_drive = H0 + drive_strength * Sx_e
        result = mesolve(H_drive, rho0, times, [], [proj_ms0])
        contrast.append(result.expect[0][-1])
        if progress:
            progress.update(task, advance=1)
    contrast = np.array(contrast)
    if odmr_logs:
        odmr_logs.add_row("[green]ODMR simulation completed ✅[/]")
        odmr_logs.add_row(f"[green]ODMR simulation took {(time.perf_counter() - start_time)*1e3:.2f} ms[/]")
        plt.figure(figsize=(8, 6))
        plt.plot(frequency_sweep, contrast, 'b-', label='ODMR Contrast')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Contrast')
        plt.title(f'ODMR Spectrum at {TEMPERATURE:.1f} K')
        plt.legend()
        plt.grid(True)
        plt.savefig('odmr_spectrum.png')
        plt.close()
        odmr_logs.add_row("[green]ODMR plot saved as odmr_spectrum.png 📊[/]")
    return frequency_sweep, contrast

def xy8_decoupling(circuit, qubit, quantum_logs=None):
    """Apply XY8 dynamical decoupling to a qubit.

    Args:
        circuit (QuantumCircuit): Circuit to apply decoupling to.
        qubit (int): Qubit index to apply XY8.
        quantum_logs (rich.table.Table, optional): Table to log messages.

    Returns:
        QuantumCircuit: Updated circuit with XY8 sequence.
    """
    start_time = time.perf_counter()
    if quantum_logs:
        quantum_logs.add_row(f"[bold blue]Applying XY8 decoupling to qubit {qubit} at {TEMPERATURE:.1f} K 🛠️[/]")
    for _ in range(4):
        circuit.ry(np.pi/2, qubit)
        time.sleep(100e-9)
        circuit.rx(np.pi, qubit)
        time.sleep(100e-9)
    if quantum_logs:
        quantum_logs.add_row("[blue]XY8 sequence applied (8 pulses) ✅[/]")
        quantum_logs.add_row(f"[blue]XY8 decoupling took {(time.perf_counter() - start_time)*1e3:.2f} ms[/]")
    return circuit

def postprocess_shors(counts, num_control, a, N):
    """Post-process Shor's algorithm results to find period and factors.

    Args:
        counts (dict): Measurement counts from Qiskit result.
        num_control (int): Number of control qubits.
        a (int): Base number for modular exponentiation.
        N (int): Number to factor.

    Returns:
        tuple: (period r, (factor1, factor2)) or (None, (None, None)) if no valid period found.
    """
    phases = []
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if count < 0.01 * sum(counts.values()):
            continue
        decimal = int(state, 2)
        frac = Fraction(decimal, 2**num_control).limit_denominator(N)
        r = frac.denominator
        if r % 2 == 0 and pow(a, r, N) == 1:
            p = pow(a, r//2, N)
            factor1 = gcd(p - 1, N)
            factor2 = gcd(p + 1, N)
            if factor1 * factor2 == N and factor1 > 1 and factor2 > 1:
                return r, (factor1, factor2)
    return None, (None, None)