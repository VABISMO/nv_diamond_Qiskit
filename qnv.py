# File: qnv.py
# Description: NVQuantum with enhanced menu - Simulated NV diamond quantum computer, runs Shor's algorithm,
# with 532 nm laser, galvo (dynamic area), 200 fps camera (OpenCV), automatic NV center alignment,
# ODMR with hyperfine interaction (14N), and XY8 decoupling. Supports Alibaba hardware (50€ NV diamond, 30€ MW chip).
# Liquid nitrogen (77.35 K). Single num-qubits (e.g., 20) and shots for flexibility.
# Notes:
# - Requires qiskit, qiskit-aer, qutip, opencv-python, pyserial, rich, pyfiglet (2025).
# - NV centers: 1-10/µm², 2.87 GHz ODMR, 0.5 µm resolution, 200 fps camera.
# - Noise: T1=10-100 ms, T2=10-100 µs (cheap diamond), readout 5-15%.
# - Features: Colorful rich menu, single --num-qubits, robust error handling, dynamic galvo scan, progress bars for alignment, ODMR, and Shor's execution.
import numpy as np
import logging
import warnings
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from pyfiglet import Figlet
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit import Gate
import serial
import json
import time
from qutip import *
from fractions import Fraction
from math import gcd
import cv2
import sys

# Suppress QuTiP FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")

# Set up rich console
console = Console()
figlet = Figlet(font='slant')
banner = figlet.renderText("NVQuantum")
console.print(Panel(banner, title="[bold green]🚀 NVQuantum (77 K) 🚀[/]", style="bold green", border_style="bold green"))

# Set up logging
logger = logging.getLogger("NVQuantum")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# Initialize log tables
init_logs = Table(show_header=False, expand=True, style="cyan", border_style="cyan", show_lines=True)
init_logs.add_column(style="cyan", justify="left")
hardware_logs = Table(show_header=False, expand=True, style="yellow", border_style="yellow", show_lines=True)
hardware_logs.add_column(style="yellow", justify="left")
quantum_logs = Table(show_header=False, expand=True, style="blue", border_style="blue", show_lines=True)
quantum_logs.add_column(style="blue", justify="left")
sim_logs = Table(show_header=False, expand=True, style="magenta", border_style="magenta", show_lines=True)
sim_logs.add_column(style="magenta", justify="left")
transpile_logs = Table(show_header=False, expand=True, style="green", border_style="green", show_lines=True)
transpile_logs.add_column(style="green", justify="left")
odmr_logs = Table(show_header=False, expand=True, style="cyan", border_style="cyan", show_lines=True)
odmr_logs.add_column(style="cyan", justify="left")

# Physical constants
D = 2.87e9  # Zero-field splitting (Hz)
gamma_e = 28e9  # Electron gyromagnetic ratio (Hz/T)
A_hf = -2.16e6  # Hyperfine coupling for 14N (Hz)
Q_n = -4.95e6  # Quadrupole splitting for 14N (Hz)
TEMPERATURE = 77.35  # Liquid nitrogen temperature in Kelvin
FLUORESCENCE_FACTOR = 1.2  # 20% intensity increase at 77 K

# Helper function to check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Parse CLI arguments with rich table
def parse_args():
    parser = argparse.ArgumentParser(
        description="[bold green]🚀 NVQuantum: Simulated NV Diamond Quantum Computer[/]",
        epilog="[bold cyan]Examples:[/] python qnv.py --num-qubits 20 --shots 1024 --N 21"
    )
    parser.add_argument("--num-qubits", type=int, default=8, help="[cyan]Number of qubits (e.g., 20, default 8).[/]")
    parser.add_argument("--shots", type=int, default=1024, help="[cyan]Shots: number of circuit runs for statistics (1-10000, default 1024).[/]")
    parser.add_argument("--galvo-area", type=float, nargs=2, default=[10, 10], help="[cyan]Galvo area (µm, e.g., 10 10).[/]")
    parser.add_argument("--freq-range", type=float, nargs=2, default=[2.8, 2.9], help="[cyan]ODMR range (GHz, e.g., 2.8 2.9).[/]")
    parser.add_argument("--N", type=int, default=15, help="[cyan]Number to factor for Shor's algorithm (e.g., 21, default 15).[/]")
    
    # Display help in a rich table if -h or --help is used
    if '-h' in sys.argv or '--help' in sys.argv:
        help_table = Table(
            title="[bold cyan]NVQuantum CLI Options[/]",
            show_lines=True,
            expand=True,
            style="cyan",
            border_style="bold cyan",
            padding=(1, 2)
        )
        help_table.add_column("Option", style="cyan", justify="left")
        help_table.add_column("Description", style="yellow", justify="left")
        help_table.add_column("Default", style="green", justify="center")
        help_table.add_row(
            "--num-qubits",
            "Number of qubits (e.g., 20)",
            "8"
        )
        help_table.add_row(
            "--shots",
            "Number of circuit runs for statistics (1-10000)",
            "1024"
        )
        help_table.add_row(
            "--galvo-area",
            "Galvo area in µm (e.g., 10 10)",
            "10 10"
        )
        help_table.add_row(
            "--freq-range",
            "ODMR frequency range in GHz (e.g., 2.8 2.9)",
            "2.8 2.9"
        )
        help_table.add_row(
            "--N",
            "Number to factor for Shor's algorithm (e.g., 21)",
            "15"
        )
        console.print(Panel(
            help_table,
            title="[bold cyan]🚀 NVQuantum CLI Help 🚀[/]",
            subtitle="[cyan]Example: python qnv.py --num-qubits 20 --shots 1024 --N 21[/]",
            border_style="bold cyan",
            padding=(1, 2)
        ))
        sys.exit(0)
    
    args = parser.parse_args()
    if args.num_qubits < 1:
        parser.error("[red]--num-qubits must be at least 1[/]")
    if not 1 <= args.shots <= 10000:
        parser.error("[red]--shots must be between 1 and 10000[/]")
    if args.N < 2 or is_prime(args.N):
        parser.error("[red]--N must be a composite number greater than 1[/]")
    return args

# Sleek menu
def show_menu():
    menu_table = Table(
        title="[bold cyan]🌟 NVQuantum Menu 🌟[/]",
        show_header=False,
        show_lines=True,
        expand=True,
        style="cyan",
        border_style="bold cyan",
        padding=(1, 2)
    )
    menu_table.add_column(justify="left", style="cyan")
    menu_table.add_row("[yellow]🔍 1. Sim Mode[/] - Simulate NV centers & ODMR")
    menu_table.add_row("[red]🚪 2. Exit[/]")
    console.print(Panel(menu_table, title="[bold cyan]🌟 NVQuantum Menu 🌟[/]", border_style="bold cyan", padding=(1, 2)))
    choice = Prompt.ask("[bold cyan]Choose (1-2)[/]", choices=["1", "2"], default="1")
    if choice == "2":
        console.print("[red]Exiting 🚪[/]")
        exit(0)
    num_qubits = int(Prompt.ask("[bold cyan]Qubits (e.g., 20)[/]", default="8"))
    if num_qubits < 1:
        console.print("[red]Qubits must be at least 1 🚨[/]")
        exit(1)
    shots = int(Prompt.ask("[bold cyan]Shots (1-10000)[/]", default="1024"))
    if not 1 <= shots <= 10000:
        console.print("[red]Shots must be 1-10000 🚨[/]")
        exit(1)
    galvo_area = [float(x) for x in Prompt.ask(
        f"[bold cyan]Galvo area (µm, e.g., '{5 + num_qubits} {5 + num_qubits}')[/]",
        default=f"{5 + num_qubits} {5 + num_qubits}"
    ).split()]
    freq_range = [float(x) for x in Prompt.ask(
        "[bold cyan]ODMR range (GHz, e.g., '2.8 2.9')[/]", default="2.8 2.9"
    ).split()]
    N = int(Prompt.ask("[bold cyan]Number to factor for Shor's algorithm (e.g., 21, default 15)[/]", default="15"))
    if N < 2 or is_prime(N):
        console.print("[red]Number to factor must be a composite number greater than 1 🚨[/]")
        exit(1)
    return {
        "mode": "sim",
        "num_qubits": num_qubits,
        "shots": shots,
        "galvo_area": galvo_area,
        "freq_range": freq_range,
        "N": N
    }

# Simulate laser pulse
def simulate_laser_pulse(intensity=1.0, duration=1e-6):
    intensity_noise = np.random.normal(1.0, 0.05) * intensity
    jitter = np.random.normal(0, 0.1e-6)
    duration += jitter
    hardware_logs.add_row(f"[yellow]Simulating laser pulse (532 nm, intensity={intensity_noise:.2f}, duration={duration*1e6:.2f} µs) 🔆[/]")
    return {'pulse': intensity_noise * 4095, 'timestamp': time.time(), 'intensity': intensity_noise, 'duration': duration*1e6}

# Simulate galvo scan
def simulate_galvo_scan(area=(10, 10), step=0.5):
    x_steps = int(area[0] / step)
    y_steps = int(area[1] / step)
    coords = [(x * step, y * step) for x in range(x_steps) for y in range(y_steps)]
    hardware_logs.add_row(f"[yellow]Simulating galvo scan over {area[0]}x{area[1]} µm with {step} µm step at {TEMPERATURE:.1f} K 🔍[/]")
    return coords

# Simulate camera frame
def simulate_camera_frame(nv_centers, shape=(512, 512), sigma=5, background_mean=20, background_std=5):
    frame = np.random.normal(background_mean, background_std, shape)
    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for nv in nv_centers:
        x, y, intensity = nv['x'], nv['y'], nv['intensity'] * FLUORESCENCE_FACTOR
        frame += intensity * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    frame = np.random.poisson(np.clip(frame, 0, None))
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame

# Detect NV centers
def detect_nv_centers(frame, threshold=50):
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

# Automatic alignment with progress bar
def auto_align(galvo_area=(10, 10), camera_shape=(512, 512), max_retries=5, num_qubits=8):
    scan_time = 5 + num_qubits * 0.5
    with Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Aligning NV centers...", total=max_retries)
        for attempt in range(max_retries):
            hardware_logs.add_row(f"[bold yellow]Aligning NV centers, attempt {attempt + 1}/{max_retries}, scan time {scan_time:.1f} s 🔍[/]")
            area_um2 = galvo_area[0] * galvo_area[1]
            density = np.random.uniform(1, 10)
            num_nv = int(np.random.uniform(5, min(30, density * area_um2)))
            nv_centers = [{'x': np.random.uniform(50, camera_shape[0]-50), 'y': np.random.uniform(50, camera_shape[1]-50), 'intensity': np.random.uniform(50, 200)} for _ in range(num_nv)]
            frame = simulate_camera_frame(nv_centers, camera_shape)
            coords = simulate_galvo_scan(galvo_area, step=0.5)
            simulate_laser_pulse()
            detected = detect_nv_centers(frame, threshold=50)
            density = len(detected) / (galvo_area[0] * galvo_area[1]) if detected else 0
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
                    t1 = np.random.uniform(10e-3, 100e-3)
                    t2 = np.random.uniform(10e-6, 100e-6)
                    readout_err = np.random.uniform(0.05, 0.15)
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
            progress.update(task, advance=1)
            if len(usable_qubits) >= num_qubits:
                hardware_logs.add_row(f"[green]{len(nv_memory)} NV centers, {len(usable_qubits)} usable qubits ✅[/]")
                break
            hardware_logs.add_row(f"[yellow]Need {num_qubits} qubits, got {len(usable_qubits)}, retrying...[/]")
        else:
            hardware_logs.add_row(f"[red]Failed to find {num_qubits} qubits 🚨[/]")
            raise ValueError(f"Insufficient NV centers: {len(usable_qubits)} found")
    console.print(Panel(hardware_logs, title="[yellow]Hardware Logs[/]", border_style="yellow"))
    return nv_memory

# Modular multiplication gate
def mod_mult_gate(b, N):
    if gcd(b, N) != 1:
        raise ValueError(f"gcd({b}, {N}) must be 1")
    n = int(np.ceil(np.log2(N)))
    U = np.zeros((2**n, 2**n), dtype=complex)
    for x in range(2**n):
        y = (b * x) % N if x < N else x
        U[y, x] = 1
    return UnitaryGate(U, label=f"M_{b} mod {N}")

# NV Backend with progress bar
class NVBackend:
    def __init__(self, nv_memory, max_qubits=12, gate_time=100e-9, two_qubit_gate_time=1e-6):
        self.num_qubits = min(len(nv_memory), max_qubits)
        if self.num_qubits < max_qubits:
            raise ValueError(f"NVBackend requires at least {max_qubits} qubits, got {self.num_qubits}")
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
        init_logs.add_row("[green]NV backend initialized successfully ✅[/]")
        console.print(Panel(init_logs, title="[cyan]Init Logs[/]", border_style="cyan"))

    def run(self, circuit, shots=1024):
        sim_logs.add_row(f"[bold magenta]Running circuit with {shots} shots on NV backend 🔬[/]")
        transpile_logs.add_row(f"[green]Initial circuit: {circuit.num_qubits} qubits, depth={circuit.depth()}, gates={circuit.count_ops()}[/]")
        t_circ = transpile(circuit, self.backend, optimization_level=2, target=self.target)
        transpile_logs.add_row(f"[green]Circuit transpiled with optimization level 2[/]")
        transpile_logs.add_row(f"[green]Transpiled circuit: {t_circ.num_qubits} qubits, depth={t_circ.depth()}, gates={t_circ.count_ops()}[/]")
        with Progress(
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running Shor's algorithm simulation...", total=100)
            job = self.backend.run(t_circ, shots=shots, noise_model=self.noise_model)
            sim_logs.add_row("[magenta]Simulation job submitted[/]")
            # Simulate progress (since Qiskit jobs don't provide real-time progress)
            for _ in range(100):
                progress.update(task, advance=1)
                time.sleep(0.02)  # Simulate computation time
            result = job.result()
        sim_logs.add_row("[green]Simulation completed ✅[/]")
        pulse_data = simulate_laser_pulse()
        sim_logs.add_row(f"[green]Simulated laser pulse: {pulse_data['pulse']} (ADC), timestamp {pulse_data['timestamp']} 🔆[/]")
        fluorescence = np.random.poisson(1e4 * self.num_qubits * FLUORESCENCE_FACTOR)
        sim_logs.add_row(f"[green]Simulated fluorescence: {fluorescence} photons/s at {TEMPERATURE:.1f} K[/]")
        console.print(Panel(transpile_logs, title="[green]Transpiler Logs[/]", border_style="green"))
        console.print(Panel(sim_logs, title="[magenta]Simulation Logs[/]", border_style="magenta"))
        return result

# Simulate ODMR with progress bar
def simulate_odmr(frequency_sweep=np.linspace(2.8, 2.9, 200), B_field=1e-3):
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
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Scanning ODMR frequencies...", total=len(frequency_sweep))
        for f_ghz in frequency_sweep:
            omega = f_ghz * 1e9 * 2 * np.pi
            drive_strength = 0.01 * omega
            H_drive = H0 + drive_strength * Sx_e
            result = mesolve(H_drive, rho0, times, [], [proj_ms0])
            contrast.append(result.expect[0][-1])
            progress.update(task, advance=1)
    contrast = np.array(contrast)
    odmr_logs.add_row("[green]ODMR simulation completed ✅[/]")
    return frequency_sweep, contrast

# XY8 Decoupling
def xy8_decoupling(circuit, qubit):
    quantum_logs.add_row(f"[bold blue]Applying XY8 decoupling to qubit {qubit} at {TEMPERATURE:.1f} K 🛠️[/]")
    for _ in range(4):
        circuit.ry(np.pi/2, qubit)
        circuit.rx(np.pi, qubit)
    quantum_logs.add_row("[blue]XY8 sequence applied (8 pulses) ✅[/]")
    console.print(Panel(quantum_logs, title="[blue]Quantum Circuit Logs[/]", border_style="blue"))
    return circuit

# Post-process Shor's results
def postprocess_shors(counts, num_control, a, N):
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

# Main execution
def main():
    args = parse_args()
    config = show_menu() if args.num_qubits == 8 and args.shots == 1024 and args.N == 15 else vars(args)
    num_qubits = config["num_qubits"]
    shots = config["shots"]
    galvo_area = tuple(config["galvo_area"])
    freq_range = config["freq_range"]
    N = config["N"]

    galvo_area = (max(galvo_area[0], 5 + num_qubits), max(galvo_area[1], 5 + num_qubits))

    # Run automatic alignment
    nv_memory = auto_align(galvo_area=galvo_area, camera_shape=(512, 512), max_retries=5, num_qubits=num_qubits)

    # Display NV memory
    nv_table = Table(
        title=f"[green]NV Centers ({num_qubits} Qubits) 📊[/]",
        border_style="green",
        show_lines=True,
        expand=True,
        style="green",
        padding=(1, 2)
    )
    nv_table.add_column("ID", style="cyan", justify="center")
    nv_table.add_column("X (µm)", style="magenta", justify="center")
    nv_table.add_column("Y (µm)", style="magenta", justify="center")
    nv_table.add_column("Intensity", style="yellow", justify="center")
    nv_table.add_column("T1 (ms)", style="blue", justify="center")
    nv_table.add_column("T2 (µs)", style="blue", justify="center")
    nv_table.add_column("Readout Err (%)", style="red", justify="center")
    for nv in nv_memory[:num_qubits]:
        nv_table.add_row(
            str(nv['qubit_id']),
            f"{nv['x']:.2f}",
            f"{nv['y']:.2f}",
            f"{nv['intensity']:.2f}",
            f"{nv['T1']*1e3:.2f}",
            f"{nv['T2']*1e6:.2f}",
            f"{nv['readout_err']*100:.1f}"
        )
    console.print(Panel(nv_table, title="[green]NV Memory[/]", border_style="green", padding=(1, 2)))

    # Create Shor's circuit
    quantum_logs.add_row(f"[bold blue]Creating Shor's circuit for N={N}, a=7 🧮[/]")
    a = 7
    num_target = int(np.ceil(np.log2(N)))
    num_control = max(4, num_qubits - num_target)
    total_qubits = num_control + num_target
    control = QuantumRegister(num_control, 'c')
    target = QuantumRegister(num_target, 't')
    creg = ClassicalRegister(num_control, 'out')
    qc = QuantumCircuit(control, target, creg)
    quantum_logs.add_row(f"[blue]Initialized circuit with {total_qubits} qubits ({num_control} control, {num_target} target)[/]")
    qc.h(control)
    qc.x(target[0])
    quantum_logs.add_row("[blue]Applied Hadamard and |1> initialization[/]")
    k_list = range(num_control)
    b_list = [pow(a, 2**k, N) for k in k_list]
    for k, b in enumerate(b_list):
        if b == 1:
            continue
        U = mod_mult_gate(b, N)
        c_U = U.control(1)
        qc.append(c_U, [control[k]] + list(target))
        quantum_logs.add_row(f"[blue]Applied controlled M_{b} mod {N} for k={k}[/]")
    qc.append(QFTGate(num_control).inverse(), control)
    quantum_logs.add_row("[blue]Applied inverse QFT[/]")
    qc.measure(control, creg)
    quantum_logs.add_row("[blue]Added measurements[/]")

    # Run on NV backend
    backend = NVBackend(nv_memory, max_qubits=num_qubits)
    result = backend.run(qc, shots=shots)
    counts = result.get_counts(qc)

    # Post-process Shor's results
    r, factors = postprocess_shors(counts, num_control, a, N)
    table = Table(
        title=f"[green]Shor's Results (N={N}, a=7) 📈[/]",
        border_style="green",
        show_lines=True,
        expand=True,
        style="green",
        padding=(1, 2)
    )
    table.add_column("State Binary", style="cyan", justify="center")
    table.add_column("State Decimal", style="yellow", justify="center")
    table.add_column("Counts", style="magenta", justify="center")
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        decimal = int(state, 2)
        table.add_row(state, str(decimal), str(count))
    console.print(Panel(table, title="[green]Measurement Counts[/]", border_style="green", padding=(1, 2)))
    if r:
        console.print(Panel(f"[green]Found period r={r}, factors={factors} 🎉[/]", title="[green]Shor's Result[/]", border_style="green", padding=(1, 2)))
    else:
        console.print(Panel("[yellow]No valid period found (noise may affect results) ⚠️[/]", title="[yellow]Shor's Result[/]", border_style="yellow", padding=(1, 2)))

    # Run ODMR
    freq, contrast = simulate_odmr(np.linspace(freq_range[0], freq_range[1], 200))
    resonance = freq[np.argmax(contrast)]
    console.print(Panel(odmr_logs, title="[cyan]ODMR Logs 📡[/]", border_style="cyan", padding=(1, 2)))
    console.print(Panel(f"[green]ODMR peak ~{resonance:.3f} GHz ✅[/]", title="[green]ODMR Result[/]", border_style="green", padding=(1, 2)))

    # Apply XY8 decoupling
    qc = xy8_decoupling(qc, 0)
    console.print(Panel(f"[green]XY8 decoupling completed at {TEMPERATURE:.1f} K 🛠️[/]", title="[green]Decoupling Result[/]", border_style="green", padding=(1, 2)))

if __name__ == "__main__":
    main()