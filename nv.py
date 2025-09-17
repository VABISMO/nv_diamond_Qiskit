# File: qnv.py
# Description: Command-line interface for NVQuantum, a simulated NV diamond quantum computer.
# Runs Shor's algorithm with a rich CLI, colorful tables, and progress bars.
# Imports core functionality from nvquantum.py.
# Notes:
# - Requires nvquantum, rich, pyfiglet, numpy, matplotlib, pytest.
# - Features: Interactive menu, CLI arguments, rich tables, progress bars for alignment, ODMR, and Shor's execution.
import argparse
import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from pyfiglet import Figlet
from nvquantum import *

# Set up rich console
console = Console()
figlet = Figlet(font='slant')
banner = figlet.renderText("NVQuantum")
console.print(Panel(banner, title="[bold green]🚀 NVQuantum (77 K) 🚀[/]", style="bold green", border_style="bold green"))

# Set up logging
import logging
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

def parse_args():
    """Parse command-line arguments and display help in a rich table.

    Returns:
        argparse.Namespace: Parsed arguments with num_qubits, shots, galvo_area, freq_range, and N.
    """
    parser = argparse.ArgumentParser(
        description="[bold green]🚀 NVQuantum: Simulated NV Diamond Quantum Computer[/]",
        epilog="[bold cyan]Examples:[/] python qnv.py --num-qubits 20 --shots 1024 --N 21"
    )
    parser.add_argument("--num-qubits", type=int, default=8, help="[cyan]Number of qubits (e.g., 20, default 8).[/]")
    parser.add_argument("--shots", type=int, default=1024, help="[cyan]Shots: number of circuit runs for statistics (1-10000, default 1024).[/]")
    parser.add_argument("--galvo-area", type=float, nargs=2, default=[10, 10], help="[cyan]Galvo area (µm, e.g., 10 10).[/]")
    parser.add_argument("--freq-range", type=float, nargs=2, default=[2.8, 2.9], help="[cyan]ODMR range (GHz, e.g., 2.8 2.9).[/]")
    parser.add_argument("--N", type=int, default=15, help="[cyan]Number to factor for Shor's algorithm (e.g., 21, default 15).[/]")

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

def show_menu():
    """Display an interactive menu for configuring the simulation.

    Returns:
        dict: Configuration with mode, num_qubits, shots, galvo_area, freq_range, and N.
    """
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

def main():
    """Main execution function for NVQuantum CLI.

    Runs NV center alignment, Shor's algorithm, ODMR simulation, and XY8 decoupling with rich UI.
    """
    args = parse_args()
    config = show_menu() if args.num_qubits == 8 and args.shots == 1024 and args.N == 15 else vars(args)
    num_qubits = config["num_qubits"]
    shots = config["shots"]
    galvo_area = tuple(config["galvo_area"])
    freq_range = config["freq_range"]
    N = config["N"]
    galvo_area = (max(galvo_area[0], 5 + num_qubits), max(galvo_area[1], 5 + num_qubits))

    # Run automatic alignment
    with Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        nv_memory = auto_align(
            galvo_area=galvo_area,
            camera_shape=(512, 512),
            max_retries=5,
            num_qubits=num_qubits,
            hardware_logs=hardware_logs,
            progress=progress
        )
    console.print(Panel(hardware_logs, title="[yellow]Hardware Logs[/]", border_style="yellow"))

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
    a = select_coprime(N)
    quantum_logs.add_row(f"[bold blue]Creating Shor's circuit for N={N}, a={a} 🧮[/]")
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
    with Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        backend = NVBackend(nv_memory, init_logs=init_logs)
        console.print(Panel(init_logs, title="[cyan]Init Logs[/]", border_style="cyan"))
        result = backend.run(qc, shots=shots, sim_logs=sim_logs, transpile_logs=transpile_logs, progress=progress)
    console.print(Panel(transpile_logs, title="[green]Transpiler Logs[/]", border_style="green"))
    console.print(Panel(sim_logs, title="[magenta]Simulation Logs[/]", border_style="magenta"))
    counts = result.get_counts(qc)

    # Post-process Shor's results
    r, factors = postprocess_shors(counts, num_control, a, N)
    table = Table(
        title=f"[green]Shor's Results (N={N}, a={a}) 📈[/]",
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
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        freq, contrast = simulate_odmr(np.linspace(freq_range[0], freq_range[1], 200), odmr_logs=odmr_logs, progress=progress)
    resonance = freq[np.argmax(contrast)]
    console.print(Panel(odmr_logs, title="[cyan]ODMR Logs 📡[/]", border_style="cyan", padding=(1, 2)))
    console.print(Panel(f"[green]ODMR peak ~{resonance:.3f} GHz ✅[/]", title="[green]ODMR Result[/]", border_style="green", padding=(1, 2)))

    # Apply XY8 decoupling
    qc = xy8_decoupling(qc, 0, quantum_logs=quantum_logs)
    console.print(Panel(quantum_logs, title="[blue]Quantum Circuit Logs[/]", border_style="blue"))
    console.print(Panel(f"[green]XY8 decoupling completed at {TEMPERATURE:.1f} K 🛠️[/]", title="[green]Decoupling Result[/]", border_style="green", padding=(1, 2)))

if __name__ == "__main__":
    main()