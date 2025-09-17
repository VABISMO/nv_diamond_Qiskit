# File: tests/tests.py
# Description: Unit tests for NVQuantum library (nvquantum.py).
# Tests is_prime, mod_mult_gate, postprocess_shors, and simulate_odmr.
import pytest
import numpy as np
from nvquantum import is_prime, mod_mult_gate, postprocess_shors, simulate_odmr

def test_is_prime():
    """Test Rabin-Miller primality test."""
    assert is_prime(2) == True
    assert is_prime(15) == False
    assert is_prime(17) == True
    assert is_prime(1000003) == True  # Large prime
    assert is_prime(1000004) == False  # Large composite

def test_mod_mult_gate():
    """Test modular multiplication gate for Shor's algorithm."""
    U = mod_mult_gate(7, 15)
    assert U.label == "M_7 mod 15"
    with pytest.raises(ValueError):
        mod_mult_gate(5, 15)  # gcd(5, 15) != 1

def test_postprocess_shors():
    """Test post-processing of Shor's algorithm results."""
    counts = {'0000': 256, '1000': 256, '0100': 256, '1100': 256}  # r=4 for N=15, a=7
    r, factors = postprocess_shors(counts, 4, 7, 15)
    assert r == 4
    assert factors == (3, 5)

def test_simulate_odmr():
    """Test ODMR simulation output."""
    freq, contrast = simulate_odmr(np.linspace(2.8, 2.9, 10))
    assert len(freq) == len(contrast) == 10
    assert np.all(contrast >= 0)