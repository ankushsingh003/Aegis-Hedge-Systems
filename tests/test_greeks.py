import pytest
import numpy as np
from models.greeks import calculate_greeks

def test_delta_bounds():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    
    g_call = calculate_greeks(S, K, T, r, sigma, "call")
    g_put = calculate_greeks(S, K, T, r, sigma, "put")
    
    assert 0.0 < g_call.delta < 1.0
    assert -1.0 < g_put.delta < 0.0
    assert np.isclose(g_call.delta - g_put.delta, 1.0) # Delta Call - Delta Put = 1

def test_gamma_symmetry():
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    g_call = calculate_greeks(S, K, T, r, sigma, "call")
    g_put = calculate_greeks(S, K, T, r, sigma, "put")
    
    assert np.isclose(g_call.gamma, g_put.gamma)
    assert g_call.gamma > 0

def test_vega_positive():
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    g = calculate_greeks(S, K, T, r, sigma, "call")
    assert g.vega > 0
