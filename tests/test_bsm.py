import pytest
import numpy as np
from models.bsm import bsm_price

def test_bsm_call_intrinsic():
    # At expiry, T=0, call should be max(S-K, 0)
    S = 110.0
    K = 100.0
    T = 0.0
    r = 0.05
    sigma = 0.20
    price = bsm_price(S, K, T, r, sigma, "call")
    assert np.isclose(price, 10.0)

def test_bsm_put_intrinsic():
    S = 90.0
    K = 100.0
    T = 0.0
    r = 0.05
    sigma = 0.20
    price = bsm_price(S, K, T, r, sigma, "put")
    assert np.isclose(price, 10.0)

def test_put_call_parity():
    S = 100.0
    K = 105.0
    T = 0.5
    r = 0.03
    sigma = 0.25
    
    call = bsm_price(S, K, T, r, sigma, "call")
    put = bsm_price(S, K, T, r, sigma, "put")
    
    # C - P = S - K * exp(-rT)
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, atol=1e-7)

def test_vectorisation():
    S = np.array([100.0, 110.0])
    K = 100.0
    T = 0.5
    r = 0.05
    sigma = 0.20
    prices = bsm_price(S, K, T, r, sigma, "call")
    assert len(prices) == 2
    assert prices[1] > prices[0]
