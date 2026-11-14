import numpy as np
from src.tmm import tmm


def _resolve(n, wl_m):
    """折射率允许两种写法：常数（实数或复数），或者函数 n(波长)->复数"""
    if callable(n):
        return n(wl_m)
    return complex(n)


def simulate_spectrum(n_layers, d_layers, wavelengths, n_in, n_substrate, angle_incidence, polarization):
    """
    wavelengths 单位：米（与 examples/ 里 400e-9 的写法保持一致）

    n_layers 的每一项可以是：
        * 常数：1.52
        * 复数常数：0.18 + 2.9j
        * 函数：lambda wl_m: complex(...)，用来接色散材料（ZnO、Ag 等）
    n_in / n_substrate 同理。
    """
    R_spectrum = []
    T_spectrum = []

    for w1 in wavelengths:
        n_layers_wl = [_resolve(n, w1) for n in n_layers]
        r, R, t, T_power = tmm(
            n_substrate=_resolve(n_substrate, w1),
            n_layer=n_layers_wl,
            n_in=_resolve(n_in, w1),
            d_layer=d_layers,
            wavelength=w1,
            angle_incidence=angle_incidence,
            polarization=polarization,
        )
        R_spectrum.append(R)
        T_spectrum.append(T_power)

    return R_spectrum, T_spectrum
