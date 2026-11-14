"""
材料色散：把 n,k 数据表变成 tmm 能用的 n(λ) 函数。

约定：
    * 表格里波长用 nm（常见的 n,k 数据库都是 nm 或 μm，load_nk_csv 会自动判断）
    * 返回的 n(λ) 函数里，λ 用米（和 examples/ 里的写法一致）
    * n = n' + i*k，k > 0 表示吸收
"""

import numpy as np


def load_nk_csv(path, wl_unit="auto"):
    """
    读两列(λ,n)或三列(λ,n,k)的文本文件，分隔符支持逗号/制表符/分号/空格。
    返回 (wl_nm, n, k) 三个数组，已按波长升序排好。

    wl_unit: "nm" / "um" / "auto"
             auto 按数值猜：最大值 > 10 认为是 nm，否则认为是 μm
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = [p.strip() for p in line.replace(";", ",").replace("\t", ",")
                     .replace(" ", ",").split(",") if p.strip()]
            if len(parts) < 2:
                continue
            try:
                vals = [float(x) for x in parts[:3]]
            except ValueError:
                continue                      # 跳过表头
            rows.append((vals[0], vals[1], vals[2] if len(vals) > 2 else 0.0))

    if not rows:
        raise ValueError(f"{path} 里没读到数据")

    arr = np.array(rows, dtype=float)
    wl = arr[:, 0].copy()

    if wl_unit == "auto":
        wl_unit = "nm" if wl.max() > 10 else "um"
    if wl_unit == "um":
        wl = wl * 1000.0

    order = np.argsort(wl)
    wl = wl[order]
    if np.any(np.diff(wl) <= 0):
        raise ValueError(f"{path} 里有重复或乱序的波长点，先清理再读")

    return wl, arr[order, 1], arr[order, 2]


def make_nk_function(table, name=""):
    """把 load_nk_csv 的结果包装成 n(λ_m)->complex；超出数据范围直接报错，绝不外推"""
    wl_nm, n_arr, k_arr = table
    lo, hi = float(wl_nm.min()), float(wl_nm.max())

    def n_of_wl(wl_m):
        wl = wl_m * 1e9
        if not (lo <= wl <= hi):
            raise ValueError(f"{name} 的数据只在 {lo:.1f}-{hi:.1f} nm，请求 {wl:.1f} nm")
        return complex(np.interp(wl, wl_nm, n_arr), np.interp(wl, wl_nm, k_arr))

    return n_of_wl


def constant(n):
    """常数折射率，可以给复数"""
    n = complex(n)
    return lambda wl_m: n


def sellmeier(coefficients, name=""):
    """
    Sellmeier 型色散：n² = 1 + Σ B_i λ² / (λ² - C_i)
    coefficients: [(B1, C1), (B2, C2), ...]，C 的单位是 μm²
    适合 ZnO、SiO2 这类透明材料；有吸收的材料请用表格
    """
    def n_of_wl(wl_m):
        l2 = (wl_m * 1e6) ** 2               # μm²
        n2 = 1.0
        for b, c in coefficients:
            n2 += b * l2 / (l2 - c)
        return complex(np.sqrt(n2), 0.0)

    return n_of_wl
