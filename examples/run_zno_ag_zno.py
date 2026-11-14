"""
复现论文里的三层对称结构：空气 | ZnO | Ag | ZnO | 玻璃

准备工作：
    1. 从《附件材料.docx》里把 ZnO 和 Ag 的 n,k 各整理成一个文本文件，例如
           data/ZnO.csv     波长(nm), n, k
           data/Ag.csv      波长(nm), n, k
       （Excel 另存为 CSV，或者直接复制成三列文本都行；两列 λ,n 也支持，
        k 默认当 0）
    2. 在项目根目录运行：
           python -m examples.run_zno_ag_zno

    想先验证流程通不通，不接真实数据：
           python -m examples.run_zno_ag_zno --demo
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from src.dispersion import load_nk_csv, make_nk_function, constant
from src.spectrum import simulate_spectrum

# ---------------- 结构参数（按论文实际数值改） ----------------
D_ZNO = 30e-9        # 上下 ZnO 厚度（对称，所以相同）
D_AG = 10e-9         # Ag 厚度
N_GLASS = 1.52       # 基底；论文如果是石英就改 1.46
N_AIR = 1.0
WL_START, WL_STOP, N_WL = 300e-9, 1200e-9, 901
AG_SWEEP = [6e-9, 8e-9, 10e-9, 12e-9, 15e-9]


def build_materials(use_demo):
    """返回 (n_ZnO, n_Ag)，都是 n(λ)->复数的函数"""
    if use_demo:
        print("[demo] 用的是粗估值，只能看流程，不能当论文结果")
        return constant(2.0), constant(complex(0.15, 3.8))

    zno_path = os.path.join("data", "ZnO.csv")
    ag_path = os.path.join("data", "Ag.csv")
    for p in (zno_path, ag_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"找不到 {p}\n"
                f"请把附件材料里对应材料的 n,k 存成三列文本（波长nm, n, k）"
            )
    return (make_nk_function(load_nk_csv(zno_path), "ZnO"),
            make_nk_function(load_nk_csv(ag_path), "Ag"))


def avg_visible_T(wavelengths, T, lo=380e-9, hi=780e-9):
    """可见光平均透光率（等权）"""
    wl = np.asarray(wavelengths)
    T = np.asarray(T)
    mask = (wl >= lo) & (wl <= hi)
    return float(T[mask].mean())


def spectrum_of(n_ZnO, n_Ag, d_zno, d_ag, wavelengths):
    n_layers = [n_ZnO, n_Ag, n_ZnO]          # 三层，上下对称
    d_layers = [d_zno, d_ag, d_zno]
    return simulate_spectrum(n_layers, d_layers, wavelengths,
                             n_in=N_AIR, n_substrate=N_GLASS,
                             angle_incidence=0, polarization="TE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="用粗估常数跑通流程")
    p.add_argument("--save", default="results", help="图片输出目录")
    args = p.parse_args()

    n_ZnO, n_Ag = build_materials(args.demo)
    wavelengths = np.linspace(WL_START, WL_STOP, N_WL)
    wl_nm = wavelengths * 1e9

    # ---------- 主结构 ----------
    R, T = spectrum_of(n_ZnO, n_Ag, D_ZNO, D_AG, wavelengths)
    R = np.array(R)
    T = np.array(T)
    A = 1.0 - R - T

    print(f"ZnO {D_ZNO*1e9:.1f} nm / Ag {D_AG*1e9:.1f} nm / ZnO {D_ZNO*1e9:.1f} nm")
    print(f"可见光平均透光率 T_vis = {avg_visible_T(wavelengths, T):.4f}")
    print(f"最大 R+T = {np.max(R + T):.4f}   （必须 <= 1，超了说明模型或数据有问题）")
    print(f"峰值透射率 = {T.max():.4f} @ {wl_nm[T.argmax()]:.0f} nm")

    # ---------- Ag 厚度扫描 ----------
    sweep = {}
    for d in AG_SWEEP:
        _, Ts = spectrum_of(n_ZnO, n_Ag, D_ZNO, d, wavelengths)
        sweep[d] = (np.array(Ts), avg_visible_T(wavelengths, Ts))

    # ---------- 画图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(wl_nm, T, label="T", lw=1.8)
    ax.plot(wl_nm, R, label="R", lw=1.2)
    ax.plot(wl_nm, A, label="A (absorption)", lw=1.0, ls="--")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Ratio")
    ax.set_title(f"ZnO {D_ZNO*1e9:.0f} / Ag {D_AG*1e9:.0f} / ZnO {D_ZNO*1e9:.0f} on glass")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for d, (Ts, tavg) in sweep.items():
        ax.plot(wl_nm, Ts, label=f"Ag {d*1e9:.0f} nm, T_vis={tavg:.3f}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance")
    ax.set_title("Ag thickness sweep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.save, exist_ok=True)
    out = os.path.join(args.save, "zno_ag_zno.png")
    plt.savefig(out, dpi=160)
    print(f"图已保存到 {out}")
    plt.show()


if __name__ == "__main__":
    main()
