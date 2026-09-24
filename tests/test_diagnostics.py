"""ACF, PACF, Ljung-Box und die Chi-Quadrat-Verteilung: von Hand und gegen statsmodels/scipy."""

import numpy as np
import pytest

import dr_diagnostics as D


def _ar1(phi, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def test_acf_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = D.acf(x, 2)
    assert r.tolist() == pytest.approx([0.25, -0.3])                     # Mittel 2,5; Nenner 5; Zähler 1,25 und -1,5
    xm = x - x.mean()
    assert r[0] == pytest.approx(np.sum(xm[1:] * xm[:-1]) / np.sum(xm ** 2)) and r[1] == pytest.approx(np.sum(xm[2:] * xm[:-2]) / np.sum(xm ** 2))


def test_acf_of_an_ar1_process_decays_geometrically_and_pacf_cuts_off():
    x = _ar1(0.7)
    r, p = D.acf(x, 5), D.pacf(x, 5)
    assert r == pytest.approx(0.7 ** np.arange(1, 6), abs=0.05) and p[0] == pytest.approx(0.7, abs=0.04) and np.all(np.abs(p[1:]) < 0.06)


def test_acf_and_pacf_match_statsmodels():
    st = pytest.importorskip("statsmodels.tsa.stattools")
    x = _ar1(0.5, n=800, seed=3) + np.sin(np.arange(800) / 7.0)
    assert D.acf(x, 20) == pytest.approx(st.acf(x, nlags=20, fft=False)[1:], abs=1e-10)
    assert D.pacf(x, 20) == pytest.approx(st.pacf(x, nlags=20, method="ywm")[1:], abs=1e-8)


@pytest.mark.parametrize("x,df", [(0.5, 1), (3.84, 1), (10.0, 5), (21.2, 10), (5.0, 14), (60.0, 14), (1.0, 30)])
def test_chi2_survival_function_matches_scipy(x, df):
    stats = pytest.importorskip("scipy.stats")
    assert D.chi2_sf(x, df) == pytest.approx(float(stats.chi2.sf(x, df)), rel=1e-9, abs=1e-14)
    assert D.chi2_sf(0.0, df) == 1.0


def test_ljung_box_matches_statsmodels_and_detects_autocorrelation():
    diag = pytest.importorskip("statsmodels.stats.diagnostic")
    x = _ar1(0.4, n=600, seed=5)
    q, p = D.ljung_box(x, 14, 3)
    ref = diag.acorr_ljungbox(x, lags=[14], model_df=3, return_df=True)
    assert q == pytest.approx(float(ref["lb_stat"].iloc[0]), rel=1e-9) and p == pytest.approx(float(ref["lb_pvalue"].iloc[0]), rel=1e-6)
    white = np.random.default_rng(2).normal(size=600)
    assert D.ljung_box(white, 14)[1] > 0.05 and p < 0.01
