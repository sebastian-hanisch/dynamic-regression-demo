"""Der Kern: Polynome, Differenzieren und Fehlerrekursion von Hand, Sonderfälle, Kreuzprobe gegen statsmodels (feste Parameter), Schätzung, kein Blick in die Zukunft."""

import warnings

import numpy as np
import pytest

import dr_constants as C
import dr_sarima as SAR
import dr_scenario as S


def _model(spec, phi=(), Phi=(), theta=(), Theta=(), mean=0.0, sigma2=1.0):
    return SAR.Model(spec, tuple(phi), tuple(Phi), tuple(theta), tuple(Theta), mean, 1.0, 100, sigma2)


def _forecast(model, y, origins, h, **kw):
    return SAR.forecast_origins(model, SAR.filter_series(model, y), np.asarray(origins), h, **kw)


# --- Polynome und Differenzieren ----------------------------------------------------------------------------------------------------------------


def test_polynomials_by_hand():
    A, B = SAR.ar_ma_polys(np.array([[0.5]]), np.array([[0.4]]), np.array([[0.3]]), np.array([[0.6]]))
    assert A.shape == (1, 8) and A[0, 0] == pytest.approx(0.5) and A[0, 6] == pytest.approx(0.4) and A[0, 7] == pytest.approx(-0.2) and np.count_nonzero(A) == 3
    assert B.shape == (1, 8) and B[0, 0] == pytest.approx(0.3) and B[0, 6] == pytest.approx(0.6) and B[0, 7] == pytest.approx(0.18) and np.count_nonzero(B) == 3
    A0, B0 = SAR.ar_ma_polys(np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((1, 0)))
    assert A0.shape == (1, 0) and B0.shape == (1, 0)


def test_pacf_transform_by_hand_and_always_stationary():
    c = SAR._pacf_to_coefs(np.array([[0.5, 0.3]]))
    assert c[0] == pytest.approx([0.5 * (1 - 0.3), 0.3])
    rng = np.random.default_rng(1)
    for k in (1, 2, 3):
        coefs = SAR._pacf_to_coefs(rng.uniform(-0.995, 0.995, size=(200, k)))
        for row in coefs:
            roots = np.roots(np.concatenate([-row[::-1], [1.0]]))               # 1 - c1 z - c2 z^2 ...
            assert np.all(np.abs(roots) > 1.0)


def test_differencing_by_hand():
    assert SAR.diff_coefs(0, 0).size == 0 and SAR.diff_coefs(1, 0).tolist() == [1.0]
    c = SAR.diff_coefs(1, 1)
    assert c[0] == 1.0 and c[6] == 1.0 and c[7] == -1.0 and np.count_nonzero(c) == 3
    z = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    assert SAR.difference(z, 1, 0).tolist() == np.diff(z).tolist() and SAR.difference(z, 2, 0).tolist() == [2.0, 2.0, 2.0]
    y = np.arange(20, dtype=float) ** 2
    assert SAR.difference(y, 0, 1) == pytest.approx(y[7:] - y[:-7]) and SAR.difference(y, 1, 1) == pytest.approx((y[8:] - y[7:-1]) - (y[1:-7] - y[:-8]))
    assert SAR.difference(z, 0, 0).tolist() == z.tolist()


def test_error_recursion_by_hand():
    A, B = np.array([[0.5]]), np.array([[0.4]])
    e, sse = SAR._css(np.array([1.0, 2.0, 3.0]), A, B)
    assert e[0].tolist() == pytest.approx([0.0, 1.5, 1.4]) and sse[0] == pytest.approx(1.5 ** 2 + 1.4 ** 2)
    e2, sse2 = SAR._css(np.array([1.0, 2.0, 3.0]), A, B, start_sse=2)
    assert sse2[0] == pytest.approx(1.4 ** 2)


def test_candidates_are_evaluated_independently():
    w = SAR.difference(S.generate(seed=2).y[:400].astype(float), 1, 1)
    A, B = SAR.ar_ma_polys(np.array([[0.2], [0.5]]), np.array([[0.1], [-0.2]]), np.array([[-0.4], [-0.6]]), np.array([[-0.8], [-0.5]]))
    _, both = SAR._css(w, A, B, 50)
    single = [SAR._css(w, A[i:i + 1], B[i:i + 1], 50)[1][0] for i in (0, 1)]
    assert both == pytest.approx(single)


# --- Sonderfälle und Prognose von Hand ------------------------------------------------------------------------------------------------------------


def test_simple_exponential_smoothing_is_arima_011_by_hand():
    m = _model(SAR.Spec(0, 1, 1), theta=(-0.6,))
    f = _forecast(m, np.array([10.0, 12.0, 11.0]), [3], 3)[0]
    # e_0 = 2, e_1 = -1 + 0,6 * 2 = 0,2, w_hat = -0,6 * 0,2 = -0,12 -> 10,88 (Niveau der einfachen Glättung mit alpha = 0,4 und Anfangsniveau 10), danach flach
    assert f == pytest.approx([10.88] * 3)


def test_naive_and_seasonal_naive_are_special_cases():
    y = S.generate(seed=4).y
    org = np.arange(800, 900, 13)
    assert _forecast(_model(SAR.Spec(0, 1, 0)), y, org, 5) == pytest.approx(np.repeat(y[org - 1][:, None], 5, axis=1))
    got = _forecast(_model(SAR.Spec(0, 0, 0, 0, 1, 0)), y, org, 10)
    assert got == pytest.approx(y[org[:, None] - 7 + (np.arange(10) % 7)[None, :]])


def test_second_difference_continues_a_straight_line():
    y = 5.0 + 2.0 * np.arange(50)
    f = _forecast(_model(SAR.Spec(0, 2, 0)), y, [40], 4)[0]
    assert f == pytest.approx(5.0 + 2.0 * np.arange(40, 44))


def test_autoregression_with_mean_reverts_to_the_mean():
    rng = np.random.default_rng(3)
    y = 50 + np.zeros(300)
    for t in range(1, 300):
        y[t] = 50 + 0.7 * (y[t - 1] - 50) + rng.normal()
    mean = float(y[:200].mean())
    m = _model(SAR.Spec(1, 0, 0), phi=(0.7,), mean=mean)
    f = _forecast(m, y, [250], 40)[0]
    assert f[0] == pytest.approx(mean + 0.7 * (y[249] - mean)) and abs(f[-1] - mean) < abs(f[0] - mean) * 0.01 + 1e-6


def test_log_models_return_the_exponential_with_bias_correction():
    y = S.generate(seed=1).y
    m = _model(SAR.Spec(0, 0, 0, 0, 1, 0, log=True), sigma2=0.04)
    med = _forecast(m, y, [800], 7, bias_correct=False)[0]
    mean = _forecast(m, y, [800], 7)[0]
    assert med == pytest.approx(y[800 - 7:800]) and mean == pytest.approx(y[793:800] * np.exp(0.02))


def test_forecasts_are_nonnegative():
    y = np.array([1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 9.0] * 20)
    f = _forecast(_model(SAR.Spec(0, 2, 0)), y, [100], 30)[0]
    assert np.all(f >= 0)


def test_forecast_does_not_depend_on_later_days():
    y = S.generate(seed=1).y.astype(float)
    m = SAR.fit(y[:C.FIRST_TEST], SAR.Spec(1, 0, 1, 1, 1, 1, log=True))
    y2 = y.copy()
    y2[850:] += 500
    a = _forecast(m, y, [800, 849], 14)
    b = _forecast(m, y2, [800, 849], 14)
    assert np.array_equal(a, b) and not np.array_equal(_forecast(m, y, [851], 14), _forecast(m, y2, [851], 14))


# --- Kreuzprobe gegen statsmodels (feste Parameter) ------------------------------------------------------------------------------------------------

CASES = [(SAR.Spec(0, 1, 1), dict(theta=[-0.6])), (SAR.Spec(1, 0, 1, 1, 1, 1), dict(phi=[0.5], theta=[-0.4], Phi=[0.3], Theta=[-0.6])), (SAR.Spec(2, 1, 1, 0, 1, 1), dict(phi=[0.3, -0.2], theta=[-0.5], Theta=[-0.7])),
         (SAR.Spec(1, 1, 2, 1, 1, 1), dict(phi=[0.4], theta=[-0.5, 0.2], Phi=[-0.3], Theta=[-0.6])), (SAR.Spec(0, 2, 2), dict(theta=[-1.0, 0.3])), (SAR.Spec(1, 0, 0), dict(phi=[0.6]))]


@pytest.mark.parametrize("spec,pr", CASES, ids=[c[0].label for c in CASES])
def test_forecast_matches_statsmodels_for_fixed_parameters(spec, pr):
    ARIMA = pytest.importorskip("statsmodels.tsa.arima.model").ARIMA
    y = S.generate(seed=3).y.astype(float)[:900]
    phi, Phi, theta, Theta = (list(pr.get(k, [])) for k in ("phi", "Phi", "theta", "Theta"))
    mean = float(SAR.difference(y[:C.FIRST_TEST], spec.d, spec.D).mean()) if spec.d + spec.D == 0 else 0.0
    m = _model(spec, phi, Phi, theta, Theta, mean)
    mine = _forecast(m, y, [900], 14)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trend = "n" if spec.d + spec.D > 0 else "c"
        mod = ARIMA(y, order=(spec.p, spec.d, spec.q), seasonal_order=(spec.P, spec.D, spec.Q, 7), trend=trend)
        vals = {"sigma2": 1.0}
        vals.update({f"ar.L{i}": v for i, v in enumerate(phi, 1)})
        vals.update({f"ma.L{i}": v for i, v in enumerate(theta, 1)})
        vals.update({f"ar.S.L{7 * i}": v for i, v in enumerate(Phi, 1)})
        vals.update({f"ma.S.L{7 * i}": v for i, v in enumerate(Theta, 1)})
        if trend == "c":
            vals["const"] = mean
        theirs = np.asarray(mod.smooth([vals[n] for n in mod.param_names]).forecast(14))
    assert mine == pytest.approx(theirs, abs=1e-6)


# --- Schätzung ----------------------------------------------------------------------------------------------------------------------------------


def test_fit_recovers_a_simulated_arma_process():
    rng = np.random.default_rng(7)
    n = 3000
    e = rng.normal(size=n + 50)
    x = np.zeros(n + 50)
    for t in range(1, n + 50):
        x[t] = 0.6 * x[t - 1] + e[t] + 0.3 * e[t - 1]
    m = SAR.fit(x[50:] + 100.0, SAR.Spec(1, 0, 1), burn=50)
    assert m.phi[0] == pytest.approx(0.6, abs=0.07) and m.theta[0] == pytest.approx(0.3, abs=0.09) and m.mean == pytest.approx(100.0, abs=1.0) and m.sigma2 == pytest.approx(1.0, abs=0.1)


def test_fit_is_deterministic_and_feasible():
    y = S.generate(seed=2).y[:C.FIRST_TEST]
    spec = SAR.Spec(2, 1, 2, 1, 1, 1, log=True)
    m = SAR.fit(y, spec)
    assert SAR.fit(y, spec) == m
    for coefs in (m.phi, m.Phi):
        assert np.all(np.abs(np.roots(np.concatenate([-np.array(coefs)[::-1], [1.0]]))) > 1.0)
    for coefs in (m.theta, m.Theta):
        assert np.all(np.abs(np.roots(np.concatenate([np.array(coefs)[::-1], [1.0]]))) > 1.0)


@pytest.mark.parametrize("spec", [SAR.Spec(0, 1, 1), SAR.Spec(1, 1, 2), SAR.Spec(1, 0, 1, 1, 1, 1)], ids=lambda s: s.label)
def test_estimated_error_is_not_larger_than_at_the_statsmodels_estimate(spec):
    ARIMA = pytest.importorskip("statsmodels.tsa.arima.model").ARIMA
    y = S.generate(seed=3).y[:C.FIRST_TEST].astype(float)
    mine = SAR.fit(y, spec, burn=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y, order=(spec.p, spec.d, spec.q), seasonal_order=(spec.P, spec.D, spec.Q, 7), trend="n" if spec.d + spec.D > 0 else "c").fit()
    p, names = res.params, res.param_names
    phi = [float(p[names.index(f"ar.L{i}")]) for i in range(1, spec.p + 1)]
    theta = [float(p[names.index(f"ma.L{i}")]) for i in range(1, spec.q + 1)]
    Phi = [float(p[names.index(f"ar.S.L{7 * i}")]) for i in range(1, spec.P + 1)]
    Theta = [float(p[names.index(f"ma.S.L{7 * i}")]) for i in range(1, spec.Q + 1)]
    w = SAR.difference(y, spec.d, spec.D)
    mean = float(w.mean()) if spec.d + spec.D == 0 else 0.0
    A, B = SAR.ar_ma_polys(np.array([phi]), np.array([Phi]), np.array([theta]), np.array([Theta]))
    _, sse = SAR._css(w - mean, A, B)
    assert mine.sse <= sse[0] * 1.001


def test_aic_counts_and_formula():
    m = SAR.Model(SAR.Spec(1, 0, 1, 1, 1, 1), (0.1,), (0.1,), (0.1,), (0.1,), 0.0, 500.0, 500, 1.0)
    assert m.n_params == 5 and SAR.Model(SAR.Spec(1, 0, 0), (0.1,), (), (), (), 0.0, 1.0, 100, 1.0).n_params == 3
    assert m.aicc == pytest.approx(500 * np.log(500.0 / 500) + 10 + 2 * 5 * 6 / (500 - 6))


def test_aicc_is_comparable_across_differencing_orders():
    y = S.generate(seed=2).y[:C.FIRST_TEST]
    a = SAR.fit(y, SAR.Spec(1, 0, 1, 0, 1, 1, log=True))
    b = SAR.fit(y, SAR.Spec(1, 1, 1, 1, 1, 1, log=True))
    assert a.n_fit == b.n_fit == C.FIRST_TEST - SAR.BURN
