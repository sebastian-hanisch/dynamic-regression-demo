"""Der Kern der dynamischen Regression: Regressoren, gefilterte Kleinste Quadrate gegen numpy und statsmodels, ARMA-Fehler, Prognose mit bekannter Zukunft, kein Blick in die Zukunft."""

import warnings

import numpy as np
import pytest

import dr_constants as C
import dr_dynreg as R
import dr_sarima as SAR
import dr_scenario as S

FULL = R.Spec(True, True, 2, True, True)


def _spec(**kw):
    base = dict(week=False, trend=False, year_k=0, holiday=False, promo=False, p=0, d=0, q=0, log=False)
    base.update(kw)
    return R.Spec(**base)


# --- Regressoren ------------------------------------------------------------------------------------------------------------------------------


def test_design_columns_and_values():
    ser = S.generate(seed=1)
    X, names = R.design(ser, FULL)
    assert X.shape == (ser.n, 15) and names[0] == "Konstante" and names[1:7] == list(C.WEEKDAYS[1:]) and names[7] == "Trend (je Jahr)" and names[-3:] == ["Feiertag", "Tag nach Feiertag", "Aktion"]
    assert np.all(X[:, 0] == 1.0) and X[:7, 1:7].sum(axis=0).tolist() == [1.0] * 6 and X[0, 1:7].sum() == 0.0                    # Tag 0 ist ein Montag: Basis
    assert X[365, 7] == pytest.approx(1.0) and X[1, 8] == pytest.approx(np.sin(2 * np.pi / 365)) and X[100, 9] == pytest.approx(np.cos(2 * np.pi * 100 / 365)) and X[100, 10] == pytest.approx(np.sin(4 * np.pi * 100 / 365))
    assert np.array_equal(X[:, -3], ser.holiday) and np.array_equal(X[:, -1], ser.promo) and np.array_equal(X[1:, -2], ser.holiday[:-1]) and X[0, -2] == 0.0


def test_design_without_constant_for_differenced_errors_and_switches():
    ser = S.generate(seed=1)
    X, names = R.design(ser, R.Spec(True, True, 1, True, True, 0, 1, 1))
    assert "Konstante" not in names and X.shape[1] == 14 - 2
    X0, n0 = R.design(ser, _spec())
    assert X0.shape == (ser.n, 1) and n0 == ["Konstante"]
    assert R.Spec().label == "Wochentag+Trend+Jahr(2)+Feiertage+Aktionen" and _spec().label == "nur Konstante" and R.Spec(p=1, d=1, q=2).error_label == "(1,1,2)"


def test_yearly_regressors_have_period_365():
    ser = S.generate(seed=1)
    X, names = R.design(ser, _spec(year_k=1))
    assert X[10, 1] == pytest.approx(X[375, 1]) and X[10, 2] == pytest.approx(X[740, 2])


# --- Kleinste Quadrate -----------------------------------------------------------------------------------------------------------------------


def _toy(n=400, seed=1, noise=1.0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    X = np.stack([np.ones(n), x1, x2], axis=1)
    y = X @ np.array([2.0, 3.0, -1.5]) + noise * rng.normal(size=n)
    return y, X, ["Konstante", "x1", "x2"]


def test_ols_matches_numpy_and_statsmodels_standard_errors():
    y, X, names = _toy()
    m = R.fit(y, X, names, _spec(), burn=0)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.array(m.beta) == pytest.approx(ref, abs=1e-6)
    sm = pytest.importorskip("statsmodels.api")
    res = sm.OLS(y, X).fit()
    n, k = len(y), 3
    assert np.array(m.se) * np.sqrt(n / (n - k)) == pytest.approx(np.asarray(res.bse), rel=1e-5) and m.sse == pytest.approx(float(np.sum(res.resid ** 2)), rel=1e-8)
    assert m.n_fit == n and m.sigma2 == pytest.approx(m.sse / n)


def test_ols_burn_in_drops_the_first_rows():
    y, X, names = _toy()
    m = R.fit(y, X, names, _spec(), burn=60)
    assert np.array(m.beta) == pytest.approx(np.linalg.lstsq(X[60:], y[60:], rcond=None)[0], abs=1e-6) and m.n_fit == len(y) - 60


def test_exact_linear_data_is_recovered_exactly():
    x = np.arange(200.0)
    X = np.stack([np.ones(200), x], axis=1)
    m = R.fit(2.0 + 0.5 * x, X, ["Konstante", "x"], _spec(), burn=0)
    assert np.array(m.beta) == pytest.approx([2.0, 0.5], abs=1e-5) and m.sse < 1e-6


def test_filtered_regression_equals_filtering_the_regression_residuals():
    """e(z) - e(X) beta = e(z - X beta): die konzentrierte Fehlersumme ist die Fehlersumme der gefilterten Fehlerreihe."""
    rng = np.random.default_rng(4)
    n = 600
    X = np.stack([np.ones(n), rng.normal(size=n)], axis=1)
    eta = np.zeros(n)
    for t in range(1, n):
        eta[t] = 0.6 * eta[t - 1] + rng.normal()
    y = X @ np.array([5.0, 2.0]) + eta
    m = R.fit(y, X, ["Konstante", "x"], _spec(p=1))
    A, B = SAR.ar_ma_polys(np.array([m.phi]), np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((1, 0)))
    _, sse = SAR._css(y - X @ np.array(m.beta), A, B, max(R.BURN, 1))
    assert sse[0] == pytest.approx(m.sse, rel=1e-6)


def test_ar_errors_are_recovered_and_standard_errors_are_larger_than_ols():
    rng = np.random.default_rng(5)
    n = 3000
    X = np.stack([np.ones(n), np.cumsum(rng.normal(size=n)) / 10], axis=1)
    eta = np.zeros(n)
    for t in range(1, n):
        eta[t] = 0.7 * eta[t - 1] + rng.normal()
    y = X @ np.array([1.0, 2.0]) + eta
    ar = R.fit(y, X, ["Konstante", "x"], _spec(p=1))
    ols = R.fit(y, X, ["Konstante", "x"], _spec())
    assert ar.phi[0] == pytest.approx(0.7, abs=0.05) and ar.beta[1] == pytest.approx(2.0, abs=0.15) and ar.sigma2 == pytest.approx(1.0, abs=0.1) and ar.sigma2 < ols.sigma2 / 1.5


def test_fit_is_deterministic_and_feasible():
    ser = S.generate(seed=2)
    spec = R.Spec(True, True, 1, True, True, 2, 0, 1)
    X, names = R.design(ser, spec)
    m = R.fit(ser.y[:C.FIRST_TEST], X[:C.FIRST_TEST], names, spec)
    assert R.fit(ser.y[:C.FIRST_TEST], X[:C.FIRST_TEST], names, spec) == m
    assert np.all(np.abs(np.roots(np.concatenate([-np.array(m.phi)[::-1], [1.0]]))) > 1.0) and np.all(np.abs(np.roots(np.concatenate([np.array(m.theta)[::-1], [1.0]]))) > 1.0)


def test_aicc_and_parameter_count():
    ser = S.generate(seed=2)
    X, names = R.design(ser, _spec(week=True))
    m = R.fit(ser.y[:C.FIRST_TEST], X[:C.FIRST_TEST], names, _spec(week=True, p=1))
    k, n = 7 + 1 + 1, m.n_fit
    assert m.n_params == k and m.aicc == pytest.approx(n * np.log(m.sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


# --- Prognose ---------------------------------------------------------------------------------------------------------------------------------


def _fixed_model(spec, names, beta, phi=(), theta=(), sigma2=1.0):
    return R.Model(spec, tuple(names), tuple(beta), tuple(0.0 for _ in beta), tuple(phi), tuple(theta), 1.0, 100, sigma2)


def test_regression_forecast_by_hand():
    ser = S.generate(seed=1)
    spec = _spec(week=True)
    X, names = R.design(ser, spec)
    beta = [100.0, 5.0, 4.0, 3.0, 2.0, -40.0, -60.0]
    m = _fixed_model(spec, names, beta)
    fl = R.filter_series(m, ser.y, X)
    f = R.forecast_origins(m, fl, X, [801], 3)[0]                       # Tage 801, 802, 803: Do, Fr, Sa (Tag 0 = Mo)
    assert f == pytest.approx([100 + 3.0, 100 + 2.0, 100 - 40.0])


def test_arma_error_forecast_adds_the_error_part():
    ser = S.generate(seed=1)
    spec = _spec(p=1)
    X, names = R.design(ser, spec)
    m = _fixed_model(spec, names, [float(ser.y.mean())], phi=(0.5,))
    fl = R.filter_series(m, ser.y, X)
    f = R.forecast_origins(m, fl, X, [800], 3)[0]
    eta = ser.y - ser.y.mean()
    assert f == pytest.approx([max(ser.y.mean() + 0.5 * eta[799], 0), max(ser.y.mean() + 0.25 * eta[799], 0), max(ser.y.mean() + 0.125 * eta[799], 0)])


def test_differenced_errors_keep_the_last_level():
    ser = S.generate(seed=1)
    spec = _spec(d=1)
    X, names = R.design(ser, spec)
    assert names == [] and X.shape == (ser.n, 0)
    m = _fixed_model(spec, names, [])
    f = R.forecast_origins(m, R.filter_series(m, ser.y, X), X, [800], 4)[0]
    assert f == pytest.approx([ser.y[799]] * 4)                          # Irrfahrt ohne Regressor = naiv


def test_log_models_return_the_exponential_with_bias_correction():
    ser = S.generate(seed=1)
    spec = _spec(log=True)
    X, names = R.design(ser, spec)
    m = _fixed_model(spec, names, [np.log(100.0)], sigma2=0.04)
    f = R.forecast_origins(m, R.filter_series(m, ser.y, X), X, [800], 2)[0]
    assert f == pytest.approx([100.0 * np.exp(0.02)] * 2)


def test_unknown_promotions_are_set_to_zero_in_the_future_only():
    ser = S.generate(seed=1, events=1.0)
    spec = _spec(promo=True)
    X, names = R.design(ser, spec)
    m = _fixed_model(spec, names, [100.0, 30.0])
    fl = R.filter_series(m, ser.y, X)
    org = np.array([int(d) - 2 for d in np.flatnonzero(ser.promo[C.FIRST_TEST:]) + C.FIRST_TEST][:1])
    known = R.forecast_origins(m, fl, X, org, 5)[0]
    unknown = R.forecast_origins(m, fl, X, org, 5, promo_known=False)[0]
    assert unknown == pytest.approx([100.0] * 5) and known.max() == pytest.approx(130.0) and (known > 100.0).any()


def test_forecast_does_not_depend_on_later_days():
    ser = S.generate(seed=1)
    X, names = R.design(ser, FULL)
    m = R.fit(ser.y[:C.FIRST_TEST], X[:C.FIRST_TEST], names, R.Spec(True, True, 2, True, True, 1, 0, 0))
    y2 = ser.y.copy()
    y2[850:] += 500
    a = R.forecast_origins(m, R.filter_series(m, ser.y, X), X, [800, 849], 14)
    b = R.forecast_origins(m, R.filter_series(m, y2, X), X, [800, 849], 14)
    c = R.forecast_origins(m, R.filter_series(m, y2, X), X, [851], 14)
    assert np.array_equal(a, b) and not np.array_equal(R.forecast_origins(m, R.filter_series(m, ser.y, X), X, [851], 14), c)


# --- Kreuzprobe gegen statsmodels (feste Parameter) ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("p,d,q,phi,theta", [(1, 0, 1, (0.6,), (-0.3,)), (0, 0, 0, (), ()), (1, 1, 1, (0.4,), (-0.5,)), (0, 1, 1, (), (-0.6,))])
def test_forecast_matches_statsmodels_regression_with_arima_errors(p, d, q, phi, theta):
    ARIMA = pytest.importorskip("statsmodels.tsa.arima.model").ARIMA
    ser = S.generate(seed=3)
    spec = R.Spec(True, True, 1, True, True, p, d, q, False)
    X, names = R.design(ser, spec)
    rng = np.random.default_rng(2)
    beta = rng.normal(scale=0.5, size=len(names))
    beta[0] = 100.0 if d == 0 else beta[0]
    m = _fixed_model(spec, names, beta, phi, theta)
    T = 900
    mine = R.forecast_origins(m, R.filter_series(m, ser.y, X), X, [T], 14)[0]
    mine_raw = mine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ARIMA(ser.y[:T].astype(float), exog=X[:T], order=(p, d, q), trend="n")
        vals = dict(zip(mod.param_names, list(beta) + list(phi) + list(theta) + [1.0]))
        theirs = np.asarray(mod.smooth([vals[n] for n in mod.param_names]).forecast(14, exog=X[T:T + 14]))
    assert mine_raw == pytest.approx(np.maximum(theirs, 0.0), abs=1e-5)
