"""Orakel für die Schätzung der dynamischen Regression: (1) die bedingten Fehlerquadrate am gefundenen Parametersatz noch einmal direkt aus eta = z - X beta mit der skalaren ARMA-Rekursion (kein Filtern der Regressorspalten);
(2) kein besserer Parametersatz in der Nähe: Nelder-Mead über (phi, theta) mit eigenem konzentriertem beta darf die Fehlerquadrate höchstens um 0,2 % senken; (3) die Prognose aus den Daten vor dem Ursprung per Schleife
(Fehler mit e = 0 in der Zukunft, ggf. aufsummierte Differenzen, exp(f + sigma^2/2))."""

import numpy as np
import pytest

import dr_dynreg as R
import dr_scenario as S


def _eta_errors(z, X, beta, phi, theta, d):
    eta = z - X @ beta
    w = np.diff(eta) if d else eta
    e = np.zeros(len(w))
    for t in range(len(phi), len(w)):
        e[t] = w[t] - sum(ph * w[t - k] for k, ph in enumerate(phi, 1)) - sum(th * e[t - k] for k, th in enumerate(theta, 1) if t - k >= 0)
    return eta, w, e


@pytest.mark.parametrize("p,d,q", [(1, 0, 1), (0, 1, 1), (2, 0, 0)])
def test_css_value_and_local_optimality_against_an_independent_recursion(p, d, q):
    opt = pytest.importorskip("scipy.optimize")
    ser = S.generate(seed=11, shift=30)
    spec = R.Spec(True, True, 1, True, True, p, d, q, True)
    X, names = R.design(ser, spec)
    m = R.fit(ser.y[:730], X[:730], names, spec)
    z = np.log(np.maximum(ser.y[:730], 1.0))
    start = max(R.BURN - d, 0)
    _, _, e = _eta_errors(z, X[:730], np.array(m.beta), m.phi, m.theta, d)
    assert float(np.sum(e[max(start, p):] ** 2)) == pytest.approx(m.sse, rel=1e-7)
    Xd, zd = (np.diff(X[:730], axis=0), np.diff(z)) if d else (X[:730], z)

    def concentrated(u):
        phi, theta = u[:p], u[p:]
        if (p and np.max(np.abs(np.roots(np.r_[1, -phi]))) >= 1 - 1e-6) or (q and np.max(np.abs(np.roots(np.r_[1, theta]))) >= 1 - 1e-6):
            return 1e12
        E = np.zeros((len(zd), 1 + Xd.shape[1]))
        W = np.column_stack([zd, Xd])
        for t in range(p, len(zd)):
            E[t] = W[t] - sum(ph * W[t - k] for k, ph in enumerate(phi, 1)) - sum(th * E[t - k] for k, th in enumerate(theta, 1) if t - k >= 0)
        s0 = max(start, p)
        ey, ex = E[s0:, 0], E[s0:, 1:]
        beta = np.linalg.solve(ex.T @ ex + 1e-8 * np.eye(ex.shape[1]), ex.T @ ey)
        return float(np.sum((ey - ex @ beta) ** 2))

    u0 = np.array(list(m.phi) + list(m.theta))
    assert concentrated(u0) == pytest.approx(m.sse, rel=1e-6)
    best = opt.minimize(concentrated, u0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 120}).fun
    assert m.sse <= best * 1.002


@pytest.mark.parametrize("p,d,q,log", [(1, 0, 0, True), (0, 1, 1, False), (1, 1, 1, True), (2, 0, 1, False)])
def test_forecasts_equal_a_loop_that_sees_only_the_days_before_the_origin(p, d, q, log):
    ser = S.generate(seed=5, noise=0.2)
    spec = R.Spec(True, True, 1, True, True, p, d, q, log)
    X, names = R.design(ser, spec)
    m = R.fit(ser.y[:730], X[:730], names, spec)
    org, h = [740, 900, 1050], 9
    got = R.forecast_origins(m, R.filter_series(m, ser.y, X), X, org, h)
    z = np.log(np.maximum(ser.y, 1.0)) if log else ser.y
    for i, t in enumerate(org):
        eta, w, e = _eta_errors(z[:t], X[:t], np.array(m.beta), m.phi, m.theta, d)
        wf, ef = list(w), list(e)
        for _ in range(h):
            L = len(wf)
            wf.append(sum(ph * wf[L - k] for k, ph in enumerate(m.phi, 1)) + sum(th * ef[L - k] for k, th in enumerate(m.theta, 1)))
            ef.append(0.0)
        fut = np.array(wf[len(w):])
        f = X[t:t + h] @ np.array(m.beta) + ((eta[-1] + np.cumsum(fut)) if d else fut)
        assert got[i] == pytest.approx(np.maximum(np.exp(f + 0.5 * m.sigma2) if log else f, 0.0), rel=1e-8, abs=1e-8)
