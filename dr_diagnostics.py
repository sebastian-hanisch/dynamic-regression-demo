"""Werkzeuge der Modellidentifikation nach Box und Jenkins: Autokorrelation (ACF), partielle Autokorrelation (PACF) und der Ljung-Box-Test auf unkorrelierte Fehler."""

import math

import numpy as np


def acf(x, nlags):
    """Autokorrelationen r_1..r_nlags (Lag 0 ist 1 und nicht enthalten)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return np.array([float(np.sum(x[k:] * x[:len(x) - k])) / denom for k in range(1, nlags + 1)])


def pacf(x, nlags):
    """Partielle Autokorrelationen 1..nlags aus der ACF (Durbin-Levinson)."""
    r = acf(x, nlags)
    out = np.zeros(nlags)
    phi = np.zeros(nlags)
    for k in range(nlags):
        if k == 0:
            out[0] = phi[0] = r[0]
            continue
        num = r[k] - np.dot(phi[:k], r[:k][::-1])
        den = 1.0 - np.dot(phi[:k], r[:k])
        a = num / den
        new = phi[:k] - a * phi[:k][::-1]
        phi[:k] = new
        phi[k] = a
        out[k] = a
    return out


def _gammainc_upper(a, x):
    """Regularisierte obere unvollständige Gammafunktion Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        term = total = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    """Überlebensfunktion der Chi-Quadrat-Verteilung: P(X > x)."""
    return _gammainc_upper(df / 2.0, x / 2.0)


def ljung_box(resid, lags, n_fitted_params=0):
    """Ljung-Box-Statistik Q und p-Wert für die Nullhypothese 'die Fehler sind unkorreliert' (Freiheitsgrade: Lags minus geschätzte ARMA-Parameter, mindestens 1)."""
    r = np.asarray(resid, dtype=float)
    n = len(r)
    rho = acf(r, lags)
    q = float(n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1))))
    return q, chi2_sf(q, max(lags - n_fitted_params, 1))
