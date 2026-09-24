"""Dynamische Regression: eine Regression auf bekannte Einflussgrößen (Wochentag, Trend, Jahresmuster, Feiertage, Aktionen) mit ARIMA-Fehlern.

    z_t = x_t' beta + eta_t,        phi(B) (1-B)^d eta_t = theta(B) e_t          (z = y oder log y)

Die Fehler eta_t dürfen autokorreliert sein (ARMA, auf Wunsch einmal differenziert). Geschätzt werden beta und die ARMA-Parameter gemeinsam durch Minimieren der bedingten Fehlerquadrate: für gegebene ARMA-Parameter ist das ein lineares
Regressionsproblem auf den mit dem ARMA-Filter gefilterten Größen (z und jede Spalte von x werden durch dieselbe Fehlerrekursion geschickt); beta folgt daher aus den Normalgleichungen ("konzentriert"), und nur die wenigen ARMA-Parameter werden gesucht
(fester Zufallsansatz, schrumpfende lokale Suche, alle Kandidaten gleichzeitig). Ohne ARMA-Terme ist das die gewöhnliche Kleinste-Quadrate-Regression.

Prognose: Regressionsteil mit den (für die Zukunft bekannten) Regressoren plus die ARIMA-Prognose der Fehler eta."""

from dataclasses import dataclass

import numpy as np

import dr_constants as C
import dr_sarima as SAR

M = C.SEASON_PERIOD
GROUPS = ("week", "trend", "year", "holiday", "promo")


@dataclass(frozen=True)
class Spec:
    week: bool = True
    trend: bool = True
    year_k: int = 2               # Anzahl der Fourier-Paare für das Jahresmuster (0 = keines)
    holiday: bool = True
    promo: bool = True
    p: int = 0
    d: int = 0
    q: int = 0
    log: bool = True

    @property
    def n_arma(self):
        return self.p + self.q

    @property
    def label(self):
        groups = [n for n, on in (("Wochentag", self.week), ("Trend", self.trend), (f"Jahr({self.year_k})", self.year_k > 0), ("Feiertage", self.holiday), ("Aktionen", self.promo)) if on]
        return "+".join(groups) if groups else "nur Konstante"

    @property
    def error_label(self):
        return f"({self.p},{self.d},{self.q})"


def design(series, spec):
    """Regressormatrix (n, K) für alle Tage und die Spaltennamen. Bei d = 0 steht die Konstante an erster Stelle; bei d = 1 entfällt sie (das Niveau trägt die Integration)."""
    n = series.n
    t = np.arange(n)
    cols, names = [], []
    if spec.d == 0:
        cols.append(np.ones(n)); names.append("Konstante")
    if spec.week:
        for dw in range(1, 7):
            cols.append((series.dow == dw).astype(float)); names.append(C.WEEKDAYS[dw])
    if spec.trend:
        cols.append(t / 365.0); names.append("Trend (je Jahr)")
    for k in range(1, spec.year_k + 1):
        cols.append(np.sin(2 * np.pi * k * (t % 365) / 365.0)); names.append(f"sin {k}")
        cols.append(np.cos(2 * np.pi * k * (t % 365) / 365.0)); names.append(f"cos {k}")
    if spec.holiday:
        after = np.roll(series.holiday, 1)
        after[0] = 0.0
        cols.append(series.holiday.astype(float)); names.append("Feiertag")
        cols.append(after.astype(float)); names.append("Tag nach Feiertag")
    if spec.promo:
        cols.append(series.promo.astype(float)); names.append("Aktion")
    X = np.stack(cols, axis=1) if cols else np.zeros((n, 0))
    return X, names


@dataclass(frozen=True)
class Model:
    spec: Spec
    names: tuple
    beta: tuple
    se: tuple
    phi: tuple
    theta: tuple
    sse: float
    n_fit: int
    sigma2: float

    @property
    def n_params(self):
        return len(self.beta) + self.spec.n_arma + 1

    @property
    def aicc(self):
        n, k = self.n_fit, self.n_params
        return float(n * np.log(self.sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))

    def coef(self, name):
        return self.beta[self.names.index(name)]


def _filter_many(W, A, Bc):
    """Bedingte Fehlerrekursion für viele Kandidaten (A, Bc: (nb, LA), (nb, LB)) auf mehreren Reihen W (m, T): Fehler (nb, m, T)."""
    nb, LA = A.shape
    LB = Bc.shape[1]
    m, T = W.shape
    Wp = np.concatenate([np.zeros((m, LA)), W], axis=1)
    if LA:
        idx = np.arange(T)[:, None] - np.arange(1, LA + 1)[None, :] + LA          # (T, LA)
        ar_part = np.einsum("nk,mtk->nmt", A, Wp[:, idx])
    else:
        ar_part = np.zeros((nb, m, T))
    E = np.zeros((nb, m, LB + T))
    for t in range(LA, T):
        c = t + LB
        ma = np.einsum("nk,nmk->nm", Bc, E[:, :, c - LB:c][:, :, ::-1]) if LB else 0.0
        E[:, :, c] = W[None, :, t] - ar_part[:, :, t] - ma
    return E[:, :, LB:]


def _solve(e, start, ridge=1e-8):
    """e: (nb, 1 + K, T) gefilterte Größen (Index 0: z). Kleinste Quadrate von e[0] auf e[1:] ab Index start: beta (nb, K) und SSE (nb,)."""
    ey, ex = e[:, 0, start:], e[:, 1:, start:]
    K = ex.shape[1]
    if K == 0:
        return np.zeros((e.shape[0], 0)), np.sum(ey ** 2, axis=1), np.zeros((e.shape[0], 0, 0))
    G = np.einsum("nkt,njt->nkj", ex, ex) + ridge * np.eye(K)[None]
    b = np.einsum("nkt,nt->nk", ex, ey)
    beta = np.linalg.solve(G, b[:, :, None])[:, :, 0]
    sse = np.sum(ey ** 2, axis=1) - np.einsum("nk,nk->n", b, beta)
    return beta, sse, G


FIT_STAGE1 = 1200
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 30
FIT_SEED = 20240926
BURN = 60


def _prepare(spec, y_train, X_train):
    z = np.log(np.maximum(y_train, 1.0)) if spec.log else np.asarray(y_train, dtype=float)
    if spec.d == 1:
        z, X = np.diff(z), np.diff(X_train, axis=0)
    else:
        X = X_train
    return np.concatenate([z[None, :], X.T], axis=0)


def fit(y_train, X_train, names, spec, seed=FIT_SEED, burn=BURN):
    """beta und ARMA-Parameter der Fehler auf den Trainingstagen schätzen (deterministisch: fester Seed der Suche)."""
    W = _prepare(spec, np.asarray(y_train, dtype=float), np.asarray(X_train, dtype=float))
    n_arma = spec.n_arma
    start = max(burn - spec.d, 0)
    rng = np.random.default_rng(seed)

    def evaluate(u):
        phi, _, theta, _ = SAR._split(SAR.Spec(spec.p, 0, spec.q), u)
        A, Bc = SAR.ar_ma_polys(phi, np.zeros((len(u), 0)), theta, np.zeros((len(u), 0)))
        with np.errstate(all="ignore"):
            e = _filter_many(W, A, Bc)
            beta, sse, G = _solve(e, max(start, A.shape[1]))
        return beta, np.where(np.isfinite(sse), sse, np.inf), G, phi, theta

    if n_arma == 0:
        u = np.zeros((1, 0))
        beta, sse, G, phi, theta = evaluate(u)
        best = (beta[0], float(sse[0]), G[0], (), ())
    else:
        U = rng.uniform(-SAR.R_MAX, SAR.R_MAX, size=(FIT_STAGE1, n_arma))
        _, S, _, _, _ = evaluate(U)
        order = np.argsort(S)[:FIT_TOP]
        top_u, top_s = U[order], S[order]
        for r in range(FIT_ROUNDS):
            scale = 0.2 * 0.55 ** r
            cand = np.clip(np.repeat(top_u, FIT_PER_START, axis=0) + scale * rng.normal(size=(FIT_TOP * FIT_PER_START, n_arma)), -SAR.R_MAX, SAR.R_MAX)
            _, cs, _, _, _ = evaluate(cand)
            pool_u, pool_s = np.vstack([top_u, cand]), np.concatenate([top_s, cs])
            order = np.argsort(pool_s)[:FIT_TOP]
            top_u, top_s = pool_u[order], pool_s[order]
        beta, sse, G, phi, theta = evaluate(top_u[:1])
        best = (beta[0], float(sse[0]), G[0], tuple(phi[0]), tuple(theta[0]))
    beta, sse, G, phi, theta = best
    n_fit = W.shape[1] - max(start, spec.p)
    sigma2 = sse / n_fit
    se = np.sqrt(np.diag(np.linalg.inv(G)) * sigma2) if len(beta) else np.zeros(0)
    return Model(spec, tuple(names), tuple(float(b) for b in beta), tuple(float(s) for s in se), tuple(phi), tuple(theta), sse, n_fit, sigma2)


@dataclass(frozen=True)
class Filtered:
    eta: np.ndarray           # Fehlerreihe z - x'beta (ganze Reihe)
    arma: object              # SAR.Model der Fehler
    filt: object              # SAR.Filtered


def filter_series(model, y, X):
    z = np.log(np.maximum(y, 1.0)) if model.spec.log else np.asarray(y, dtype=float)
    eta = z - X @ np.array(model.beta)
    arma = SAR.Model(SAR.Spec(model.spec.p, model.spec.d, model.spec.q), model.phi, (), model.theta, (), 0.0, model.sse, model.n_fit, model.sigma2)
    return Filtered(eta, arma, SAR.filter_series(arma, eta))


def forecast_origins(model, fl, X, origins, h, promo_known=True):
    """Prognosen (Ursprünge, h). promo_known=False setzt den Aktionsregressor in der Zukunft auf 0 (keine geplante Aktion bekannt)."""
    org = np.asarray(origins)
    days = org[:, None] + np.arange(h)[None, :]
    Xf = X[days].copy()                                                    # (n_o, h, K)
    if not promo_known and "Aktion" in model.names:
        Xf[:, :, model.names.index("Aktion")] = 0.0
    reg = Xf @ np.array(model.beta)
    err = SAR.forecast_origins(fl.arma, fl.filt, org, h, clip=False)
    f = reg + err
    if model.spec.log:
        return np.maximum(np.exp(f + 0.5 * model.sigma2), 0.0)
    return np.maximum(f, 0.0)
