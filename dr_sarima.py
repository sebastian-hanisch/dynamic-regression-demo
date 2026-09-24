"""SARIMA(p,d,q)(P,D,Q)_7 in numpy: Differenzieren, ARMA-Modell der Differenzen, Schätzung durch Minimieren der bedingten Fehlerquadrate (CSS), Prognose durch Fortschreiben und Rückrechnen.

Mit dem Rückwärtsschiebeoperator B ($B y_t = y_{t-1}$):

    (1 - phi_1 B - ...)(1 - Phi_1 B^7 - ...)  (1-B)^d (1-B^7)^D  y_t  =  (1 + theta_1 B + ...)(1 + Theta_1 B^7 + ...)  e_t

Das ARMA-Modell wird für die Differenzen w_t = (1-B)^d (1-B^7)^D y_t geschätzt; die Fehler e_t entstehen rekursiv (e_t = 0 vor dem ersten Tag, an dem alle Rückgriffe der AR-Seite Daten haben: "bedingt").
Stationarität und Umkehrbarkeit werden über Partial-Autokorrelationen erzwungen (Durbin-Levinson). Optional wird log(y) modelliert (dann sind Saison und Niveau multiplikativ) und die Prognose mit exp(f + sigma^2/2) zurückgerechnet.

Alle Kandidaten der Parametersuche laufen gleichzeitig durch die Reihe; danach werden die Fehler mit den festen Parametern durch die ganze Reihe fortgeschrieben - zu jedem Ursprung t kennt das Modell nur y[:t]."""

from dataclasses import dataclass

import numpy as np

import dr_constants as C

M = C.SEASON_PERIOD

FIT_STAGE1 = 2000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 30
FIT_SEED = 20240925
BURN = 200                                          # die ersten Fehler zählen nicht in die Fehlersumme (die bedingte Anfangsannahme e = 0 verzerrt sie)
R_MAX = 0.995


@dataclass(frozen=True)
class Spec:
    p: int = 0
    d: int = 0
    q: int = 0
    P: int = 0
    D: int = 0
    Q: int = 0
    log: bool = False

    @property
    def n_arma(self):
        return self.p + self.q + self.P + self.Q

    @property
    def lost(self):
        """Tage, die das Differenzieren am Anfang verbraucht."""
        return self.d + M * self.D

    @property
    def label(self):
        return f"({self.p},{self.d},{self.q})({self.P},{self.D},{self.Q})" + (" log" if self.log else "")


@dataclass(frozen=True)
class Model:
    spec: Spec
    phi: tuple
    Phi: tuple
    theta: tuple
    Theta: tuple
    mean: float               # Mittel der Differenzen (nur ohne Differenzieren, sonst 0)
    sse: float
    n_fit: int
    sigma2: float             # Fehlervarianz (der modellierten Größe, bei log: im Log)

    @property
    def n_params(self):
        return self.spec.n_arma + 1 + (self.spec.d + self.spec.D == 0)

    @property
    def aicc(self):
        n, k = self.n_fit, self.n_params
        return float(n * np.log(self.sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


@dataclass(frozen=True)
class Filtered:
    w: np.ndarray             # Differenzen (T - lost,), ohne das Mittel
    e: np.ndarray             # Ein-Schritt-Fehler je Differenz (bedingt), Länge wie w
    z: np.ndarray             # die modellierte Reihe (y oder log y)


# --- Polynome -----------------------------------------------------------------------------------------------------------------------------------


def _pacf_to_coefs(r):
    """(nb, k) Partial-Autokorrelationen -> (nb, k) Koeffizienten eines stationären AR-Polynoms 1 - sum c_j B^j (Durbin-Levinson)."""
    nb, k = r.shape
    c = np.zeros((nb, k))
    for i in range(k):
        prev = c[:, :i].copy()
        c[:, i] = r[:, i]
        if i > 0:
            c[:, :i] = prev - r[:, i:i + 1] * prev[:, ::-1]
    return c


def _poly_mul(a, b):
    out = np.zeros((a.shape[0], a.shape[1] + b.shape[1] - 1))
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            out[:, i + j] += a[:, i] * b[:, j]
    return out


def ar_ma_polys(phi, Phi, theta, Theta):
    """Gesamtkoeffizienten A (a(B) = 1 - sum A_k B^k) und Bc (b(B) = 1 + sum Bc_k B^k) aus den nicht saisonalen und saisonalen Teilen; jedes Argument hat die Form (nb, Anzahl)."""
    nb = phi.shape[0]

    def seasonal(x, sign):
        out = np.zeros((nb, M * x.shape[1] + 1))
        out[:, 0] = 1.0
        for j in range(x.shape[1]):
            out[:, M * (j + 1)] = sign * x[:, j]
        return out

    def plain(x, sign):
        out = np.ones((nb, x.shape[1] + 1))
        out[:, 1:] = sign * x
        return out

    a = _poly_mul(plain(phi, -1.0), seasonal(Phi, -1.0))
    b = _poly_mul(plain(theta, 1.0), seasonal(Theta, 1.0))
    return -a[:, 1:], b[:, 1:]


def diff_coefs(d, D):
    """c_k mit (1-B)^d (1-B^7)^D = 1 - sum c_k B^k (leer, wenn nicht differenziert wird)."""
    poly = np.array([1.0])
    for _ in range(d):
        poly = np.convolve(poly, [1.0, -1.0])
    for _ in range(D):
        poly = np.convolve(poly, [1.0] + [0.0] * (M - 1) + [-1.0])
    return -poly[1:]


def difference(z, d, D):
    c = diff_coefs(d, D)
    lost = d + M * D
    if lost == 0:
        return np.asarray(z, dtype=float).copy()
    z = np.asarray(z, dtype=float)
    w = z[lost:].copy()
    for k, ck in enumerate(c, start=1):
        if ck != 0.0:
            w -= ck * z[lost - k:len(z) - k]
    return w


# --- Fehler und Schätzung -----------------------------------------------------------------------------------------------------------------------


def _css(w, A, Bc, start_sse=0):
    """Fehler e (nb, T) und Fehlerquadrate ab Index start_sse für alle Kandidaten gleichzeitig. Bedingt: e = 0 vor Index LA (Länge der AR-Seite)."""
    nb, LA = A.shape
    LB = Bc.shape[1]
    T = len(w)
    wp = np.concatenate([np.zeros(LA), w])
    idx = np.arange(T)[:, None] - np.arange(1, LA + 1)[None, :] + LA
    ar_part = A @ wp[idx].T if LA else np.zeros((nb, T))
    E = np.zeros((nb, LB + T))
    for t in range(LA, T):
        c = t + LB
        ma = np.sum(Bc * E[:, c - LB:c][:, ::-1], axis=1) if LB else 0.0
        E[:, c] = w[t] - ar_part[:, t] - ma
    e = E[:, LB:]
    start = max(start_sse, LA)
    return e, np.sum(e[:, start:] ** 2, axis=1)


def _transform(spec, z):
    return np.log(np.maximum(z, 1.0)) if spec.log else np.asarray(z, dtype=float)


def _split(spec, u):
    """Einheitswürfel (nb, n_arma) mit Werten in (-1, 1) -> (phi, Phi, theta, Theta)."""
    nb = u.shape[0]
    r = np.clip(u, -R_MAX, R_MAX)
    i = 0
    parts = []
    for k, sign in ((spec.p, 1.0), (spec.P, 1.0), (spec.q, -1.0), (spec.Q, -1.0)):
        parts.append(sign * _pacf_to_coefs(r[:, i:i + k]) if k else np.zeros((nb, 0)))
        i += k
    return tuple(parts)


def fit(y_train, spec, seed=FIT_SEED, burn=BURN):
    """Parameter durch Minimieren der bedingten Fehlerquadrate auf y_train schätzen (deterministisch: fester Seed der Suche)."""
    z = _transform(spec, y_train)
    w = difference(z, spec.d, spec.D)
    mean = float(w.mean()) if spec.d + spec.D == 0 else 0.0
    w = w - mean
    n_arma = spec.n_arma
    rng = np.random.default_rng(seed)
    start = max(burn - spec.lost, 0)                            # Burn-in in Tagen der Reihe: Modelle mit anderer Differenzierung zählen dieselben Tage

    def score(u):
        phi, Phi, theta, Theta = _split(spec, u)
        A, Bc = ar_ma_polys(phi, Phi, theta, Theta)
        with np.errstate(all="ignore"):
            e, sse = _css(w, A, Bc, start)
        return np.where(np.isfinite(sse), sse, np.inf)

    if n_arma == 0:
        e, sse = _css(w, np.zeros((1, 0)), np.zeros((1, 0)), start)
        best_u, best_s = np.zeros((1, 0)), float(sse[0])
    else:
        U = rng.uniform(-R_MAX, R_MAX, size=(FIT_STAGE1, n_arma))
        S = score(U)
        order = np.argsort(S)[:FIT_TOP]
        top_u, top_s = U[order], S[order]
        for r in range(FIT_ROUNDS):
            scale = 0.2 * 0.55 ** r
            cand = np.clip(np.repeat(top_u, FIT_PER_START, axis=0) + scale * rng.normal(size=(FIT_TOP * FIT_PER_START, n_arma)), -R_MAX, R_MAX)
            cs = score(cand)
            pool_u, pool_s = np.vstack([top_u, cand]), np.concatenate([top_s, cs])
            order = np.argsort(pool_s)[:FIT_TOP]
            top_u, top_s = pool_u[order], pool_s[order]
        best_u, best_s = top_u[:1], float(top_s[0])
    phi, Phi, theta, Theta = _split(spec, best_u)
    LA = spec.p + M * spec.P
    n_fit = len(w) - max(start, LA)
    return Model(spec, tuple(phi[0]), tuple(Phi[0]), tuple(theta[0]), tuple(Theta[0]), mean, best_s, n_fit, best_s / n_fit)


def filter_series(model, y):
    """Fehler mit den festen Parametern durch die ganze Reihe fortschreiben."""
    spec = model.spec
    z = _transform(spec, y)
    w = difference(z, spec.d, spec.D) - model.mean
    A, Bc = ar_ma_polys(np.array([model.phi]), np.array([model.Phi]), np.array([model.theta]), np.array([model.Theta]))
    e, _ = _css(w, A, Bc)
    return Filtered(w, e[0], z)


def forecast_origins(model, filt, origins, h, bias_correct=True, clip=True):
    """Prognosen (Ursprünge, h) für die Tage t..t+h-1 aus den Daten vor dem Ursprung t; bei log-Modellen mit Bias-Korrektur exp(f + sigma^2/2) (Mittelwert; ohne: exp(f), der Median); negative Werte auf 0."""
    spec = model.spec
    org = np.asarray(origins)
    n_o = len(org)
    A, Bc = ar_ma_polys(np.array([model.phi]), np.array([model.Phi]), np.array([model.theta]), np.array([model.Theta]))
    A, Bc = A[0], Bc[0]
    LA, LB = len(A), len(Bc)
    i0 = org - spec.lost                                        # Index der ersten zukünftigen Differenz
    wpad = np.concatenate([np.zeros(max(LA, 1)), filt.w])
    epad = np.concatenate([np.zeros(max(LB, 1)), filt.e])
    LAp, LBp = max(LA, 1), max(LB, 1)
    wf = np.zeros((n_o, LAp + h))
    ef = np.zeros((n_o, LBp + h))
    for k in range(LAp):
        wf[:, k] = wpad[i0 + k]                                 # w[i0 - LAp + k]
    for k in range(LBp):
        ef[:, k] = epad[i0 + k]
    for j in range(h):
        val = np.zeros(n_o)
        for k in range(1, LA + 1):
            val += A[k - 1] * wf[:, LAp + j - k]
        for k in range(1, LB + 1):
            val += Bc[k - 1] * ef[:, LBp + j - k]
        wf[:, LAp + j] = val
    wfut = wf[:, LAp:] + model.mean
    c = diff_coefs(spec.d, spec.D)
    Lc = max(len(c), 1)
    zpad = np.concatenate([np.zeros(Lc), filt.z])
    zf = np.zeros((n_o, Lc + h))
    for k in range(Lc):
        zf[:, k] = zpad[org + k]                                # z[org - Lc + k]
    for j in range(h):
        val = wfut[:, j].copy()
        for k, ck in enumerate(c, start=1):
            if ck != 0.0:
                val += ck * zf[:, Lc + j - k]
        zf[:, Lc + j] = val
    f = zf[:, Lc:]
    if spec.log:
        f = np.exp(f + (0.5 * model.sigma2 if bias_correct else 0.0))
    return np.maximum(f, 0.0) if clip else f
