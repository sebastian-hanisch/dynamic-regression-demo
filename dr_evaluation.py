"""Auswertung: eine frei wählbare dynamische Regression im Rolling-Origin-Vergleich mit SARIMA, Holt-Winters und dem Wochenmittel; wahre Effekte des Vehikels; drei Experimente (Regressoren, Effekte, Fehlermodelle)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dr_constants as C
import dr_diagnostics as D
import dr_dynreg as R
import dr_ets as ETS
import dr_forecast as F
import dr_sarima as SAR
import dr_scenario as S

SPEC_SARIMA = SAR.Spec(1, 0, 1, 1, 1, 1, True)


@dataclass(frozen=True)
class Settings:
    trend: int = C.DEFAULT_TREND
    weekly: float = C.DEFAULT_WEEKLY
    yearly: float = C.DEFAULT_YEARLY
    noise: float = C.DEFAULT_NOISE
    shift: int = C.DEFAULT_SHIFT
    events: float = C.DEFAULT_EVENTS
    horizon: int = C.DEFAULT_HORIZON
    step: int = C.DEFAULT_STEP
    k: int = C.DEFAULT_K_WEEKS
    seed: int = 3
    week: bool = True
    r_trend: bool = True
    year_k: int = C.DEFAULT_YEAR_K
    holiday: bool = True
    promo: str = C.DEFAULT_PROMO
    p: int = C.DEFAULT_ERR[0]
    d: int = C.DEFAULT_ERR[1]
    q: int = C.DEFAULT_ERR[2]
    log: bool = True

    @property
    def spec(self):
        return R.Spec(self.week, self.r_trend, self.year_k, self.holiday, self.promo != "none", self.p, self.d, self.q, self.log)

    @property
    def series_key(self):
        return (self.trend, self.weekly, self.yearly, self.noise, self.shift, self.events, self.seed)


@dataclass
class Analysis:
    settings: Settings
    series: S.Series
    X: np.ndarray
    user: tuple                # (Modell, Filtered)
    origins: np.ndarray
    actual: np.ndarray
    errors: dict
    summary: dict
    oracle: dict
    horizon_mae: dict
    origin_mae: dict

    @property
    def best(self):
        return min(self.summary, key=lambda m: self.summary[m]["mase"])

    @property
    def train_resid(self):
        """Ein-Schritt-Fehler der Regression auf den Trainingstagen (ohne Burn-in)."""
        model, fl = self.user
        sp = model.spec
        lo = max(R.BURN - sp.d, sp.p)
        return fl.filt.e[lo:C.FIRST_TEST - sp.d]


@lru_cache(maxsize=128)
def _series(key):
    trend, weekly, yearly, noise, shift, events, seed = key
    return S.generate(trend, weekly, yearly, noise, shift, events, seed=seed)


@lru_cache(maxsize=2048)
def _fit_dyn(key, spec):
    series = _series(key)
    X, names = R.design(series, spec)
    model = R.fit(series.y[:C.FIRST_TEST], X[:C.FIRST_TEST], names, spec)
    return X, model, R.filter_series(model, series.y, X)


@lru_cache(maxsize=256)
def _fit_sarima(key):
    y = _series(key).y
    model = SAR.fit(y[:C.FIRST_TEST], SPEC_SARIMA)
    return model, SAR.filter_series(model, y)


@lru_cache(maxsize=256)
def _fit_ets(key):
    y = _series(key).y
    model = ETS.fit(y[:C.FIRST_TEST], "hw_mult")
    return model, ETS.filter_states(model, y)


def oracle_summary(series, org, h):
    """Kennzahlen der Orakel-Prognose: der wahre Erwartungswert als Prognose, gemessen am beobachteten Wert (untere Grenze für jedes Verfahren im Mittel)."""
    E = np.stack([series.mu[t:t + h] - series.y[t:t + h] for t in org])
    scale = F.mase_scale(series.y, C.FIRST_TEST)
    mae = float(np.abs(E).mean())
    return {"mae": mae, "rmse": float(np.sqrt((E ** 2).mean())), "me": float(E.mean()), "mase": mae / scale}


def analyse(s, methods=C.METHODS):
    series = _series(s.series_key)
    y = series.y
    org = F.origins(series.n, s.horizon, step=s.step)
    actual = np.stack([y[t:t + s.horizon] for t in org])
    X, model, fl = _fit_dyn(s.series_key, s.spec)
    errors = {}
    for m in methods:
        if m == "user":
            f = R.forecast_origins(model, fl, X, org, s.horizon, promo_known=(s.promo != "unknown"))
        elif m == "sarima":
            sm, sf = _fit_sarima(s.series_key)
            f = SAR.forecast_origins(sm, sf, org, s.horizon)
        elif m == "hw_mult":
            em, es = _fit_ets(s.series_key)
            f = ETS.forecast_origins(em, es, org, s.horizon)
        elif m == "snaive_k":
            f = F.snaive_k_origins(y, org, s.horizon, s.k)
        else:
            raise ValueError(m)
        errors[m] = f - actual
    return Analysis(s, series, X, (model, fl), org, actual, errors, F.summarize(errors, y), oracle_summary(series, org, s.horizon), F.per_horizon(errors), F.per_origin(errors))


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


def _spec_kw(spec, promo="known"):
    return dict(week=spec.week, r_trend=spec.trend, year_k=spec.year_k, holiday=spec.holiday, promo=promo if spec.promo else "none", p=spec.p, d=spec.d, q=spec.q, log=spec.log)


# --- Wahre Effekte des Vehikels ---------------------------------------------------------------------------------------------------------------


def true_effects(s):
    """Wahre Log-Effekte der Regressoren im Vehikel (Feiertag, Nachholeffekt, Aktion; Wochentage gegenüber Montag)."""
    week = np.array(C.WEEKLY_PATTERN)
    week = week / week.mean()
    week_f = 1.0 + s.weekly * (week - 1.0)
    out = {C.WEEKDAYS[d]: float(np.log(week_f[d] / week_f[0])) for d in range(1, 7)}
    out["Feiertag"] = float(np.log(1.0 - s.events * C.HOLIDAY_DROP))
    out["Tag nach Feiertag"] = float(np.log(1.0 + s.events * C.HOLIDAY_REBOUND))
    out["Aktion"] = float(np.log(1.0 + s.events * 0.5))
    return out


def yearly_curves(model, s):
    """Jahresmuster über die Tage des Jahres: geschätzt (Fourier-Teil des Modells, auf Mittel 1) und wahr (1 + a sin(2 pi doy / 365 + phase), auf Mittel 1)."""
    doy = np.arange(365)
    fit = np.zeros(365)
    for name, b in zip(model.names, model.beta):
        if name.startswith("sin "):
            fit += b * np.sin(2 * np.pi * int(name.split()[1]) * doy / 365.0)
        elif name.startswith("cos "):
            fit += b * np.cos(2 * np.pi * int(name.split()[1]) * doy / 365.0)
    phase = np.random.default_rng(s.seed).uniform(0, 2 * np.pi)
    true = 1.0 + s.yearly * np.sin(2 * np.pi * doy / 365.0 + phase)
    est = np.exp(fit)
    return doy, est / est.mean(), true / true.mean()


# --- Diagnose ---------------------------------------------------------------------------------------------------------------------------------


def diagnostics(a):
    model, _ = a.user
    resid = a.train_resid
    q, p = D.ljung_box(resid, C.LJUNG_LAGS, model.spec.n_arma)
    return {"resid": resid, "acf_res": D.acf(resid, C.ACF_LAGS), "lb_q": q, "lb_p": p}


# --- Experiment 1: Regressoren ----------------------------------------------------------------------------------------------------------------

LADDER = (
    ("Nur Konstante", dict(week=False, r_trend=False, year_k=0, holiday=False, promo="none")),
    ("+ Wochentag", dict(week=True, r_trend=False, year_k=0, holiday=False, promo="none")),
    ("+ Trend", dict(week=True, r_trend=True, year_k=0, holiday=False, promo="none")),
    ("+ Jahr (2 Paare)", dict(week=True, r_trend=True, year_k=2, holiday=False, promo="none")),
    ("+ Feiertage", dict(week=True, r_trend=True, year_k=2, holiday=True, promo="none")),
    ("+ Aktionen", dict(week=True, r_trend=True, year_k=2, holiday=True, promo="known")),
)


def ladder_experiment(seeds=None, base=None):
    """Regressoren Schritt für Schritt hinzufügen (OLS-Fehler): MASE über die Seeds; dazu Aktionen nur in der Vergangenheit bekannt, ohne Log, die Zahl der Fourier-Paare und die Vergleichsverfahren."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = _replace(Settings() if base is None else base, p=0, d=0, q=0, log=True)
    res = {name: [] for name, _ in LADDER}
    extra = {"unknown": [], "nolog": []}
    fourier = {k: [] for k in C.YEAR_K_LEVELS}
    refs = {m: [] for m in ("sarima", "hw_mult", "snaive_k")}
    floor = []
    for sd in seeds:
        for name, kw in LADDER:
            res[name].append(analyse(_replace(base, seed=sd, **kw), ("user",)).summary["user"]["mase"])
        full = _replace(base, seed=sd)
        extra["unknown"].append(analyse(_replace(full, promo="unknown"), ("user",)).summary["user"]["mase"])
        extra["nolog"].append(analyse(_replace(full, log=False), ("user",)).summary["user"]["mase"])
        for k in C.YEAR_K_LEVELS:
            fourier[k].append(analyse(_replace(full, year_k=k), ("user",)).summary["user"]["mase"])
        a = analyse(full, ("sarima", "hw_mult", "snaive_k"))
        for m in refs:
            refs[m].append(a.summary[m]["mase"])
        floor.append(a.oracle["mase"])
    return {"n_seeds": len(seeds), "rows": [{"name": name, **dict(zip(("mase", "se"), _mean_se(res[name])))} for name, _ in LADDER], "unknown": _mean_se(extra["unknown"]), "nolog": _mean_se(extra["nolog"]),
            "fourier": {k: _mean_se(v) for k, v in fourier.items()}, "floor": float(np.mean(floor)), **{m: _mean_se(v)[0] for m, v in refs.items()}}


# --- Experiment 2: Effekte -----------------------------------------------------------------------------------------------------------------------

EFFECT_NAMES = ("Feiertag", "Tag nach Feiertag", "Aktion", "Sa", "So")


def effects_experiment(levels=None, seeds=None, base=None):
    """Geschätzte Log-Effekte (Mittel über die Seeds) gegen die wahren Effekte für verschiedene Ereignisstärken; dazu der mittlere Standardfehler und der Anteil der Reihen, in denen der wahre Wert innerhalb von zwei Standardfehlern liegt."""
    levels = C.EVENT_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = _replace(Settings() if base is None else base, p=0, d=0, q=0, log=True, week=True, holiday=True, promo="known")
    rows = []
    for ev in levels:
        est = {n: [] for n in EFFECT_NAMES}
        se = {n: [] for n in EFFECT_NAMES}
        hit = {n: [] for n in EFFECT_NAMES}
        truth = true_effects(_replace(base, events=ev))
        for sd in seeds:
            st = _replace(base, events=ev, seed=sd)
            model = _fit_dyn(st.series_key, st.spec)[1]
            for n in EFFECT_NAMES:
                i = model.names.index(n)
                est[n].append(model.beta[i])
                se[n].append(model.se[i])
                hit[n].append(abs(model.beta[i] - truth[n]) <= 2 * model.se[i])
        rows.append({"events": ev, "n_seeds": len(seeds), "truth": {n: truth[n] for n in EFFECT_NAMES}, "est": {n: float(np.mean(est[n])) for n in EFFECT_NAMES}, "se": {n: float(np.mean(se[n])) for n in EFFECT_NAMES},
                     "coverage": {n: float(np.mean(hit[n])) for n in EFFECT_NAMES}})
    return rows


# --- Experiment 3: Fehlermodelle ------------------------------------------------------------------------------------------------------------------

ERROR_MODELS = (("OLS-Fehler (0,0,0)", (0, 0, 0)), ("AR(1)-Fehler (1,0,0)", (1, 0, 0)), ("ARIMA(0,1,1)-Fehler", (0, 1, 1)), ("Irrfahrt-Fehler (0,1,0)", (0, 1, 0)))
COMPLETE = dict(week=True, r_trend=True, year_k=2, holiday=True, promo="known")
SCENARIOS = (
    ("Standardreihe, alle Regressoren", {}, COMPLETE),
    ("Niveausprung +30 %", dict(shift=30), COMPLETE),
    ("Jahresregressor fehlt", {}, dict(COMPLETE, year_k=0)),
    ("Ereignis-Regressoren fehlen (Stärke 1,0)", dict(events=1.0), dict(COMPLETE, holiday=False, promo="none")),
)


def errors_experiment(seeds=None, base=None):
    """Vier Fehlermodelle in vier Szenarien (vollständig, Niveausprung, Regressor fehlt): MASE über die Seeds, dazu Holt-Winters und der Mittelwert des geschätzten AR(1)-Koeffizienten."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for label, sc_kw, reg in SCENARIOS:
        res = {name: [] for name, _ in ERROR_MODELS}
        hw, phi = [], []
        lb = {name: [] for name, _ in ERROR_MODELS}
        for sd in seeds:
            for name, (p, d, q) in ERROR_MODELS:
                st = _replace(base, seed=sd, p=p, d=d, q=q, log=True, **sc_kw, **reg)
                a = analyse(st, ("user",) if (p, d, q) != (0, 0, 0) else ("user", "hw_mult"))
                res[name].append(a.summary["user"]["mase"])
                lb[name].append(diagnostics(a)["lb_p"] < 0.05)
                if (p, d, q) == (0, 0, 0):
                    hw.append(a.summary["hw_mult"]["mase"])
                if (p, d, q) == (1, 0, 0):
                    phi.append(a.user[0].phi[0])
        rows.append({"scenario": label, "n_seeds": len(seeds), "hw_mult": float(np.mean(hw)), "phi": float(np.mean(phi)), "lb": {name: float(np.mean(v)) for name, v in lb.items()}, **{name: _mean_se(v)[0] for name, v in res.items()}})
    return rows
