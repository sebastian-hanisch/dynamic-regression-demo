"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. ari_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dr_constants as C


def _bool(value):
    v = str(value).strip().lower()
    if v in ("1", "true", "ja", "yes"):
        return True
    if v in ("0", "false", "nein", "no"):
        return False
    raise ValueError(value)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _promo(value):
    v = str(value).strip().lower()
    if v not in C.PROMO_MODES:
        raise ValueError(value)
    return v


SETTING_SPECS = {
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "weekly_slider": SettingSpec("weekly", float, C.DEFAULT_WEEKLY, C.WEEKLY_MIN, C.WEEKLY_MAX),
    "yearly_slider": SettingSpec("yearly", float, C.DEFAULT_YEARLY, C.YEARLY_MIN, C.YEARLY_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "shift_slider": SettingSpec("shift", int, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX),
    "events_slider": SettingSpec("events", float, C.DEFAULT_EVENTS, C.EVENTS_MIN, C.EVENTS_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "step_slider": SettingSpec("step", int, C.DEFAULT_STEP, C.STEP_MIN, C.STEP_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K_WEEKS, C.K_WEEKS_MIN, C.K_WEEKS_MAX),
    "week_check": SettingSpec("week", _bool, True),
    "rtrend_check": SettingSpec("rtrend", _bool, True),
    "year_slider": SettingSpec("yeark", int, C.DEFAULT_YEAR_K, C.YEAR_K_MIN, C.YEAR_K_MAX),
    "holiday_check": SettingSpec("holiday", _bool, True),
    "promo_select": SettingSpec("promo", _promo, C.DEFAULT_PROMO),
    "p_slider": SettingSpec("p", int, C.DEFAULT_ERR[0], 0, C.P_MAX),
    "d_slider": SettingSpec("d", int, C.DEFAULT_ERR[1], 0, C.D_MAX),
    "q_slider": SettingSpec("q", int, C.DEFAULT_ERR[2], 0, C.Q_MAX),
    "log_check": SettingSpec("log", _bool, True),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"trend": "trend_slider", "weekly": "weekly_slider", "yearly": "yearly_slider", "noise": "noise_slider", "shift": "shift_slider", "events": "events_slider", "horizon": "horizon_slider",
               "step": "step_slider", "k": "k_slider", "seed": "seed_input", "week": "week_check", "r_trend": "rtrend_check", "year_k": "year_slider", "holiday": "holiday_check", "promo": "promo_select", "p": "p_slider",
               "d": "d_slider", "q": "q_slider", "log": "log_check"}
STEPS = {"trend_slider": C.TREND_STEP, "weekly_slider": C.WEEKLY_STEP, "yearly_slider": C.YEARLY_STEP, "noise_slider": C.NOISE_STEP, "shift_slider": C.SHIFT_STEP, "events_slider": C.EVENTS_STEP}


def _p(**kw):
    base = {"trend": C.DEFAULT_TREND, "weekly": C.DEFAULT_WEEKLY, "yearly": C.DEFAULT_YEARLY, "noise": C.DEFAULT_NOISE, "shift": C.DEFAULT_SHIFT, "events": C.DEFAULT_EVENTS, "horizon": C.DEFAULT_HORIZON,
            "step": C.DEFAULT_STEP, "k": C.DEFAULT_K_WEEKS, "seed": 3, "week": True, "r_trend": True, "year_k": C.DEFAULT_YEAR_K, "holiday": True, "promo": C.DEFAULT_PROMO, "p": C.DEFAULT_ERR[0], "d": C.DEFAULT_ERR[1],
            "q": C.DEFAULT_ERR[2], "log": True}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: alle Regressoren, OLS-Fehler": _p(),
    "Mit ARIMA(0,1,1)-Fehlern": _p(d=1, q=1),
    "Niveausprung +30 %, OLS-Fehler": _p(shift=30),
    "Niveausprung +30 %, ARIMA(0,1,1)-Fehler": _p(shift=30, d=1, q=1),
    "Ohne Jahresregressor, ARIMA(0,1,1)-Fehler": _p(year_k=0, d=1, q=1),
    "Ohne Ereignis-Regressoren (Stärke 1,0)": _p(events=1.0, holiday=False, promo="none"),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value)) if isinstance(value, bool) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: alle Regressoren, OLS-Fehler": "Seed 3: MASE 0,76 gegen Holt-Winters 0,84, SARIMA 0,84 und Wochenmittel 0,88; Orakel-Untergrenze 0,75 (nur 2 % darüber). Geschätzt: Feiertag −27 % (wahr −25 %), Aktion +24 % (wahr +25 %); Ljung-Box p = 0,75. "
                                                  "Im Mittel über 12 Reihen: 0,746 gegen 0,880 bei Holt-Winters.",
    "Mit ARIMA(0,1,1)-Fehlern": "Seed 3: MASE 0,76 (θ = −0,995); im Mittel über 12 Reihen 0,743 gegen 0,746 mit OLS-Fehlern - die Absicherung kostet bei vollständigen Regressoren nichts.",
    "Niveausprung +30 %, OLS-Fehler": "Seed 3, Sprung +30 % an Tag 809: die Regression erreicht nur 1,62 - schlechter als Holt-Winters (1,03), SARIMA (1,02) und das Wochenmittel (1,08), denn kein Regressor kennt den Sprung. Im Mittel über 12 Reihen 1,41 gegen 1,06 bei Holt-Winters.",
    "Niveausprung +30 %, ARIMA(0,1,1)-Fehler": "Seed 3: 1,23 statt 1,62 mit OLS-Fehlern; im Mittel über 12 Reihen 1,09 (OLS: 1,41, Holt-Winters: 1,06). Die ARIMA-Fehler fangen den Sprung teilweise auf.",
    "Ohne Jahresregressor, ARIMA(0,1,1)-Fehler": "Seed 3: 0,81 (θ = −0,89) gegen Holt-Winters 0,84; im Mittel über 12 Reihen 0,79 gegen 0,88 (mit OLS-Fehlern und ohne Jahresregressor: 1,20). Die Fehlerreihe fängt das fehlende Jahresmuster auf; der Ljung-Box-Test bleibt dabei unauffällig.",
    "Ohne Ereignis-Regressoren (Stärke 1,0)": "Seed 3, Ereignisstärke 1,0, Feiertage und Aktionen nicht im Modell: MASE 0,74 gegen Holt-Winters 0,81 und SARIMA 0,79, Orakel 0,62; der Ljung-Box-Test lehnt ab (p < 0,001). Im Mittel über 12 Reihen 0,81 gegen 0,90 bei Holt-Winters - schon das Jahresmuster trägt viel.",
}
