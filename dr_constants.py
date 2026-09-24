"""Konstanten der Dynamische-Regression-Demo: Vehikel "Tagesaufträge eines Depots" (Stück 4 der Zeitreihen-Prognose-Linie, wortgleich zu Stück 1 bis 3), Regressoren, Modelle, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095                                       # drei Jahre täglich
FIRST_TEST = 730                                    # Ursprünge liegen im letzten Jahr; die Parameter werden auf den ersten zwei Jahren geschätzt
LEVEL = 100.0                                       # mittlere Tagesaufträge zu Beginn
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)     # Tage im Jahr (0 = 1. Januar), an denen das Depot ruht
HOLIDAY_DROP = 0.5                                  # Rückgang am Feiertag (Faktor 1 - 0,5 * Stärke)
HOLIDAY_REBOUND = 0.15                              # Nachholeffekt am Folgetag
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3

TREND_MIN, TREND_MAX, TREND_STEP, DEFAULT_TREND = -20, 40, 5, 10              # Prozent je Jahr
WEEKLY_MIN, WEEKLY_MAX, WEEKLY_STEP, DEFAULT_WEEKLY = 0.0, 1.5, 0.25, 1.0
YEARLY_MIN, YEARLY_MAX, YEARLY_STEP, DEFAULT_YEARLY = 0.0, 0.5, 0.05, 0.2
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.02, 0.5, 0.02, 0.14
SHIFT_MIN, SHIFT_MAX, SHIFT_STEP, DEFAULT_SHIFT = -40, 40, 10, 0             # Prozent, Niveausprung an einem zufälligen Tag im Testjahr
EVENTS_MIN, EVENTS_MAX, EVENTS_STEP, DEFAULT_EVENTS = 0.0, 1.0, 0.25, 0.5     # Stärke von Feiertagen und Aktionen
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 14
STEP_MIN, STEP_MAX, DEFAULT_STEP = 1, 14, 1
K_WEEKS_MIN, K_WEEKS_MAX, DEFAULT_K_WEEKS = 2, 12, 4

SEASON_PERIOD = 7

# --- Glättungsmodelle aus Stück 2 (zum Vergleich; ari_ets.py ist der Kern von es_ets.py) -------------------------------------------------------------
ETS_MODELS = {
    "ses": ("none", "none"),
    "holt": ("add", "none"),
    "damped": ("damped", "none"),
    "hw_add": ("add", "add"),
    "hw_mult": ("add", "mul"),
    "hw_mult_damped": ("damped", "mul"),
}
INIT_DAYS = 28
INIT_WEEKS = 8
FIT_STAGE1 = 3000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98

# --- Dynamische Regression: Regler, Verfahren, Experimente --------------------------------------------------------------------------------------
YEAR_K_MIN, YEAR_K_MAX, DEFAULT_YEAR_K = 0, 4, 2    # Fourier-Paare für das Jahresmuster
P_MAX, D_MAX, Q_MAX = 2, 1, 2                       # Ordnungen der ARIMA-Fehler
DEFAULT_ERR = (0, 0, 0)                             # p, d, q
PROMO_MODES = ("none", "known", "unknown")          # Aktionen: nicht im Modell / im Modell, Zukunft bekannt / im Modell, Zukunft unbekannt (0)
PROMO_LABELS = {"none": "nicht im Modell", "known": "im Modell, Aktionsplan bekannt", "unknown": "im Modell, Zukunft unbekannt (0)"}
DEFAULT_PROMO = "known"

METHODS = ("user", "sarima", "hw_mult", "snaive_k")
METHOD_NAMES = {
    "user": "Ihre dynamische Regression",
    "sarima": "SARIMA(1,0,1)(1,1,1) im Log (Stück 3)",
    "hw_mult": "Holt-Winters multiplikativ (Stück 2)",
    "snaive_k": "Wochenmittel der letzten k Wochen (Stück 1)",
}

ACF_LAGS = 28
LJUNG_LAGS = 14

# --- Experimente (feste Seeds) -------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(12))
YEAR_K_LEVELS = (0, 1, 2, 3, 4)
EVENT_LEVELS = (0.25, 0.5, 0.75, 1.0)
EVENT_AFTER_DAYS = 14
