"""Vehikel "Tagesaufträge eines Depots": drei Jahre täglich, Erwartungswert = Niveau x Trend x Wochenmuster x Jahresmuster x Niveausprung x Aktionen x Feiertage; beobachtet wird ein Zählwert mit multiplikativem
log-normalem Rauschen (mittelwerttreu). Alle Komponenten sind einzeln regelbar; das Vehikel wird von den späteren Stücken der Linie wiederverwendet (Feiertage, Aktionen und der Erwartungswert stehen daher mit im Ergebnis)."""

from dataclasses import dataclass

import numpy as np

import dr_constants as C


@dataclass(frozen=True)
class Series:
    y: np.ndarray             # (n,) beobachtete Tagesaufträge (ganze Zahlen, >= 0)
    mu: np.ndarray            # (n,) Erwartungswert ohne Rauschen (Orakel-Prognose)
    dow: np.ndarray           # (n,) Wochentag 0 = Montag
    holiday: np.ndarray       # (n,) 1, wenn Feiertag
    promo: np.ndarray         # (n,) 1, wenn Aktionstag
    shift_day: int            # Tag des Niveausprungs (-1: keiner)
    seed: int

    @property
    def n(self):
        return len(self.y)


def generate(trend=C.DEFAULT_TREND, weekly=C.DEFAULT_WEEKLY, yearly=C.DEFAULT_YEARLY, noise=C.DEFAULT_NOISE, shift=C.DEFAULT_SHIFT, events=C.DEFAULT_EVENTS, n_days=C.N_DAYS, seed=0):
    """trend: Prozent je Jahr; weekly: Stärke des Wochenmusters (1 = Standard); yearly: Amplitude des Jahresmusters; noise: Streuung des Rauschens (log-normal); shift: Niveausprung in Prozent an einem zufälligen Tag im
    Testjahr (Ursprünge ab Tag 730, mindestens 30 Tage nach dem ersten Ursprung); events: Stärke von Feiertagen und Aktionen (0 = keine)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dow = t % 7
    doy = t % 365
    pattern = np.array(C.WEEKLY_PATTERN)
    pattern = pattern / pattern.mean()
    week_f = 1.0 + weekly * (pattern[dow] - 1.0)
    phase = rng.uniform(0, 2 * np.pi)
    year_f = 1.0 + yearly * np.sin(2 * np.pi * doy / 365.0 + phase)
    trend_f = 1.0 + (trend / 100.0) * t / 365.0
    shift_day = int(rng.integers(C.FIRST_TEST + 30, n_days - 60)) if shift != 0 else -1
    shift_f = np.where((t >= shift_day) & (shift_day >= 0), 1.0 + shift / 100.0, 1.0)
    holiday = np.isin(doy, C.HOLIDAY_DOY).astype(float)
    holiday_f = 1.0 - events * C.HOLIDAY_DROP * holiday
    after = np.roll(holiday, 1)
    after[0] = 0.0
    holiday_f = holiday_f + events * C.HOLIDAY_REBOUND * after
    promo = np.zeros(n_days)
    starts = rng.choice(np.arange(30, n_days - C.PROMO_LENGTH), size=C.PROMO_PER_YEAR * (n_days // 365), replace=False)
    for s in starts:
        promo[s:s + C.PROMO_LENGTH] = 1.0
    promo_f = 1.0 + events * 0.5 * promo
    mu = C.LEVEL * trend_f * week_f * year_f * shift_f * holiday_f * promo_f
    mu = np.maximum(mu, 1.0)
    z = rng.normal(size=n_days)
    y = np.maximum(np.rint(mu * np.exp(noise * z - 0.5 * noise ** 2)), 0.0)
    return Series(y, mu, dow, holiday, promo, shift_day, int(seed))
