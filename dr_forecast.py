"""Rolling-Origin-Auswertung: dynamische Regression, SARIMA (Stück 3), Holt-Winters (Stück 2) und Wochenmittel (Stück 1) auf denselben Ursprüngen.

Ein Ursprung t heißt: bekannt sind die Tage 0..t-1 (und der Kalender - Feiertage, Aktionsplan - auch für die Zukunft), prognostiziert werden die Tage t..t+h-1. Alle Parameter stammen aus den Trainingstagen; Fehler und Zustände laufen mit jedem Tag weiter."""

import numpy as np

import dr_constants as C


def origins(n, h, first=C.FIRST_TEST, step=C.DEFAULT_STEP):
    return np.arange(first, n - h + 1, step)


def mase_scale(y, t0):
    """Mittlerer absoluter Fehler der saisonal naiven Prognose (Periode 7) innerhalb der Trainingsdaten y[:t0] - Nenner der MASE."""
    return float(np.abs(y[7:t0] - y[:t0 - 7]).mean())


def snaive_k_origins(y, org, h, k=C.DEFAULT_K_WEEKS):
    """Wochenmittel: derselbe Wochentag, gemittelt über die letzten k Wochen (Stück 1)."""
    org = np.asarray(org)
    j = np.arange(h)
    return np.mean([y[org[:, None] - 7 * (i + 1) + (j % 7)[None, :]] for i in range(k)], axis=0)


def summarize(errors, y, first=C.FIRST_TEST):
    """MAE, RMSE, ME (Verzerrung) und MASE je Verfahren über alle Ursprünge und Horizonte."""
    scale = mase_scale(y, first)
    out = {}
    for mth, E in errors.items():
        mae = float(np.abs(E).mean())
        out[mth] = {"mae": mae, "rmse": float(np.sqrt((E ** 2).mean())), "me": float(E.mean()), "mase": mae / scale}
    return out


def per_horizon(errors):
    return {m: np.abs(E).mean(axis=0) for m, E in errors.items()}


def per_origin(errors):
    """MAE je Ursprung (über den Horizont gemittelt)."""
    return {m: np.abs(E).mean(axis=1) for m, E in errors.items()}
