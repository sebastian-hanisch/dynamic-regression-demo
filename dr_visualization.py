"""Plotly-Abbildungen der Dynamische-Regression-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dr_constants as C
import dr_dynreg as R
import dr_ets as ETS
import dr_evaluation as E
import dr_forecast as F
import dr_sarima as SAR

COLORS = {"user": "#00897b", "sarima": "#c2185b", "hw_mult": "#e6550d", "snaive_k": "#17becf"}
SHORT = {"user": "Dynamische Regression", "sarima": "SARIMA (Stück 3)", "hw_mult": "Holt-Winters mult. (Stück 2)", "snaive_k": "Wochenmittel (Stück 1)"}
ACTUAL = "#14233B"
ORACLE = "#54a24b"
WARN = "#f58518"
ERROR_COLORS = ("#00897b", "#4db6ac", "#26a69a", "#b2dfdb")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def method_label(m, k):
    return f"Wochenmittel der letzten {k} Wochen (Stück 1)" if m == "snaive_k" else C.METHOD_NAMES[m]


def build_series(a, origin):
    """Die ganze Reihe (drei Jahre) mit Erwartungswert, dem Testjahr und dem gewählten Ursprung."""
    s = a.series
    t = np.arange(s.n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s.y, mode="lines", name="Tagesaufträge", line=dict(color=ACTUAL, width=1)))
    fig.add_trace(go.Scatter(x=t, y=s.mu, mode="lines", name="Erwartungswert (ohne Rauschen)", line=dict(color=ORACLE, width=1.5, dash="dot")))
    fig.add_vrect(x0=C.FIRST_TEST, x1=s.n, fillcolor="rgba(245,133,24,0.08)", line_width=0, annotation_text="Testjahr (Ursprünge)", annotation_position="top left")
    fig.add_vline(x=origin, line=dict(color=WARN, dash="dash"))
    if s.shift_day >= 0:
        fig.add_vline(x=s.shift_day, line=dict(color="#e45756", dash="dot"), annotation_text="Niveausprung", annotation_position="bottom right")
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 300)


def build_origin(a, origin):
    """Die 42 Tage vor dem Ursprung und die nächsten h Tage: Ist, Erwartungswert und die Prognosen der vier Verfahren."""
    s, st = a.series, a.settings
    h = st.horizon
    org = np.array([origin])
    x_hist = np.arange(origin - 42, origin)
    x_fut = np.arange(origin, origin + h)
    model, fl = a.user
    sm, sf = E._fit_sarima(st.series_key)
    em, es = E._fit_ets(st.series_key)
    fc = {"user": R.forecast_origins(model, fl, a.X, org, h, promo_known=(st.promo != "unknown"))[0], "sarima": SAR.forecast_origins(sm, sf, org, h)[0], "hw_mult": ETS.forecast_origins(em, es, org, h)[0],
          "snaive_k": F.snaive_k_origins(s.y, org, h, st.k)[0]}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=s.y[origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=s.y[origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=2), marker=dict(size=6, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=x_fut, y=s.mu[origin:origin + h], mode="lines", name="Erwartungswert", line=dict(color=ORACLE, width=2, dash="dot")))
    for m, f in fc.items():
        fig.add_trace(go.Scatter(x=x_fut, y=f, mode="lines", name=method_label(m, st.k), line=dict(color=COLORS[m], width=3 if m == "user" else 1.4, dash="solid" if m == "user" else "dash")))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.35))


def build_weekly(model, settings):
    """Wochentagsfaktoren (Montag = 1): geschätzt (exp der Koeffizienten) gegen wahr."""
    names = [C.WEEKDAYS[d] for d in range(1, 7)]
    est = [1.0] + [float(np.exp(model.coef(n))) for n in names]
    truth = [1.0] + [float(np.exp(v)) for v in (E.true_effects(settings)[n] for n in names)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(C.WEEKDAYS), y=est, name="geschätzt", marker=dict(color=COLORS["user"])))
    fig.add_trace(go.Scatter(x=list(C.WEEKDAYS), y=truth, mode="markers", name="wahr", marker=dict(color=ORACLE, size=11, symbol="diamond")))
    fig.update_yaxes(title_text="Faktor gegenüber Montag", rangemode="tozero")
    return _base(fig, 300)


def build_yearly(model, settings):
    doy, est, true = E.yearly_curves(model, settings)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=doy, y=est, mode="lines", name="geschätzt (Fourier-Paare)", line=dict(color=COLORS["user"], width=3)))
    fig.add_trace(go.Scatter(x=doy, y=true, mode="lines", name="wahr", line=dict(color=ORACLE, width=2, dash="dot")))
    fig.update_xaxes(title_text="Tag im Jahr")
    fig.update_yaxes(title_text="Faktor (Mittel 1)")
    return _base(fig, 300)


def build_acf(values, title, n, color="#00897b"):
    """Balkendiagramm der Autokorrelationen mit dem 95-%-Band 1,96 / sqrt(n); Lags 7, 14, 21, 28 sind hervorgehoben."""
    lags = np.arange(1, len(values) + 1)
    fig = go.Figure(go.Bar(x=lags, y=values, marker=dict(color=[WARN if lag % 7 == 0 else color for lag in lags]), showlegend=False))
    band = 1.96 / np.sqrt(n)
    fig.add_hline(y=band, line=dict(color="#7f7f7f", dash="dot"))
    fig.add_hline(y=-band, line=dict(color="#7f7f7f", dash="dot"))
    fig.update_xaxes(title_text="Lag (Tage)", dtick=7)
    fig.update_yaxes(title_text="ACF", range=[-1.05, 1.05])
    fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return _base(fig, 260)


def build_bars(a):
    """MASE je Verfahren über alle Ursprünge und Horizonte, dazu die Orakel-Untergrenze."""
    ms = sorted(a.summary, key=lambda m: a.summary[m]["mase"])
    fig = go.Figure(go.Bar(x=[SHORT[m] for m in ms], y=[a.summary[m]["mase"] for m in ms], marker=dict(color=[COLORS[m] for m in ms]), text=[f"{a.summary[m]['mase']:.2f}".replace(".", ",") for m in ms], textposition="outside", showlegend=False))
    fig.add_hline(y=a.oracle["mase"], line=dict(color=ORACLE, dash="dot"), annotation_text="Orakel (wahrer Erwartungswert)", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color="#7f7f7f", dash="dash"), annotation_text="MASE 1 = saisonal naiv im Training", annotation_position="bottom right")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 360)


def build_horizon(a):
    scale = F.mase_scale(a.series.y, C.FIRST_TEST)
    xs = list(range(1, a.settings.horizon + 1))
    fig = go.Figure()
    for m in a.summary:
        fig.add_trace(go.Scatter(x=xs, y=a.horizon_mae[m] / scale, mode="lines+markers", name=method_label(m, a.settings.k), line=dict(color=COLORS[m], width=2.5 if m == "user" else 1.4), marker=dict(size=4)))
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", dtick=1 if a.settings.horizon <= 14 else 2)
    fig.update_yaxes(title_text="MASE je Horizont", rangemode="tozero")
    return _base(fig, 320).update_layout(legend=dict(orientation="h", y=-0.35))


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_ladder(res):
    rows = res["rows"]
    labels = [r["name"] for r in rows] + ["Aktionen nur in der Vergangenheit bekannt", "ohne Log (alle Regressoren)"]
    vals = [r["mase"] for r in rows] + [res["unknown"][0], res["nolog"][0]]
    errs = [r["se"] for r in rows] + [res["unknown"][1], res["nolog"][1]]
    colors = ["#80cbc4"] * (len(rows) - 1) + [COLORS["user"], "#4db6ac", "#b2dfdb"]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], subplot_titles=("MASE: Regressoren hinzufügen (OLS-Fehler)", "Fourier-Paare für das Jahresmuster"))
    fig.add_trace(go.Bar(x=labels, y=vals, error_y=dict(type="data", array=errs), marker=dict(color=colors), text=[f"{v:.2f}".replace(".", ",") for v in vals], textposition="outside", showlegend=False), row=1, col=1)
    for m, name in (("hw_mult", "Holt-Winters"), ("sarima", "SARIMA")):
        fig.add_hline(y=res[m], line=dict(color=COLORS[m], dash="dash"), row=1, col=1, annotation_text=name, annotation_position="top left" if m == "hw_mult" else "bottom left")
    fig.add_hline(y=res["floor"], line=dict(color=ORACLE, dash="dot"), row=1, col=1, annotation_text="Orakel", annotation_position="bottom right")
    ks = list(res["fourier"])
    fig.add_trace(go.Scatter(x=[str(k) for k in ks], y=[res["fourier"][k][0] for k in ks], error_y=dict(type="data", array=[res["fourier"][k][1] for k in ks]), mode="lines+markers", line=dict(color=COLORS["user"], width=2), showlegend=False), row=1, col=2)
    fig.add_hline(y=res["hw_mult"], line=dict(color=COLORS["hw_mult"], dash="dash"), row=1, col=2)
    fig.update_xaxes(title_text="Paare (0 = kein Jahresmuster)", type="category", row=1, col=2)
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=2)
    return _base(fig, 400)


def build_effects(rows):
    names = ("Feiertag", "Tag nach Feiertag", "Aktion")
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{n}: log-Effekt" for n in names])
    xs = [r["events"] for r in rows]
    for i, n in enumerate(names, start=1):
        fig.add_trace(go.Scatter(x=xs, y=[r["est"][n] for r in rows], error_y=dict(type="data", array=[2 * r["se"][n] for r in rows]), mode="lines+markers", name="geschätzt (± 2 Standardfehler)", line=dict(color=COLORS["user"], width=2), showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(x=xs, y=[r["truth"][n] for r in rows], mode="lines", name="wahr", line=dict(color=ORACLE, dash="dot", width=2), showlegend=(i == 1)), row=1, col=i)
        fig.update_xaxes(title_text="Ereignisstärke", row=1, col=i)
    fig.update_yaxes(title_text="Effekt auf log(Aufträge)", col=1)
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_errors(rows):
    fig = go.Figure()
    models = [k for k in rows[0] if k not in ("scenario", "n_seeds", "hw_mult", "phi", "lb")]
    labels = [r["scenario"] for r in rows]
    for name, color in zip(models, ERROR_COLORS):
        fig.add_trace(go.Bar(x=labels, y=[r[name] for r in rows], name=name, marker=dict(color=color), text=[f"{r[name]:.2f}".replace(".", ",") for r in rows], textposition="outside"))
    fig.add_trace(go.Scatter(x=labels, y=[r["hw_mult"] for r in rows], mode="markers", name="Holt-Winters mult. (Stück 2)", marker=dict(color=COLORS["hw_mult"], size=14, symbol="diamond")))
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _base(fig, 400).update_layout(legend=dict(orientation="h", y=-0.35))
