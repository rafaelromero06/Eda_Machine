from __future__ import annotations

from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COBALTO = "#1d5fc4"
OCRE = "#d07a10"
VINO = "#8c2346"
VERDE = "#7d9a1e"
TINTA = "#10203f"
TINTA_2 = "#4a5570"
GRIS = "#9aa3b2"
SUPERFICIE = "#fffdf8"
REJILLA = "#ece5d6"
EJE = "#cfc7b6"
NEUTRO = "#efe9dc"
FUENTE = "'Instrument Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
BUENO, NO_BUENO = "Bueno (nota ≥ 7)", "No bueno (nota ≤ 6)"
CLASES = {NO_BUENO: OCRE, BUENO: COBALTO}
TIPOS = {"Tinto": ("red", VINO), "Blanco": ("white", VERDE)}
SECUENCIA = [[0, "#e4edf9"], [0.5, "#7fa8e4"], [1, "#0e3f8f"]]
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"], "toImageButtonOptions": {"scale": 2}}

PLANTILLA = go.layout.Template(layout=go.Layout(
    font=dict(family=FUENTE, size=13, color=TINTA),
    title=dict(font=dict(family=FUENTE, size=15, color=TINTA), x=0, xanchor="left", y=0.97),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=[COBALTO, OCRE],
    xaxis=dict(gridcolor=REJILLA, linecolor=EJE, zeroline=False, ticks="outside", tickcolor=EJE, automargin=True,
               title=dict(font=dict(color=TINTA_2))),
    yaxis=dict(gridcolor=REJILLA, linecolor=EJE, zeroline=False, ticks="outside", tickcolor=EJE, automargin=True,
               title=dict(font=dict(color=TINTA_2))),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=TINTA_2)),
    hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FUENTE, color=TINTA), bordercolor=EJE),
    margin=dict(l=60, r=20, t=60, b=50), barcornerradius=4, bargap=0.35, separators=". ",
))


def grafico(id_, alto, **kw):
    """dcc.Graph de alto fijo y ancho adaptable (el alto lo fija el contenedor)."""
    from dash import dcc
    return dcc.Graph(id=id_, responsive=True, style={"height": f"{alto}px"}, config=CONFIG, **kw)


def figura(fig, **kw):
    """dcc.Graph con una figura ya hecha: el contenedor toma el alto de la figura y el ancho se adapta."""
    from dash import dcc
    return dcc.Graph(figure=fig, responsive=True, style={"height": f"{fig.layout.height or 380}px"}, config=CONFIG, **kw)


def _base(fig, titulo, alto=380):
    fig.update_layout(template=PLANTILLA, title=titulo, height=alto, transition=dict(duration=450, easing="cubic-in-out"))
    return fig


def vacia(mensaje, alto=300):
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(color=TINTA_2, size=14), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, "", alto)


def _por_clase(df):
    bueno = df["quality"] >= 7
    return ((NO_BUENO, OCRE, df[~bueno]), (BUENO, COBALTO, df[bueno]))


def huella_deslizador(df, variable, bajo, alto, puntos=36):
    """Silueta de la distribución por clase para el fondo de un deslizador, como imagen SVG en línea."""
    bordes = np.linspace(bajo, alto, puntos + 1)
    bueno = df["quality"] >= 7
    h1 = np.histogram(df.loc[bueno, variable], bordes, density=True)[0]
    h0 = np.histogram(df.loc[~bueno, variable], bordes, density=True)[0]
    tope = max(h1.max(), h0.max(), 1e-9)
    trazo = lambda h: "M0 24 " + " ".join(f"L{i} {24 - 22 * v / tope:.1f}" for i, v in enumerate(h)) + f" L{puntos} 24Z"
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {puntos} 24' preserveAspectRatio='none'>"
           f"<path d='{trazo(h0)}' fill='{OCRE}' fill-opacity='.38'/><path d='{trazo(h1)}' fill='{COBALTO}' fill-opacity='.42'/></svg>")
    return f"url(\"data:image/svg+xml,{quote(svg)}\")"


def distribucion_objetivo(df):
    conteo = df["quality"].value_counts().sort_index()
    total = conteo.sum()
    fig = go.Figure()
    for nombre, color, m in ((NO_BUENO, OCRE, conteo.index < 7), (BUENO, COBALTO, conteo.index >= 7)):
        c = conteo[m]
        fig.add_trace(go.Bar(x=c.index.astype(str), y=c.values, name=nombre, marker_color=color, customdata=100 * c.values / total,
                             text=[f"{100 * v / total:.1f} %" for v in c.values], textposition="outside", cliponaxis=False,
                             textfont=dict(size=11, color=TINTA_2),
                             hovertemplate="Nota %{x}<br>%{y} vinos (%{customdata:.1f} %)<extra></extra>"))
    fig.update_xaxes(title="Nota de calidad (0 a 10)", type="category", categoryorder="array", categoryarray=[str(i) for i in range(3, 10)])
    fig.update_yaxes(title="Número de vinos")
    fig = _base(fig, "¿Cuántos vinos hay de cada nota?", 360)
    fig.update_layout(bargap=0.55)
    return fig


def histograma(df, variable, etiqueta):
    fig = go.Figure()
    for nombre, color, sub in _por_clase(df):
        fig.add_trace(go.Histogram(x=sub[variable], name=nombre, marker_color=color, opacity=0.62, histnorm="percent", nbinsx=40,
                                   hovertemplate="%{x}<br>%{y:.1f} % de la clase<extra>" + nombre + "</extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=etiqueta)
    fig.update_yaxes(title="% de los vinos de cada clase")
    return _base(fig, f"Distribución de {etiqueta.split(' (')[0].lower()} por clase", 360)


def caja_atipicos(df, variable, etiqueta):
    fig = go.Figure()
    for nombre, color, sub in _por_clase(df):
        fig.add_trace(go.Box(y=sub[variable], name=nombre.split(" (")[0], marker_color=color, boxpoints="outliers", line_width=1.5, marker=dict(size=5)))
    fig.update_yaxes(title=etiqueta)
    fig.update_layout(showlegend=False)
    return _base(fig, "Valores atípicos (regla de 1.5 RIC)", 360)


def correlaciones(df, columnas, etiquetas):
    c = df[columnas].corr().round(2)
    nombres = [etiquetas[x] for x in columnas]
    fig = go.Figure(go.Heatmap(z=c.values, x=nombres, y=nombres, zmin=-1, zmax=1, xgap=3, ygap=3,
                               colorscale=[[0, OCRE], [0.5, NEUTRO], [1, COBALTO]], texttemplate="%{z:.2f}", textfont=dict(size=10),
                               hovertemplate="%{y} y %{x}<br>r = %{z:.2f}<extra></extra>",
                               colorbar=dict(title="r de Pearson", thickness=10, outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_xaxes(showgrid=False, tickangle=-45, ticks="")
    return _base(fig, "Mural de correlaciones de Pearson", 560)


def huella_quimica(df, columnas, etiquetas):
    """Diferencia entre vinos buenos y no buenos en cada variable, en desviaciones estándar."""
    bueno = df["quality"] >= 7
    dif = ((df.loc[bueno, columnas].mean() - df.loc[~bueno, columnas].mean()) / df[columnas].std().replace(0, np.nan)).dropna().sort_values()
    fig = go.Figure()
    for nombre, color, parte in (("Más bajo en los vinos buenos", OCRE, dif[dif < 0]), ("Más alto en los vinos buenos", COBALTO, dif[dif >= 0])):
        y = [etiquetas[c] for c in parte.index]
        fig.add_trace(go.Bar(x=parte.values, y=y, orientation="h", width=0.12, marker_color=color, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=parte.values, y=y, mode="markers+text", name=nombre, text=[f"{v:+.2f}" for v in parte.values],
                                 textposition=["middle left" if v < 0 else "middle right" for v in parte.values],
                                 textfont=dict(size=11, color=TINTA_2), cliponaxis=False,
                                 marker=dict(size=11, color=color, line=dict(width=2, color=SUPERFICIE)),
                                 hovertemplate="%{y}<br>%{x:+.2f} desviaciones estándar<extra></extra>"))
    limite = 1.3 * float(dif.abs().max() or 1)
    fig.add_vline(x=0, line_color=EJE, line_width=1)
    fig.update_xaxes(title="Diferencia de medias (en desviaciones estándar)", range=[-limite, limite])
    fig.update_yaxes(categoryorder="array", categoryarray=[etiquetas[c] for c in dif.index], showgrid=False, ticks="")
    return _base(fig, "Huella química del vino bueno", 440)


def dispersion(df, x, y, color, etiquetas):
    base = dict(mode="markers")
    ficha = f"{etiquetas[x]}: %{{x}}<br>{etiquetas[y]}: %{{y}}"
    marca = dict(size=7, opacity=0.55, line=dict(width=1, color=SUPERFICIE))
    fig = go.Figure()
    if color == "clase":
        for nombre, col, sub in _por_clase(df):
            fig.add_trace(go.Scattergl(x=sub[x], y=sub[y], name=nombre, marker=dict(color=col, **marca), hovertemplate=ficha + f"<extra>{nombre}</extra>", **base))
    elif color == "tipo_vino":
        for nombre, (valor, col) in TIPOS.items():
            sub = df[df["tipo_vino"] == valor]
            fig.add_trace(go.Scattergl(x=sub[x], y=sub[y], name=nombre, marker=dict(color=col, **marca), hovertemplate=ficha + f"<extra>{nombre}</extra>", **base))
    else:
        fig.add_trace(go.Scattergl(x=df[x], y=df[y], name=etiquetas[color],
                                   marker=dict(color=df[color], size=7, opacity=0.7, colorscale=SECUENCIA, showscale=True,
                                               colorbar=dict(title=etiquetas[color].split(" (")[0], thickness=10, outlinewidth=0)),
                                   hovertemplate=ficha + f"<br>{etiquetas[color]}: %{{marker.color}}<extra></extra>", **base))
    fig.update_xaxes(title=etiquetas[x])
    fig.update_yaxes(title=etiquetas[y])
    return _base(fig, f"{etiquetas[y].split(' (')[0]} frente a {etiquetas[x].split(' (')[0].lower()}", 460)


def danza_notas(df, x, y, etiquetas, por_nota=260):
    """Nube de vinos de cada nota con la trayectoria de sus medianas, animada de la nota más baja a la más alta."""
    notas = sorted(df["quality"].unique())
    medianas = df.groupby("quality")[[x, y]].median()

    def nube(n):
        sub = df[df["quality"] == n]
        sub = sub.sample(min(len(sub), por_nota), random_state=0)
        return go.Scatter(x=sub[x], y=sub[y], mode="markers", name=f"Vinos de nota {n}",
                          marker=dict(size=8, color=COBALTO if n >= 7 else OCRE, opacity=0.5, line=dict(width=1, color=SUPERFICIE)),
                          hovertemplate=f"Nota {n}<br>{etiquetas[x]}: %{{x}}<br>{etiquetas[y]}: %{{y}}<extra></extra>")

    def centro(n):
        return go.Scatter(x=[medianas.loc[n, x]], y=[medianas.loc[n, y]], mode="markers+text", text=[f"nota {n}"], textposition="top center",
                          textfont=dict(size=13, color=TINTA), name="Mediana de la nota", marker=dict(size=16, color=TINTA, line=dict(width=2, color=SUPERFICIE)),
                          hoverinfo="skip")

    ruta = go.Scatter(x=medianas[x], y=medianas[y], mode="lines", line=dict(color=TINTA, width=2), name="Camino de las medianas", hoverinfo="skip")
    pasos = [dict(method="animate", label=str(n), args=[[str(n)], dict(mode="immediate", frame=dict(duration=500, redraw=False),
                                                                        transition=dict(duration=400))]) for n in notas]
    fig = go.Figure(data=[nube(notas[0]), ruta, centro(notas[0])],
                    frames=[go.Frame(name=str(n), data=[nube(n), ruta, centro(n)], traces=[0, 1, 2]) for n in notas])
    fig.update_xaxes(title=etiquetas[x], range=[df[x].quantile(0.002), df[x].quantile(0.998)])
    fig.update_yaxes(title=etiquetas[y], range=[df[y].quantile(0.002), df[y].quantile(0.998)])
    fig = _base(fig, "Cómo se mueven los vinos al subir la nota", 500)
    fig.update_layout(
        transition=None, margin=dict(l=60, r=20, t=64, b=110),
        updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=-0.32, xanchor="left", yanchor="top", pad=dict(r=8, t=8),
                          bgcolor=SUPERFICIE, bordercolor=EJE,
                          buttons=[dict(label="▶ Reproducir", method="animate",
                                        args=[None, dict(frame=dict(duration=900, redraw=False), transition=dict(duration=550), fromcurrent=True)]),
                                   dict(label="❚❚ Pausa", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.26, len=0.74, y=-0.3, yanchor="top", steps=pasos, currentvalue=dict(prefix="Nota: ", font=dict(color=TINTA)),
                      pad=dict(t=8), bordercolor=EJE, tickcolor=EJE, font=dict(color=TINTA_2))])
    return fig


def barras_modelos(tabla, columna, titulo, eje, resaltar, error=None, formato=".3f"):
    t = tabla.sort_values(columna)
    colores = [COBALTO if m == resaltar else GRIS for m in t["modelo"]]
    fig = go.Figure(go.Bar(x=t[columna], y=t["modelo"], orientation="h", marker_color=colores,
                           error_x=dict(type="data", array=t[error], color=TINTA_2, thickness=1) if error else None,
                           text=None if error else [f"{v:{formato}}" for v in t[columna]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>" + eje + ": %{x:" + formato + "}<extra></extra>"))
    if error:
        for m, v, e in zip(t["modelo"], t[columna], t[error]):
            fig.add_annotation(x=v + (e if np.isfinite(e) else 0), y=m, text=f"{v:{formato}}", showarrow=False, xanchor="left", xshift=6,
                               font=dict(size=11, color=TINTA_2))
    fig.update_xaxes(title=eje)
    fig.update_layout(margin=dict(l=200, r=60))
    return _base(fig, titulo, 60 + 34 * len(t))


def costo_exactitud(t, resaltar, cortos):
    """Tiempo de predicción frente a AUC: cuánto cuesta cada modelo por el desempeño que da."""
    fig = go.Figure()
    for destacado, color in ((False, GRIS), (True, COBALTO)):
        sub = t[(t["modelo"] == resaltar) == destacado]
        fig.add_trace(go.Scatter(x=sub["prediccion_ms_por_caso"], y=sub["auc_media"], mode="markers+text", text=[cortos[m] for m in sub["modelo"]],
                                 textposition="top center", textfont=dict(size=12, color=TINTA), cliponaxis=False, customdata=sub[["modelo"]], showlegend=False,
                                 marker=dict(size=14 if destacado else 11, color=color, line=dict(width=2, color=SUPERFICIE)),
                                 hovertemplate="%{customdata[0]}<br>AUC %{y:.4f}<br>%{x:.3f} ms por vino<extra></extra>"))
    fig.update_xaxes(title="Tiempo de predicción (ms por vino, escala logarítmica)", type="log")
    fig.update_yaxes(title="AUC medio en 25 bloques")
    return _base(fig, "¿Cuánto cuesta cada punto de AUC?", 400)


def mural(tabla, filas, columnas, valor, titulo, etiqueta, invertir=False, formato=".3f"):
    """Mural de baldosas de una métrica: una baldosa por combinación y una estrella en la mejor de cada fila."""
    piv = tabla.pivot_table(index=filas, columns=columnas, values=valor)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=invertir).index]
    mejor = piv.idxmin(axis=1) if invertir else piv.idxmax(axis=1)
    texto = [["★" if col == mejor[m] else "" for col in piv.columns] for m in piv.index]
    if isinstance(piv.columns, pd.MultiIndex):
        x = [list(piv.columns.get_level_values(0)), list(piv.columns.get_level_values(1))]
    else:
        x = list(piv.columns)
    escala = [[p, c] for (p, c) in (SECUENCIA[::-1] if invertir else SECUENCIA)]
    if invertir:
        escala = [[1 - p, c] for p, c in escala]
    fig = go.Figure(go.Heatmap(z=piv.values, x=x, y=list(piv.index), text=texto, texttemplate="%{text}", textfont=dict(size=14),
                               xgap=3, ygap=3, colorscale=escala, reversescale=invertir, hoverongaps=False,
                               colorbar=dict(title=etiqueta, thickness=10, outlinewidth=0),
                               hovertemplate="%{y}<br>%{x}<br>" + etiqueta + ": %{z:" + formato + "}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_xaxes(showgrid=False, ticks="")
    return _base(fig, titulo, 90 + 52 * len(piv))


def adicionales(tabla, conjuntos, resaltar, mejores):
    """AUC de los 10 modelos en cada conjunto adicional; el modelo original y el mejor rival quedan resaltados."""
    fig = go.Figure()
    t = tabla.assign(nombre=tabla["conjunto"].map(conjuntos))
    resto = t[t["modelo"] != resaltar]
    fig.add_trace(go.Scatter(x=resto["auc_media"], y=resto["nombre"], mode="markers", name="Otros modelos", customdata=resto[["modelo"]],
                             marker=dict(size=10, color=GRIS, opacity=0.7, line=dict(width=2, color=SUPERFICIE)),
                             hovertemplate="%{customdata[0]}<br>AUC %{x:.4f}<extra></extra>"))
    top = pd.DataFrame([t[(t["conjunto"] == c) & (t["modelo"] == m)].iloc[0] for c, m in mejores.items()])
    fig.add_trace(go.Scatter(x=top["auc_media"], y=top["nombre"], mode="markers", name="Mejor rival del conjunto", customdata=top[["modelo"]],
                             marker=dict(size=13, symbol="diamond", color=OCRE, line=dict(width=2, color=SUPERFICIE)),
                             hovertemplate="%{customdata[0]}<br>AUC %{x:.4f}<extra></extra>"))
    bof = t[t["modelo"] == resaltar]
    fig.add_trace(go.Scatter(x=bof["auc_media"], y=bof["nombre"], mode="markers", name="Bosque Ordinal Focalizado",
                             marker=dict(size=14, color=COBALTO, line=dict(width=2, color=SUPERFICIE)),
                             hovertemplate="Bosque Ordinal Focalizado<br>AUC %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="AUC medio en 5 pliegues")
    fig.update_yaxes(showgrid=False, ticks="")
    return _base(fig, "¿Dónde queda el modelo original en otros problemas?", 340)


def curva_roc(fpr, tpr, auc, nombre):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=EJE, width=1), name="Azar", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=COBALTO, width=2), name=nombre,
                             hovertemplate="Falsos positivos: %{x:.2f}<br>Sensibilidad: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Umbral elegido", hoverinfo="skip",
                             marker=dict(size=14, color=OCRE, line=dict(width=2, color=SUPERFICIE))))
    fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])
    fig.update_yaxes(title="Sensibilidad", range=[0, 1.02])
    return _base(fig, f"Curva ROC (AUC {auc:.3f})", 360)


def curva_pr(recall, precision, pr_auc, prevalencia):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalencia, prevalencia], mode="lines", line=dict(color=EJE, width=1), name="Sin modelo", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line=dict(color=COBALTO, width=2), name="Modelo",
                             hovertemplate="Sensibilidad: %{x:.2f}<br>Precisión: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Umbral elegido", hoverinfo="skip",
                             marker=dict(size=14, color=OCRE, line=dict(width=2, color=SUPERFICIE))))
    fig.update_xaxes(title="Sensibilidad", range=[0, 1])
    fig.update_yaxes(title="Precisión", range=[0, 1.02])
    return _base(fig, f"Precisión y sensibilidad (PR-AUC {pr_auc:.3f})", 360)


def curva_calibracion(curva, nombre):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=EJE, width=1), name="Calibración perfecta", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curva["p_media"], y=curva["frecuencia"], mode="lines+markers", name=nombre, line=dict(color=COBALTO, width=2),
                             marker=dict(size=9, line=dict(width=2, color=SUPERFICIE)), customdata=curva[["n"]],
                             hovertemplate="Probabilidad media: %{x:.2f}<br>Proporción real: %{y:.2f}<br>%{customdata[0]} vinos<extra></extra>"))
    fig.update_xaxes(title="Probabilidad predicha", range=[0, 1])
    fig.update_yaxes(title="Proporción real de vinos buenos", range=[0, 1])
    return _base(fig, "Curva de calibración (10 grupos)", 360)


def importancia_global(tabla, titulo, eje="Media del valor absoluto del aporte SHAP"):
    t = tabla.sort_values("importancia")
    fig = go.Figure(go.Bar(x=t["importancia"], y=t["variable"], orientation="h", marker_color=COBALTO,
                           hovertemplate="%{y}<br>" + eje + ": %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title=eje)
    fig.update_yaxes(showgrid=False, ticks="")
    return _base(fig, titulo, 70 + 30 * len(t))


def distribucion_nota(niveles, prob):
    etiquetas = [str(int(n)) for n in niveles]
    colores = [COBALTO if n >= 7 else OCRE for n in niveles]
    fig = go.Figure(go.Bar(x=etiquetas, y=100 * prob, marker_color=colores, text=[f"{100 * p:.0f} %" for p in prob], textposition="outside",
                           cliponaxis=False, textfont=dict(color=TINTA_2), hovertemplate="Nota %{x}<br>%{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title="Nota de calidad", type="category")
    fig.update_yaxes(title="Probabilidad (%)", range=[0, max(100 * prob) * 1.25])
    return _base(fig, "Nota que espera el modelo original", 300)


def diagrama_cd(rangos, cd, resaltar, rival=None, titulo="Diagrama de diferencia crítica (Nemenyi, α = 0.05)"):
    r = rangos.sort_values()
    y = np.arange(len(r))[::-1]
    color = [COBALTO if m == resaltar else (OCRE if m == rival else GRIS) for m in r.index]
    fig = go.Figure(go.Scatter(x=r.values, y=y, mode="markers+text", text=r.index, textposition="middle right", textfont=dict(color=TINTA),
                               marker=dict(size=12, color=color, line=dict(width=2, color=SUPERFICIE)),
                               hovertemplate="%{text}<br>rango medio: %{x:.2f}<extra></extra>", showlegend=False, cliponaxis=False))
    mejor = r.iloc[0]
    fig.add_vrect(x0=mejor, x1=mejor + cd, fillcolor=COBALTO, opacity=0.08, line_width=0)
    fig.add_shape(type="line", x0=mejor, x1=mejor + cd, y0=len(r) - 0.4, y1=len(r) - 0.4, line=dict(color=TINTA, width=2))
    fig.add_annotation(x=mejor + cd / 2, y=len(r) - 0.1, text=f"CD = {cd:.2f}", showarrow=False, font=dict(color=TINTA_2))
    fig.update_xaxes(title="Rango medio en los 25 bloques (1 = mejor)", range=[0.5, max(r.max() + 3, mejor + cd + 0.5)])
    fig.update_yaxes(visible=False, range=[-0.8, len(r) + 0.4])
    return _base(fig, titulo, 90 + 34 * len(r))


def cajas_bloques(tab, resaltar, titulo="AUC en cada bloque"):
    orden = tab.mean().sort_values().index
    fig = go.Figure()
    for m in orden:
        fig.add_trace(go.Box(x=tab[m], name=m, orientation="h", marker_color=COBALTO if m == resaltar else GRIS, boxpoints="all", jitter=0.3,
                             pointpos=0, marker=dict(size=5), line_width=1.5, hovertemplate=m + "<br>AUC %{x:.3f}<extra></extra>"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="AUC en el pliegue externo")
    fig.update_yaxes(showgrid=False, ticks="")
    return _base(fig, titulo, 80 + 38 * len(orden))


def lineas(df, x, y, grupo, titulo, eje_x, eje_y, resaltar=None, referencia=None, log_x=False, ordenar=True):
    fig = go.Figure()
    for g in df[grupo].unique():
        sub = df[df[grupo] == g]
        sub = sub.sort_values(x) if ordenar else sub
        destacado = resaltar is None or g == resaltar
        fig.add_trace(go.Scatter(x=sub[x], y=sub[y], mode="lines+markers", name=str(g),
                                 line=dict(color=COBALTO if destacado else GRIS, width=2 if destacado else 1.5),
                                 marker=dict(size=9 if destacado else 6, line=dict(width=2, color=SUPERFICIE)), opacity=1 if destacado else 0.75,
                                 hovertemplate=f"{g}<br>{eje_x}: %{{x}}<br>{eje_y}: %{{y:.3f}}<extra></extra>"))
    if referencia is not None:
        fig.add_hline(y=referencia[1], line_color=OCRE, line_width=1.5, annotation_text=referencia[0], annotation_position="bottom right",
                      annotation_font_color=TINTA_2)
    fig.update_xaxes(title=eje_x, type="log" if log_x else "linear")
    fig.update_yaxes(title=eje_y)
    return _base(fig, titulo, 400)
