from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import offset_copy

from . import config

TINTA, TINTA_2, TINTA_3 = "#0b0b0b", "#52514e", "#898781"
GRILLA, EJE, FONDO = "#e1e0d9", "#c3c2b7", "#ffffff"
AZUL, NARANJA, AQUA, ROJO = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRIS_NEUTRO = "#f0efec"
AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SECUENCIAL = mcolors.LinearSegmentedColormap.from_list("azules", AZULES)
CMAP_DIVERGENTE = mcolors.LinearSegmentedColormap.from_list("rojo_gris_azul", [ROJO, GRIS_NEUTRO, AZUL])

RUTA_TABLAS = config.RUTA_EXPERIMENTOS / "tablas"

NOMBRE_CORTO = {"Regresión Logística (L1/L2)": "Reg. logística", "sin_balanceo": "Sin balanceo"}
BALANCEOS = ["sin_balanceo", "class_weight", "SMOTE", "ADASYN"]
ETIQUETA_BALANCEO = {"sin_balanceo": "Sin balanceo", "class_weight": "class_weight",
                     "SMOTE": "SMOTE", "ADASYN": "ADASYN", "no_aplica": "—"}
COLOR_BALANCEO = {"sin_balanceo": TINTA_3, "class_weight": AZUL, "SMOTE": NARANJA, "ADASYN": AQUA}
OPTIMIZADORES = ["Grid Search", "Random Search", "Bayesiano", "Genético"]
VARIABLES = {"fixed acidity": "Acidez fija", "volatile acidity": "Acidez volátil",
             "citric acid": "Ácido cítrico", "residual sugar": "Azúcar residual",
             "chlorides": "Cloruros", "free sulfur dioxide": "SO₂ libre",
             "total sulfur dioxide": "SO₂ total", "density": "Densidad", "pH": "pH",
             "sulphates": "Sulfatos", "alcohol": "Alcohol", "tipo_vino": "Tipo de vino"}
UNIDADES = {"fixed acidity": "g/L", "volatile acidity": "g/L", "citric acid": "g/L", "residual sugar": "g/L",
            "chlorides": "g/L", "free sulfur dioxide": "mg/L", "total sulfur dioxide": "mg/L", "density": "g/cm³",
            "pH": "", "sulphates": "g/L", "alcohol": "% vol", "tipo_vino": ""}


def aplicar_estilo():
    """Parámetros de matplotlib comunes a todas las figuras."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": TINTA,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8, "axes.labelcolor": TINTA_2, "axes.labelsize": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": TINTA,
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.labelsize": 9, "ytick.labelsize": 9.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 6,
        "grid.color": GRILLA, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 9, "legend.handletextpad": 0.5, "legend.columnspacing": 1.6,
    })


def corto(nombre):
    return NOMBRE_CORTO.get(nombre, nombre)


def mas_menos(media, de, decimales=3):
    return f"{media:.{decimales}f} ± {de:.{decimales}f}"


def guardar(fig, nombre):
    fig.savefig(config.RUTA_FIGURAS / f"{nombre}.png")


def guardar_tabla(tabla, nombre):
    RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_TABLAS / f"{nombre}.csv", index=False)


def estilo_ejes(ax, grilla="both"):
    """Grilla tenue detrás de los datos y sin bordes superiores/derechos."""
    if grilla:
        ax.grid(axis=grilla)
    ax.set_axisbelow(True)


def encabezado(fig, titulo, subtitulo=None):
    """Agrega título y subtítulo alineados al borde izquierdo de la figura."""
    fig.canvas.draw()
    caja = fig.get_tightbbox(fig.canvas.get_renderer())
    ancho, alto = fig.get_size_inches()
    x, y = caja.x0 / ancho, caja.y1 / alto
    desplazamiento = 8
    if subtitulo:
        fig.text(x, y, subtitulo, ha="left", va="bottom", fontsize=9.5, color=TINTA_2, linespacing=1.45,
                 transform=offset_copy(fig.transFigure, fig=fig, y=desplazamiento, units="points"))
        desplazamiento += 9.5 * 1.45 * (subtitulo.count("\n") + 1) + 5
    fig.text(x, y, titulo, ha="left", va="bottom", fontsize=13.5, fontweight="semibold", color=TINTA,
             transform=offset_copy(fig.transFigure, fig=fig, y=desplazamiento, units="points"))


def barra_redondeada(ax, x, ancho, alto, color, radio_pt=3.0, horizontal=False):
    """Barra con el extremo redondeado y la base recta."""
    caja = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    radio_px = radio_pt * ax.figure.dpi / 72
    rx = radio_px * (x1 - x0) / caja.width
    ry = radio_px * (y1 - y0) / caja.height
    if not horizontal:
        ry = min(ry, alto / 2)
        ax.add_patch(FancyBboxPatch((x - ancho / 2, 0), ancho, alto, boxstyle=f"round,pad=0,rounding_size={rx}",
                                    mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x - ancho / 2, 0), ancho, alto - ry, facecolor=color, edgecolor="none", zorder=3))
    else:
        rx = min(rx, alto / 2)
        ax.add_patch(FancyBboxPatch((0, x - ancho / 2), alto, ancho, boxstyle=f"round,pad=0,rounding_size={ry}",
                                    mutation_aspect=rx / ry, facecolor=color, edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((0, x - ancho / 2), alto - rx, ancho, facecolor=color, edgecolor="none", zorder=3))


def luminancia(color):
    canales = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in mcolors.to_rgb(color)]
    return 0.2126 * canales[0] + 0.7152 * canales[1] + 0.0722 * canales[2]


def mapa_calor(ax, tabla, cmap, norm, formato, negrita=None, texto_na="no aplica", fuente=9.5):
    """Mapa de calor con el valor de cada celda anotado."""
    datos = tabla.to_numpy(dtype=float)
    filas, columnas = datos.shape
    ax.imshow(np.ma.masked_invalid(datos), cmap=cmap, norm=norm, aspect="auto")
    for i in range(filas):
        for j in range(columnas):
            valor = datos[i, j]
            if np.isnan(valor):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=GRIS_NEUTRO, edgecolor="none"))
                ax.text(j, i, texto_na, ha="center", va="center", fontsize=fuente - 1, color=TINTA_3)
                continue
            ax.text(j, i, formato(valor), ha="center", va="center", fontsize=fuente,
                    color=FONDO if luminancia(cmap(norm(valor))) < 0.18 else TINTA,
                    fontweight="bold" if negrita is not None and negrita[i, j] else "normal")
    ax.set_xticks(range(columnas), labels=tabla.columns)
    ax.set_yticks(range(filas), labels=tabla.index)
    ax.set_xticks(np.arange(1, columnas) - 0.5, minor=True)
    ax.set_yticks(np.arange(1, filas) - 0.5, minor=True)
    ax.grid(which="minor", color=FONDO, linewidth=2.5)
    ax.tick_params(which="both", length=0)
    ax.tick_params(axis="y", labelcolor=TINTA)
    ax.xaxis.tick_top()
    for lado in ax.spines.values():
        lado.set_visible(False)


def barra_color(fig, ax, cmap, norm, etiqueta, formato=None):
    barra = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal",
                         location="bottom", shrink=0.8, aspect=35, pad=0.03, format=formato)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=8.5, colors=TINTA_2, length=0)
    barra.set_label(etiqueta, fontsize=9, color=TINTA_2)
    return barra


def intervalos(ax, etiquetas, estimacion, inferior, superior, colores=None, referencia=0.0,
               texto=None, ancho_texto=None):
    """Gráfico de intervalos (forest plot)."""
    n = len(etiquetas)
    y = np.arange(n)[::-1]
    colores = colores or [AZUL] * n
    if referencia is not None:
        ax.axvline(referencia, color=TINTA_3, lw=1, zorder=1)
    for yi, e, a, b, c in zip(y, estimacion, inferior, superior, colores):
        ax.plot([a, b], [yi, yi], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(e, yi, "o", ms=7, color=c, mec=FONDO, mew=1.4, zorder=3)
    ax.set_yticks(y, labels=etiquetas)
    ax.tick_params(axis="y", labelcolor=TINTA)
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    if texto is not None:
        transformacion = ax.get_yaxis_transform()
        for yi, t in zip(y, texto):
            ax.text(1.02, yi, t, transform=transformacion, va="center", fontsize=8.8, color=TINTA_2)
    return y
