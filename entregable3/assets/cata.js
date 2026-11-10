(function () {
  "use strict";

  var raiz = document.documentElement;
  raiz.classList.add("con-js");
  var reducido = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var COLORES = {
    white: ["#eef0a6", "#cbd765", "#a3b73a"],
    red: ["#b8406a", "#82193c", "#560d25"]
  };
  var NO = function () { return window.dash_clientside.no_update; };

  function formato(v, d) {
    var partes = Math.abs(Number(v)).toFixed(d || 0).split(".");
    var entero = partes[0].length > 4 ? partes[0].replace(/\B(?=(\d{3})+(?!\d))/g, " ") : partes[0];
    return (Number(v) < 0 ? "−" : "") + entero + (partes[1] ? "." + partes[1] : "");
  }

  function limitar(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function crear(etiqueta, clase, texto) {
    var el = document.createElement(etiqueta);
    if (clase) el.className = clase;
    if (texto !== undefined) el.textContent = texto;
    return el;
  }

  function crearSvg(etiqueta, atributos) {
    var el = document.createElementNS("http://www.w3.org/2000/svg", etiqueta);
    Object.keys(atributos || {}).forEach(function (k) { el.setAttribute(k, atributos[k]); });
    return el;
  }

  function barajar(lista) {
    for (var i = lista.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = lista[i];
      lista[i] = lista[j];
      lista[j] = t;
    }
    return lista;
  }

  function caliz(c, k) {
    var x = function (v) { return 100 + (v - 100) * k; };
    c.beginPath();
    c.moveTo(x(57), 14);
    c.bezierCurveTo(x(49), 54, x(30), 84, x(30), 118);
    c.bezierCurveTo(x(30), 160, x(66), 189, 100, 189);
    c.bezierCurveTo(x(134), 189, x(170), 160, x(170), 118);
    c.bezierCurveTo(x(170), 84, x(151), 54, x(143), 14);
  }

  function Copa(lienzo) {
    this.lienzo = lienzo;
    this.ctx = lienzo.getContext("2d");
    this.nivel = 0;
    this.vel = 0;
    this.objetivo = limitar(parseFloat(lienzo.dataset.nivel || "0") || 0, 0, 1);
    this.tipo = lienzo.dataset.tipo || "white";
    this.tono = lienzo.dataset.tono || "oscuro";
    this.escala = lienzo.dataset.escala === "1";
    this.inclinacion = 0;
    this.inclinacionObjetivo = 0;
    this.energia = 0.5;
    this.t = Math.random() * 10;
    this.antes = 0;
    this.burbujas = [];
    this.visible = true;
    this.medir();
    if (reducido) this.nivel = this.objetivo;
  }

  Copa.prototype.medir = function () {
    var r = this.lienzo.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = Math.max(r.width, 20);
    this.h = Math.max(r.height, 20);
    this.dpr = dpr;
    this.lienzo.width = Math.round(this.w * dpr);
    this.lienzo.height = Math.round(this.h * dpr);
    var ancho = this.escala ? 236 : 200;
    this.s = Math.min(this.w / ancho, this.h / 282);
    this.ox = (this.w - ancho * this.s) / 2;
    this.oy = (this.h - 282 * this.s) / 2;
  };

  Copa.prototype.fijar = function (nivel, tipo) {
    nivel = limitar(Number(nivel) || 0, 0, 1);
    this.energia = Math.min(2.2, this.energia + Math.abs(nivel - this.objetivo) * 2.4 + 0.15);
    this.objetivo = nivel;
    if (tipo) this.tipo = tipo;
    if (reducido) {
      this.nivel = nivel;
      this.vel = 0;
      this.dibujar();
    }
  };

  Copa.prototype.superficie = function (x, y0) {
    var A = 0.7 + this.energia * 7.5;
    var pendiente = this.inclinacion + Math.sin(this.t * 3.1) * this.energia * 0.07;
    return y0 + pendiente * (x - 100) + A * 0.6 * Math.sin(x * 0.07 + this.t * 2.3) + A * 0.4 * Math.sin(x * 0.13 - this.t * 3.5);
  };

  Copa.prototype.paso = function (ms) {
    var dt = this.antes ? Math.min(ms - this.antes, 48) / 16.67 : 1;
    this.antes = ms;
    if (!reducido) {
      var a = (this.objetivo - this.nivel) * 0.05 - this.vel * 0.2;
      this.vel += a * dt;
      this.nivel += this.vel * dt;
      this.energia = this.energia * Math.pow(0.982, dt) + Math.abs(this.vel) * 0.8 * dt;
      this.inclinacion += (this.inclinacionObjetivo - this.inclinacion) * 0.05 * dt;
      this.t += 0.016 * dt;
      this.moverBurbujas(dt);
    }
    this.dibujar();
  };

  Copa.prototype.moverBurbujas = function (dt) {
    var y0 = 186 - this.nivel * 140;
    var tasa = (this.tipo === "white" ? 0.22 : 0.07) * this.nivel;
    if (Math.random() < tasa * dt && this.burbujas.length < 34) {
      this.burbujas.push({ x: 66 + Math.random() * 68, y: 184, r: 0.6 + Math.random() * 1.7, v: 0.22 + Math.random() * 0.45, f: Math.random() * 6 });
    }
    var self = this;
    this.burbujas = this.burbujas.filter(function (b) {
      b.y -= b.v * dt;
      b.x += Math.sin(self.t * 4 + b.f) * 0.12 * dt;
      return b.y > self.superficie(b.x, y0) + 2;
    });
  };

  Copa.prototype.dibujar = function () {
    var c = this.ctx;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);
    c.translate(this.ox, this.oy);
    c.scale(this.s, this.s);
    var claro = this.tono === "claro";
    var tinta = claro ? "rgba(255,255,255,0.92)" : "#10203f";
    var colores = COLORES[this.tipo] || COLORES.white;
    var y0 = 186 - this.nivel * 140;
    var i, x;
    if (this.nivel > 0.003) {
      c.save();
      caliz(c, 0.962);
      c.closePath();
      c.clip();
      var degradado = c.createLinearGradient(0, y0 - 12, 0, 190);
      degradado.addColorStop(0, colores[0]);
      degradado.addColorStop(0.45, colores[1]);
      degradado.addColorStop(1, colores[2]);
      c.fillStyle = degradado;
      c.beginPath();
      c.moveTo(20, 200);
      for (i = 0; i <= 48; i++) {
        x = 20 + i * (160 / 48);
        c.lineTo(x, this.superficie(x, y0));
      }
      c.lineTo(180, 200);
      c.closePath();
      c.fill();
      c.fillStyle = "rgba(255,255,255,0.55)";
      this.burbujas.forEach(function (b) {
        c.beginPath();
        c.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        c.fill();
      });
      c.beginPath();
      for (i = 0; i <= 48; i++) {
        x = 20 + i * (160 / 48);
        var y = this.superficie(x, y0) + 1.6;
        if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
      }
      c.strokeStyle = "rgba(255,255,255,0.5)";
      c.lineWidth = 1.6;
      c.stroke();
      c.restore();
    }
    c.lineJoin = "round";
    c.lineCap = "round";
    c.strokeStyle = tinta;
    c.lineWidth = 2.2;
    caliz(c, 1);
    c.stroke();
    c.beginPath();
    c.ellipse(100, 14, 43, 3.4, 0, 0, Math.PI * 2);
    c.lineWidth = 1.5;
    c.stroke();
    c.beginPath();
    c.moveTo(100, 189);
    c.lineTo(100, 262);
    c.lineWidth = 3;
    c.stroke();
    c.beginPath();
    c.ellipse(100, 267, 44, 5.5, 0, 0, Math.PI * 2);
    c.lineWidth = 2;
    c.stroke();
    c.beginPath();
    c.moveTo(46, 72);
    c.quadraticCurveTo(37, 104, 45, 142);
    c.strokeStyle = claro ? "rgba(255,255,255,0.32)" : "rgba(16,32,63,0.16)";
    c.lineWidth = 4;
    c.stroke();
    if (this.escala) {
      c.strokeStyle = claro ? "rgba(255,255,255,0.55)" : "rgba(16,32,63,0.45)";
      c.fillStyle = claro ? "rgba(255,255,255,0.8)" : "#4a5570";
      c.lineWidth = 1.2;
      c.font = "500 10px 'IBM Plex Mono', monospace";
      c.textBaseline = "middle";
      [0, 0.5, 1].forEach(function (k) {
        var yy = 186 - k * 140;
        c.beginPath();
        c.moveTo(178, yy);
        c.lineTo(186, yy);
        c.stroke();
        c.fillText(Math.round(k * 100) + " %", 190, yy);
      });
      c.fillStyle = claro ? "#ffc06b" : "#d07a10";
      c.beginPath();
      c.moveTo(176, y0);
      c.lineTo(169, y0 - 5);
      c.lineTo(169, y0 + 5);
      c.closePath();
      c.fill();
    }
  };

  var copas = new Map();
  var vista = "IntersectionObserver" in window ? new IntersectionObserver(function (entradas) {
    entradas.forEach(function (e) {
      var copa = copas.get(e.target);
      if (copa) copa.visible = e.isIntersecting;
    });
  }) : null;
  var medidor = "ResizeObserver" in window ? new ResizeObserver(function (entradas) {
    entradas.forEach(function (e) {
      var copa = copas.get(e.target);
      if (copa) {
        copa.medir();
        copa.dibujar();
      }
    });
  }) : null;

  function registrarCopa(el) {
    var copa = new Copa(el);
    copas.set(el, copa);
    if (vista) vista.observe(el);
    if (medidor) medidor.observe(el);
    copa.dibujar();
    return copa;
  }

  function copaDe(el) {
    if (!el) return null;
    return copas.get(el) || registrarCopa(el);
  }

  function bucle(ms) {
    copas.forEach(function (copa, el) {
      if (!el.isConnected) {
        copas.delete(el);
        if (vista) vista.unobserve(el);
        if (medidor) medidor.unobserve(el);
        return;
      }
      if (copa.visible) copa.paso(ms);
    });
    window.requestAnimationFrame(bucle);
  }
  window.requestAnimationFrame(bucle);

  window.addEventListener("pointermove", function (e) {
    if (reducido || e.pointerType === "touch") return;
    copas.forEach(function (copa, el) {
      var r = el.getBoundingClientRect();
      copa.inclinacionObjetivo = limitar((e.clientX - (r.left + r.width / 2)) / window.innerWidth * -0.22, -0.11, 0.11);
    });
  }, { passive: true });

  function contar(el) {
    var destino = parseFloat(el.dataset.contar);
    var decimales = parseInt(el.dataset.decimales || "0", 10);
    var sufijo = el.dataset.sufijo || "";
    if (reducido || !isFinite(destino)) return;
    var inicio = null;
    function paso(ts) {
      if (inicio === null) inicio = ts;
      var k = Math.min(1, (ts - inicio) / 1300);
      el.textContent = formato(destino * (1 - Math.pow(1 - k, 3)), decimales) + sufijo;
      if (k < 1) window.requestAnimationFrame(paso);
    }
    window.requestAnimationFrame(paso);
  }

  var aparicion = "IntersectionObserver" in window ? new IntersectionObserver(function (entradas, obs) {
    entradas.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add("visible");
      if (e.target.dataset && e.target.dataset.contar !== undefined) contar(e.target);
      e.target.querySelectorAll && e.target.querySelectorAll("[data-contar]").forEach(function (n) {
        if (!n.dataset.contado) {
          n.dataset.contado = "1";
          contar(n);
        }
      });
      obs.unobserve(e.target);
    });
  }, { rootMargin: "0px 0px -8% 0px" }) : null;

  function escanear() {
    document.querySelectorAll("canvas.copa").forEach(function (el) {
      if (!copas.has(el)) registrarCopa(el);
    });
    document.querySelectorAll(".aparecer:not(.observado), .cifra:not(.observado)").forEach(function (el) {
      el.classList.add("observado");
      if (aparicion) aparicion.observe(el); else el.classList.add("visible");
    });
  }

  var pendienteEscaneo = false;
  new MutationObserver(function () {
    if (pendienteEscaneo) return;
    pendienteEscaneo = true;
    window.requestAnimationFrame(function () {
      pendienteEscaneo = false;
      escanear();
    });
  }).observe(document.body, { childList: true, subtree: true });
  escanear();

  var ruta = window.location.pathname;
  var enCurso = false;

  function cambiarPagina() {
    if (window.location.pathname === ruta) return;
    ruta = window.location.pathname;
    var telon = document.getElementById("telon");
    if (!telon || reducido || enCurso) {
      window.scrollTo(0, 0);
      return;
    }
    enCurso = true;
    var tam = window.innerWidth < 700 ? 92 : 140;
    var columnas = Math.ceil(window.innerWidth / tam);
    var filas = Math.ceil(window.innerHeight / tam);
    telon.style.setProperty("--cols", columnas);
    telon.textContent = "";
    for (var f = 0; f < filas; f++) {
      for (var k = 0; k < columnas; k++) {
        var baldosa = document.createElement("i");
        baldosa.style.setProperty("--d", f + k);
        telon.appendChild(baldosa);
      }
    }
    var duracion = (filas + columnas) * 9 + 240;
    telon.className = "visible tapa";
    window.setTimeout(function () {
      window.scrollTo(0, 0);
      telon.className = "visible destapa";
      window.setTimeout(function () {
        telon.className = "";
        telon.textContent = "";
        enCurso = false;
      }, duracion + 40);
    }, duracion + 160);
  }
  window.addEventListener("_dashprivate_pushstate", cambiarPagina);
  window.addEventListener("popstate", cambiarPagina);

  var BOTELLA = "M24 4h12v8h-1v44c0 12 19 18 19 40v110a8 8 0 0 1-8 8H14a8 8 0 0 1-8-8V96c0-22 19-28 19-40V12h-1z";
  var METRICAS = {
    auc: { campo: "auc_media", mayor: true, decimales: 3, nombre: "AUC" },
    pr: { campo: "pr_auc_media", mayor: true, decimales: 3, nombre: "PR-AUC" },
    brier: { campo: "brier_media", mayor: false, decimales: 3, nombre: "Brier" }
  };

  function construirBotella(contenedor, fila, i) {
    var div = crear("div", "botella");
    div.dataset.modelo = fila.modelo;
    div.setAttribute("role", "img");
    var valor = crear("div", "botella-valor", "");
    var svg = crearSvg("svg", { viewBox: "0 0 60 222", "aria-hidden": "true" });
    var defs = crearSvg("defs");
    var clip = crearSvg("clipPath", { id: "cava-recorte-" + i });
    clip.appendChild(crearSvg("path", { d: BOTELLA }));
    defs.appendChild(clip);
    svg.appendChild(defs);
    var grupo = crearSvg("g", { "clip-path": "url(#cava-recorte-" + i + ")" });
    var liquido = crearSvg("rect", { class: "liquido", x: "0", y: "0", width: "60", height: "230", fill: fila.destacado ? "#1d5fc4" : "#9aa3b2" });
    liquido.style.transform = "translateY(222px)";
    grupo.appendChild(liquido);
    svg.appendChild(grupo);
    svg.appendChild(crearSvg("path", { class: "vidrio", d: BOTELLA }));
    svg.appendChild(crearSvg("rect", { class: "etiqueta-botella", x: "12", y: "128", width: "36", height: "34", rx: "3" }));
    var puesto = crearSvg("text", { x: "30", y: "151", "text-anchor": "middle" });
    svg.appendChild(puesto);
    var nombre = crear("div", "botella-nombre");
    nombre.appendChild(crear("span", "largo", fila.nombre || fila.modelo));
    nombre.appendChild(crear("span", "corto", fila.corto));
    div.appendChild(valor);
    div.appendChild(svg);
    div.appendChild(nombre);
    contenedor.appendChild(div);
    return div;
  }

  function cava(datos, metrica) {
    var contenedor = document.getElementById("cava");
    if (!contenedor || !datos) return [NO(), NO()];
    var m = METRICAS[metrica] || METRICAS.auc;
    var filas = datos.filas;
    var base = metrica === "auc" ? 0.5 : (metrica === "pr" ? datos.prevalencia : datos.prevalencia * (1 - datos.prevalencia));
    var validos = filas.filter(function (f) { return Number.isFinite(f[m.campo]); });
    var mejor = m.mayor ? Math.max.apply(null, validos.map(function (f) { return f[m.campo]; })) : Math.min.apply(null, validos.map(function (f) { return f[m.campo]; }));
    var orden = filas.slice().sort(function (a, b) {
      var va = a[m.campo], vb = b[m.campo];
      if (!Number.isFinite(va)) return 1;
      if (!Number.isFinite(vb)) return -1;
      return m.mayor ? vb - va : va - vb;
    });
    if (!contenedor.querySelector(".botella")) {
      filas.forEach(function (f, i) { construirBotella(contenedor, f, i); });
    }
    orden.forEach(function (f, pos) {
      var div = contenedor.querySelector('.botella[data-modelo="' + CSS.escape(f.modelo) + '"]');
      if (!div) return;
      var v = f[m.campo];
      var nivel = Number.isFinite(v) ? limitar((m.mayor ? v - base : base - v) / Math.abs(mejor - base || 1), 0, 1) * 0.88 : 0;
      div.style.setProperty("--pos", pos);
      div.querySelector(".liquido").style.transform = "translateY(" + (214 - nivel * 196).toFixed(1) + "px)";
      var ok = Number.isFinite(v);
      div.querySelector(".botella-valor").textContent = ok ? v.toFixed(m.decimales) : "—";
      div.querySelector("text").textContent = ok ? String(pos + 1) : "—";
      var texto = f.modelo + ": " + m.nombre + " " + (ok ? v.toFixed(m.decimales) + ", puesto " + (pos + 1) + " de " + filas.length : "no disponible");
      div.setAttribute("aria-label", texto);
      div.title = texto;
    });
    var azar = datos.prevalencia;
    var explica = {
      auc: "AUC: de cada 100 parejas de vinos (uno bueno y uno que no), en cuántas el modelo le da más puntaje al bueno. 0.5 es adivinar y 1 es no fallar nunca.",
      pr: "PR-AUC: qué tan limpia es la lista de vinos que el modelo marca como buenos. Marcando al azar valdría " + formato(azar, 2) + ", la proporción de vinos buenos.",
      brier: "Brier: cuánto se equivocan sus probabilidades. Aquí menos es mejor: decir siempre " + Math.round(azar * 100) + " % daría " + formato(azar * (1 - azar), 3) + ". SVM no entrega probabilidades y queda al final."
    };
    return ["Modelos ordenados por " + m.nombre, explica[metrica] || explica.auc];
  }

  function nucleo(indice, desplazamiento, escala) {
    var ANCHOS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 5, Infinity];
    var h = ANCHOS[indice] === undefined ? 1 : ANCHOS[indice];
    var centro = 7 + (desplazamiento || 0) * escala;
    var umbrales = [4, 5, 6, 7, 8, 9];
    var pesos;
    if (!isFinite(h)) {
      pesos = umbrales.map(function () { return 1; });
    } else if (h === 0) {
      var cercano = umbrales.reduce(function (a, b) { return Math.abs(b - centro) < Math.abs(a - centro) ? b : a; });
      pesos = umbrales.map(function (t) { return t === cercano ? 1 : 0; });
    } else {
      pesos = umbrales.map(function (t) { return Math.exp(-0.5 * Math.pow((t - centro) / (h * escala), 2)); });
    }
    var suma = pesos.reduce(function (a, b) { return a + b; }, 0);
    pesos = pesos.map(function (w) { return w / suma; });
    var tope = Math.max.apply(null, pesos);
    document.querySelectorAll("#nucleo .nucleo-columna").forEach(function (col, i) {
      col.style.setProperty("--p", (pesos[i] / tope).toFixed(3));
      col.querySelector(".nucleo-peso").textContent = Math.round(pesos[i] * 100) + " %";
      col.classList.toggle("foco", pesos[i] === tope);
    });
    if (h === 0) return "Ancho 0: el modelo solo escucha al umbral «nota ≥ 7». Es lo mismo que aprender con la etiqueta binaria bueno / no bueno: los vinos de 6 y de 3 cuentan igual.";
    if (!isFinite(h)) return "Ancho infinito: todos los umbrales pesan lo mismo, como una regresión sobre la nota. Separar un 3 de un 4 vale tanto como separar un 6 de un 7, aunque solo lo segundo importa para el evento.";
    var vecinos = Math.round((pesos[2] + pesos[4]) * 100);
    return "Ancho " + formato(h, 2) + ": el umbral «nota ≥ 7» pesa " + Math.round(pesos[3] * 100) + " % y sus vecinos (≥ 6 y ≥ 8) suman " + vecinos + " %. Así el modelo aprovecha la información de los vinos de 6 y de 8 sin perder de vista el evento.";
  }

  function decision(datos, modelo, umbral, costo) {
    var no = NO();
    var vacio = [no, no, no, no, no, no, no, no, no, no];
    if (!datos || !modelo || !datos.modelos[modelo]) return vacio;
    var i = Math.round(limitar(umbral, 0, 1) * 100);
    var m = datos.modelos[modelo];
    var P = datos.positivos, N = datos.negativos, total = P + N;
    var vp = m.vp[i], fp = m.fp[i], fn = P - vp, vn = N - fp;
    var precision = vp + fp > 0 ? vp / (vp + fp) : NaN;
    var sensibilidad = vp / P;
    var f1 = isFinite(precision) && precision + sensibilidad > 0 ? 2 * precision * sensibilidad / (precision + sensibilidad) : NaN;
    var nvp = Math.round(100 * vp / total), nfn = Math.round(100 * fn / total), nfp = Math.round(100 * fp / total);
    var nvn = Math.max(0, 100 - nvp - nfn - nfp);
    var mosaico = document.getElementById("mosaico");
    if (mosaico) {
      if (mosaico.children.length !== 100) {
        mosaico.textContent = "";
        for (var k = 0; k < 100; k++) {
          var teja = crear("div", "teja vn");
          teja.style.setProperty("--d", k);
          mosaico.appendChild(teja);
        }
      }
      var clases = [];
      for (var a = 0; a < nvp; a++) clases.push(["vp", "✓"]);
      for (var b = 0; b < nfn; b++) clases.push(["fn", "○"]);
      for (var c = 0; c < nfp; c++) clases.push(["fp", "!"]);
      while (clases.length < 100) clases.push(["vn", ""]);
      clases.slice(0, 100).forEach(function (par, j) {
        var t = mosaico.children[j];
        t.className = "teja " + par[0];
        t.textContent = par[1];
      });
    }
    var graficas = [["g-roc", fp / N, sensibilidad], ["g-pr", sensibilidad, isFinite(precision) ? precision : null]];
    window.setTimeout(function () {
      graficas.forEach(function (g) {
        var gd = document.querySelector("#" + g[0] + " .js-plotly-plot");
        if (gd && window.Plotly && gd.data && gd.data.length > 2) window.Plotly.restyle(gd, { x: [[g[1]]], y: [[g[2]]] }, [2]);
      });
    }, 120);
    var mejorCosto = Infinity, mejorUmbral = 0.5;
    datos.umbrales.forEach(function (t, j) {
      var coste = costo * (P - m.vp[j]) + m.fp[j];
      if (coste < mejorCosto) {
        mejorCosto = coste;
        mejorUmbral = t;
      }
    });
    var jm = Math.round(mejorUmbral * 100);
    var sensMejor = m.vp[jm] / P;
    var catas = m.vp[jm] + m.fp[jm];
    var sugerencia = "Conviene un umbral de " + Math.round(mejorUmbral * 100) + " %. Si perder un vino bueno es " + formato(costo, 0) +
      " veces peor que catar uno que no lo es, ese es el más barato: encuentra " + Math.round(sensMejor * 100) +
      " de cada 100 vinos buenos y manda a cata " + formato(catas, 0) + " de los " + formato(total, 0) + " vinos.";
    var pct = function (v) { return isFinite(v) ? Math.round(v * 100) + " %" : "—"; };
    var detalle = "En los " + formato(total, 0) + " vinos: " + formato(vp, 0) + " buenos encontrados, " + formato(fn, 0) + " perdidos, " +
      formato(fp, 0) + " falsas alarmas y " + formato(vn, 0) + " bien descartados.";
    return [String(nvp), String(nfn), String(nfp), String(nvn), pct(precision), pct(sensibilidad), isFinite(f1) ? formato(f1, 2) : "—",
      "Con " + Math.round(umbral * 100) + " %: el modelo llama «bueno» a todo vino al que le da " + Math.round(umbral * 100) + " % o más de probabilidad.",
      sugerencia, detalle];
  }

  var Juego = {
    iniciar: function (raizJuego, datos) {
      this.raiz = raizJuego;
      this.datos = datos;
      this.nuevaPartida();
    },
    nuevaPartida: function () {
      var buenos = barajar(this.datos.buenos.slice());
      var malos = barajar(this.datos.malos.slice());
      this.rondas = [];
      for (var i = 0; i < 10; i++) this.rondas.push(Math.random() < 0.5 ? [buenos[i], malos[i]] : [malos[i], buenos[i]]);
      this.i = 0;
      this.tu = 0;
      this.modelo = 0;
      this.ronda();
    },
    marcador: function () {
      var m = crear("div", "juego-marcador");
      m.appendChild(crear("div", "juego-ronda", "Ronda " + (this.i + 1) + " de " + this.rondas.length));
      var p = crear("div", "juego-puntos");
      var tu = crear("span", "", "Tú");
      tu.appendChild(crear("strong", "", String(this.tu)));
      var mo = crear("span", "", "Modelo");
      mo.appendChild(crear("strong", "", String(this.modelo)));
      p.appendChild(tu);
      p.appendChild(mo);
      m.appendChild(p);
      return m;
    },
    lado: function (vino, letra, indice) {
      var self = this;
      var tarjeta = crear("article", "ronda-vino elegible");
      tarjeta.tabIndex = 0;
      tarjeta.setAttribute("role", "button");
      tarjeta.setAttribute("aria-label", "Elegir el vino " + letra + " como el mejor");
      tarjeta.appendChild(crear("div", "ronda-letra", "Vino " + letra));
      var tipo = crear("div", "ronda-tipo");
      tipo.appendChild(crear("span", "gota " + (vino.t === "red" ? "gota-tinto" : "gota-blanco")));
      tipo.appendChild(document.createTextNode(vino.t === "red" ? "Tinto" : "Blanco"));
      tarjeta.appendChild(tipo);
      var lienzo = crear("canvas", "copa");
      lienzo.dataset.nivel = "0";
      lienzo.dataset.tipo = vino.t;
      lienzo.setAttribute("role", "img");
      lienzo.setAttribute("aria-label", "Copa del vino " + letra + ", vacía hasta revelar la probabilidad del modelo");
      tarjeta.appendChild(lienzo);
      tarjeta.appendChild(crear("div", "ronda-revelado"));
      var elegir = function () { self.elegir(indice); };
      tarjeta.addEventListener("click", elegir);
      tarjeta.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          elegir();
        }
      });
      return tarjeta;
    },
    comparacion: function (a, b) {
      var tabla = crear("div", "comparacion");
      var pistas = this.datos.pistas || [];
      this.datos.escala.forEach(function (v, k) {
        var fila = crear("div", "comparacion-fila");
        [a, null, b].forEach(function (vino, j) {
          if (vino === null) {
            var nombre = crear("div", "comparacion-nombre", v.etiqueta);
            nombre.appendChild(crear("small", "", v.unidad));
            var pista = pistas[k];
            if (pista && pista.sentido !== 0) {
              nombre.appendChild(crear("span", "pista" + (pista.clave ? " fuerte" : ""), pista.sentido > 0 ? "▲ más alto en los buenos" : "▼ más bajo en los buenos"));
            }
            fila.appendChild(nombre);
            return;
          }
          var lado = crear("div", "comparacion-lado " + (j === 0 ? "a" : "b"));
          lado.appendChild(crear("span", "", formato(vino.v[k], vino.v[k] < 1 ? 3 : (vino.v[k] < 20 ? 1 : 0))));
          var barra = crear("div", "comparacion-barra");
          var marca = crear("i");
          marca.style.setProperty("--p", limitar((vino.v[k] - v.bajo) / (v.alto - v.bajo), 0.03, 1).toFixed(3));
          barra.appendChild(marca);
          lado.appendChild(barra);
          fila.appendChild(lado);
        });
        tabla.appendChild(fila);
      });
      return tabla;
    },
    ronda: function () {
      var par = this.rondas[this.i];
      this.raiz.textContent = "";
      this.raiz.appendChild(this.marcador());
      var tablero = crear("div", "ronda-tablero");
      tablero.appendChild(this.lado(par[0], "A", 0));
      tablero.appendChild(this.comparacion(par[0], par[1]));
      tablero.appendChild(this.lado(par[1], "B", 1));
      this.raiz.appendChild(tablero);
      var mensaje = crear("div", "juego-mensaje");
      mensaje.setAttribute("aria-live", "polite");
      mensaje.textContent = "Uno de los dos obtuvo 7 o más en la cata y el otro no. ¿Cuál es el bueno? Toca su copa.";
      this.raiz.appendChild(mensaje);
      this.respondida = false;
    },
    elegir: function (indice) {
      if (this.respondida) return;
      this.respondida = true;
      var par = this.rondas[this.i];
      var bueno = par[0].n >= 7 ? 0 : 1;
      var modelo = par[0].p >= par[1].p ? 0 : 1;
      if (indice === bueno) this.tu += 1;
      if (modelo === bueno) this.modelo += 1;
      var tarjetas = this.raiz.querySelectorAll(".ronda-vino");
      tarjetas.forEach(function (t, k) {
        t.classList.remove("elegible");
        if (k === indice) t.classList.add("elegido");
        t.removeAttribute("role");
        t.tabIndex = -1;
        var vino = par[k];
        var lienzo = t.querySelector("canvas.copa");
        var copa = copaDe(lienzo);
        if (copa) copa.fijar(vino.p, vino.t);
        lienzo.setAttribute("aria-label", "Copa del vino " + (k === 0 ? "A" : "B") + " llena al " + Math.round(vino.p * 100) + " %, la probabilidad del modelo");
        var revelado = t.querySelector(".ronda-revelado");
        revelado.textContent = "";
        var nota = crear("div", "ronda-nota", "Nota real: " + vino.n + (vino.n >= 7 ? " · bueno" : " · no bueno"));
        revelado.appendChild(nota);
        revelado.appendChild(crear("div", "", "El modelo le dio " + Math.round(vino.p * 100) + " %"));
      });
      this.raiz.querySelector(".juego-marcador").replaceWith(this.marcador());
      var tuAcierto = indice === bueno, moAcierto = modelo === bueno;
      var frase = (tuAcierto ? "¡Acertaste! " : "Esta vez no. ") + (moAcierto ? "El modelo también eligió el vino " + (modelo === 0 ? "A" : "B") + "." : "El modelo se equivocó: le dio más puntaje al vino que no es bueno.");
      var mensaje = this.raiz.querySelector(".juego-mensaje");
      mensaje.textContent = frase + " ";
      var self = this;
      var siguiente = crear("button", "boton", this.i + 1 < this.rondas.length ? "Siguiente pareja →" : "Ver el resultado");
      siguiente.type = "button";
      siguiente.addEventListener("click", function () {
        self.i += 1;
        if (self.i < self.rondas.length) self.ronda(); else self.final();
      });
      mensaje.appendChild(document.createElement("br"));
      mensaje.appendChild(siguiente);
      siguiente.focus({ preventScroll: true });
    },
    final: function () {
      var self = this;
      this.raiz.textContent = "";
      var caja = crear("div", "juego-final");
      caja.appendChild(crear("div", "etiqueta-seccion", "Resultado"));
      caja.appendChild(crear("h2", "", this.tu > this.modelo ? "Le ganaste al modelo" : (this.tu === this.modelo ? "Empate con el modelo" : "El modelo cató mejor")));
      var resultado = crear("div", "juego-resultado");
      [["Tú", this.tu], ["Modelo", this.modelo]].forEach(function (p) {
        var d = crear("div");
        d.appendChild(crear("strong", "", p[1] + "/10"));
        d.appendChild(document.createTextNode(p[0]));
        resultado.appendChild(d);
      });
      caja.appendChild(resultado);
      caja.appendChild(crear("p", "", "Elegir el mejor de dos vinos es exactamente lo que mide el AUC. Con todas las parejas posibles de los 5320 vinos, el modelo original acierta en el " +
        Math.round(this.datos.auc * 100) + " % de los casos, usando predicciones de modelos que nunca vieron esos vinos. En 10 parejas el azar pesa mucho: juega otra vez y mira cómo se reparte."));
      var otra = crear("button", "boton", "Jugar otra vez");
      otra.type = "button";
      otra.addEventListener("click", function () { self.nuevaPartida(); });
      caja.appendChild(otra);
      this.raiz.appendChild(caja);
    }
  };

  document.addEventListener("change", function (e) {
    if (e.target && e.target.id === "juego-pistas") {
      var juego = document.getElementById("juego");
      if (juego) juego.classList.toggle("con-pistas", e.target.checked);
    }
  });

  function filtrarGlosario(texto) {
    var buscado = texto.trim().toLowerCase();
    var visibles = 0;
    document.querySelectorAll("#glosario-lista .glosario-bloque").forEach(function (bloque) {
      var enBloque = 0;
      bloque.querySelectorAll(".glosario-entrada").forEach(function (entrada) {
        var coincide = !buscado || entrada.textContent.toLowerCase().indexOf(buscado) !== -1;
        entrada.hidden = !coincide;
        if (coincide) enBloque += 1;
      });
      bloque.hidden = enBloque === 0;
      visibles += enBloque;
    });
    var vacio = document.getElementById("glosario-vacio");
    if (vacio) vacio.hidden = visibles > 0;
  }

  document.addEventListener("input", function (e) {
    if (e.target && e.target.id === "glosario-buscar") filtrarGlosario(e.target.value);
  });

  function abrirGlosario(clave) {
    var boton = document.getElementById("glosario-boton");
    if (!boton) return;
    if (!document.querySelector(".panel-glosario.show")) boton.click();
    var intentos = 0;
    var buscar = function () {
      var entrada = document.querySelector('#glosario-lista .glosario-entrada[data-termino="' + CSS.escape(clave) + '"]');
      if (!entrada) {
        if (intentos++ < 20) window.setTimeout(buscar, 60);
        return;
      }
      var campo = document.getElementById("glosario-buscar");
      if (campo && campo.value) {
        campo.value = "";
        filtrarGlosario("");
      }
      document.querySelectorAll(".glosario-entrada.resaltada").forEach(function (el) { el.classList.remove("resaltada"); });
      entrada.classList.add("resaltada");
      entrada.scrollIntoView({ block: "center", behavior: reducido ? "auto" : "smooth" });
    };
    window.setTimeout(buscar, 80);
  }

  document.addEventListener("click", function (e) {
    var palabra = e.target && e.target.closest ? e.target.closest(".termino[data-clave]") : null;
    if (palabra) abrirGlosario(palabra.dataset.clave);
  });
  document.addEventListener("keydown", function (e) {
    if ((e.key === "Enter" || e.key === " ") && e.target && e.target.matches && e.target.matches(".termino[data-clave]")) {
      e.preventDefault();
      abrirGlosario(e.target.dataset.clave);
    }
  });

  window.Cata = { copa: copaDe };

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    cata: {
      copa: function (datos) {
        if (!datos) return NO();
        var lienzo = document.getElementById("copa-pred");
        var copa = copaDe(lienzo);
        if (copa) copa.fijar(datos.p, datos.tipo);
        var numero = document.getElementById("prob-valor");
        if (numero) {
          var desde = parseFloat(numero.dataset.valor || "0"), hasta = datos.p * 100;
          numero.dataset.valor = String(hasta);
          if (reducido) {
            numero.textContent = formato(hasta, 0) + " %";
          } else {
            var inicio = null;
            var paso = function (ts) {
              if (inicio === null) inicio = ts;
              var k = Math.min(1, (ts - inicio) / 900);
              numero.textContent = formato(desde + (hasta - desde) * (1 - Math.pow(1 - k, 3)), 0) + " %";
              if (k < 1) window.requestAnimationFrame(paso);
            };
            window.requestAnimationFrame(paso);
          }
        }
        return "Copa llena al " + Math.round(datos.p * 100) + " %: probabilidad de que el vino sea bueno según " + datos.modelo;
      },
      cava: cava,
      ir: function () {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered.length || !ctx.triggered[0].value) return NO();
        var destino = JSON.parse(ctx.triggered[0].prop_id.split(".n_clicks")[0]).pestana;
        var pestanas = document.getElementById("pestanas");
        if (pestanas) pestanas.scrollIntoView({ block: "start", behavior: reducido ? "auto" : "smooth" });
        return destino;
      },
      recordar: function (pestana) {
        if (pestana && window.location.pathname === "/modelos") {
          var destino = "/modelos?pestana=" + encodeURIComponent(pestana);
          if (window.location.pathname + window.location.search !== destino) window.history.replaceState(window.history.state, "", destino);
        }
        return NO();
      },
      nucleo: nucleo,
      decision: decision,
      juego: function (datos) {
        var el = document.getElementById("juego");
        if (!el || !datos) return NO();
        Juego.iniciar(el, datos);
        return "Juego de cata a ciegas";
      },
      sincronizar: function (numero, deslizador) {
        var ctx = window.dash_clientside.callback_context;
        var disparo = ctx.triggered.length ? ctx.triggered[0].prop_id : "";
        if (disparo.indexOf('"tipo":"sl"') !== -1) return [NO(), deslizador];
        var v = parseFloat(numero);
        if (!isFinite(v)) return [NO(), NO()];
        return [v, NO()];
      }
    }
  });
})();
