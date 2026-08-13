/* Interactive constraint panel.
 *
 * Every configuration was computed by tools/build_constraint_panel.py, which
 * calls advance_beamlab.constraint_demo. Nothing is simulated here; the page
 * only displays what that produced. See doc/panel.rst.
 */
(function () {
  "use strict";

  var MOUNT = "constraint-panel";
  var METHOD_LABEL = {
    lcmv: "LCMV",
    mcmv: "MCMV",
    recipsiicos: "ReciPSIICOS",
    abmc: "ABMC",
  };

  /* Luminance rises monotonically from end to end, so the map reads the same
   * way on the light and the dark version of the page, and no part of the ramp
   * disappears into either background. */
  var RAMP = [
    [0.0, 60, 75, 105],
    [0.25, 40, 120, 170],
    [0.5, 45, 175, 175],
    [0.75, 180, 210, 100],
    [1.0, 255, 240, 170],
  ];

  function rampColour(t) {
    t = Math.max(0, Math.min(1, t));
    for (var i = 1; i < RAMP.length; i++) {
      if (t <= RAMP[i][0]) {
        var a = RAMP[i - 1],
          b = RAMP[i];
        var span = b[0] - a[0] || 1;
        var f = (t - a[0]) / span;
        return [
          Math.round(a[1] + f * (b[1] - a[1])),
          Math.round(a[2] + f * (b[2] - a[2])),
          Math.round(a[3] + f * (b[3] - a[3])),
        ];
      }
    }
    return [255, 240, 170];
  }

  function css(el, name) {
    return getComputedStyle(el).getPropertyValue(name).trim();
  }

  function fitCanvas(canvas, height) {
    var ratio = window.devicePixelRatio || 1;
    var width = canvas.clientWidth || 320;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.height = height + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx: ctx, w: width, h: height };
  }

  function decode(blob) {
    var binary = atob(blob);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    if (typeof DecompressionStream === "undefined") {
      return Promise.reject(new Error("DecompressionStream unavailable"));
    }
    var stream = new Blob([bytes])
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).arrayBuffer();
  }

  function build(root, P, buf) {
    /* Throws rather than guessing. An earlier build wrote the dtype as
     * "<class 'numpy.int16'>", and a reader that quietly fell back to bytes
     * produced a page full of plausible, wrong curves. */
    var TYPES = { int16: Int16Array, uint8: Uint8Array };
    function view(entry) {
      var T = TYPES[entry.dtype];
      if (!T) throw new Error("unknown array type '" + entry.dtype + "'");
      return new T(buf, entry.offset, entry.length);
    }

    var pos = view(P.positions);
    var cortex = view(P.cortex);
    var maps = view(P.maps);
    var trueTcs = view(P.true_tcs);
    var recon = view(P.reconstructed);
    var sensor = view(P.sensor);
    var gscale = P.geometry_scale;
    var wscale = P.waveform_scale;
    var nT = P.n_times;
    var nSrc = P.n_sources;
    var nTraces = P.n_sensor_traces;

    /* Look the indices up rather than assuming the build wrote them in a
     * particular order. */
    var sceneOf = {};
    P.scenes.forEach(function (s, i) {
      sceneOf[s.requested.sep + "|" + s.requested.corr + "|" + s.requested.snr] = i;
    });
    var resultOf = {};
    P.results.forEach(function (r, i) {
      resultOf[r.scene + "|" + r.method] = i;
    });

    var state = {
      method: 0,
      corr: P.correlations.indexOf(0.95) >= 0 ? P.correlations.indexOf(0.95) : 0,
      sep: 1,
      snr: 1,
      az: -0.6,
      el: 0.25,
    };

    root.innerHTML =
      '<div class="cp-controls">' +
      '<div class="cp-control"><label>Method</label><div class="cp-methods" id="cp-methods"></div></div>' +
      '<div class="cp-control"><label>Source correlation</label>' +
      '<input type="range" id="cp-corr" min="0" max="' +
      (P.correlations.length - 1) +
      '" step="1"><div class="cp-value" id="cp-corr-v"></div></div>' +
      '<div class="cp-control"><label>Separation</label>' +
      '<input type="range" id="cp-sep" min="0" max="' +
      (P.separations.length - 1) +
      '" step="1"><div class="cp-value" id="cp-sep-v"></div></div>' +
      '<div class="cp-control"><label>Sensor SNR</label>' +
      '<input type="range" id="cp-snr" min="0" max="' +
      (P.snrs.length - 1) +
      '" step="1"><div class="cp-value" id="cp-snr-v"></div></div>' +
      "</div>" +
      '<div class="cp-grid">' +
      '<div class="cp-card"><h4>Where the method says the sources are</h4>' +
      '<p class="cp-hint">Drag to rotate. Rings mark the two simulated sources; ' +
      "they are drawn unfilled so they cannot hide the peak underneath. Colour is " +
      "the localiser value by <em>rank</em>, because the four methods produce very " +
      "different value distributions and any common scaling would flatten one of them." +
      '</p><canvas id="cp-brain"></canvas>' +
      '<div class="cp-legend"><span>low</span><div class="cp-ramp" id="cp-ramp"></div><span>high</span></div></div>' +
      '<div class="cp-card"><h4>The constraint table</h4>' +
      '<p class="cp-hint">Row <em>i</em> is the filter for source <em>i</em>; column ' +
      "<em>j</em> is the gain it has at source <em>j</em>. The diagonal is what the " +
      "distortionless constraint pins to one. The off-diagonal is the part LCMV " +
      "leaves free, and it is where the cancellation lives.</p>" +
      '<table class="cp-table" id="cp-gains"></table>' +
      '<div class="cp-readout" id="cp-readout"></div></div>' +
      '<div class="cp-card"><h4>What the sensors record</h4>' +
      '<p class="cp-hint">Twenty gradiometers, evenly spaced across the array. This ' +
      "is the only thing a beamformer ever sees.</p>" +
      '<canvas id="cp-sensor"></canvas></div>' +
      '<div class="cp-card"><h4>True and recovered source waveforms</h4>' +
      '<p class="cp-hint">Both sources, on one scale. Grey is the simulated ' +
      "waveform, colour is what the method returned; the gap between them is the " +
      "amplitude the filter lost.</p>" +
      '<canvas id="cp-wave"></canvas></div>' +
      "</div>";

    var ramp = root.querySelector("#cp-ramp");
    var stops = RAMP.map(function (s) {
      return "rgb(" + s[1] + "," + s[2] + "," + s[3] + ") " + s[0] * 100 + "%";
    });
    ramp.style.background = "linear-gradient(to right," + stops.join(",") + ")";

    var methodsBox = root.querySelector("#cp-methods");
    P.methods.forEach(function (m, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = METHOD_LABEL[m] || m;
      b.addEventListener("click", function () {
        state.method = i;
        render();
      });
      methodsBox.appendChild(b);
    });

    ["corr", "sep", "snr"].forEach(function (key) {
      var input = root.querySelector("#cp-" + key);
      input.value = state[key];
      input.addEventListener("input", function () {
        state[key] = parseInt(input.value, 10);
        render();
      });
    });

    var brain = root.querySelector("#cp-brain");
    var dragging = false,
      lastX = 0,
      lastY = 0;
    brain.addEventListener("pointerdown", function (e) {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      brain.setPointerCapture(e.pointerId);
    });
    brain.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      state.az += (e.clientX - lastX) * 0.01;
      state.el += (e.clientY - lastY) * 0.01;
      state.el = Math.max(-1.4, Math.min(1.4, state.el));
      lastX = e.clientX;
      lastY = e.clientY;
      drawBrain();
    });
    brain.addEventListener("pointerup", function (e) {
      dragging = false;
      brain.releasePointerCapture(e.pointerId);
    });

    function current() {
      var s =
        sceneOf[
          P.separations[state.sep] +
            "|" +
            P.correlations[state.corr] +
            "|" +
            P.snrs[state.snr]
        ];
      var r = resultOf[s + "|" + P.methods[state.method]];
      return { scene: s, result: r };
    }

    function rotated(x, y, z) {
      var ca = Math.cos(state.az),
        sa = Math.sin(state.az);
      var X = x * ca - y * sa,
        Y = x * sa + y * ca;
      var ce = Math.cos(state.el),
        se = Math.sin(state.el);
      return [X, Y * ce - z * se, Y * se + z * ce];
    }

    function drawBrain() {
      var c = fitCanvas(brain, 280);
      var ctx = c.ctx;
      var cur = current();
      var scene = P.scenes[cur.scene];
      ctx.clearRect(0, 0, c.w, c.h);
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);

      var pts = [];
      var i, p;
      for (i = 0; i < cortex.length / 3; i++) {
        p = rotated(cortex[i * 3] / gscale, cortex[i * 3 + 1] / gscale, cortex[i * 3 + 2] / gscale);
        pts.push([p[0], p[2], p[1], -1, 0]);
      }
      var base = cur.result * nSrc;
      for (i = 0; i < nSrc; i++) {
        p = rotated(pos[i * 3] / gscale, pos[i * 3 + 1] / gscale, pos[i * 3 + 2] / gscale);
        pts.push([p[0], p[2], p[1], i, maps[base + i] / 255]);
      }

      var xs = pts.map(function (q) { return q[0]; });
      var ys = pts.map(function (q) { return q[1]; });
      var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
      var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
      var scale = 0.86 * Math.min(c.w / (maxX - minX || 1), c.h / (maxY - minY || 1));
      var cx = c.w / 2 - ((minX + maxX) / 2) * scale;
      var cy = c.h / 2 + ((minY + maxY) / 2) * scale;

      pts.sort(function (a, b) { return a[2] - b[2]; });
      var backdrop = css(root, "--cp-grid");
      for (i = 0; i < pts.length; i++) {
        var q = pts[i];
        var X = cx + q[0] * scale,
          Y = cy - q[1] * scale;
        if (q[3] < 0) {
          ctx.fillStyle = backdrop;
          ctx.globalAlpha = 0.32;
          ctx.fillRect(X, Y, 1.4, 1.4);
        } else {
          var rgb = rampColour(q[4]);
          ctx.globalAlpha = 0.35 + 0.65 * q[4];
          ctx.fillStyle = "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
          ctx.beginPath();
          ctx.arc(X, Y, 1.6 + 3.4 * q[4] * q[4], 0, 6.2832);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;

      /* Unfilled rings, so the marker cannot cover the peak it points at. */
      ctx.lineWidth = 2;
      ctx.strokeStyle = css(root, "--cp-warm");
      scene.sources.forEach(function (src) {
        var q = rotated(pos[src * 3] / gscale, pos[src * 3 + 1] / gscale, pos[src * 3 + 2] / gscale);
        ctx.beginPath();
        ctx.arc(cx + q[0] * scale, cy - q[2] * scale, 8, 0, 6.2832);
        ctx.stroke();
      });
    }

    function drawTraces(canvas, data, offset, count, height, colours) {
      var c = fitCanvas(canvas, height);
      var ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var pad = 6;
      var lane = (c.h - 2 * pad) / count;
      for (var k = 0; k < count; k++) {
        ctx.beginPath();
        ctx.strokeStyle = colours(k);
        ctx.lineWidth = 1.1;
        for (var t = 0; t < nT; t++) {
          var v = data[offset + k * nT + t] / wscale;
          var X = pad + (t / (nT - 1)) * (c.w - 2 * pad);
          var Y = pad + lane * (k + 0.5) - v * lane * 0.45;
          if (t === 0) ctx.moveTo(X, Y);
          else ctx.lineTo(X, Y);
        }
        ctx.stroke();
      }
    }

    function drawSensor() {
      var cur = current();
      drawTraces(
        root.querySelector("#cp-sensor"),
        sensor,
        cur.scene * nTraces * nT,
        nTraces,
        200,
        function () { return css(root, "--cp-muted"); }
      );
    }

    function drawWave() {
      var cur = current();
      var canvas = root.querySelector("#cp-wave");
      var c = fitCanvas(canvas, 200);
      var ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var pad = 8;
      var lane = (c.h - 2 * pad) / 2;
      /* True and recovered share one scale per scene, so a shrunken
       * reconstruction looks shrunken instead of being renormalised back. */
      for (var k = 0; k < 2; k++) {
        [
          [trueTcs, cur.scene * 2 * nT, css(root, "--cp-true"), 1.1],
          [recon, cur.result * 2 * nT, css(root, "--cp-warm"), 1.6],
        ].forEach(function (spec) {
          ctx.beginPath();
          ctx.strokeStyle = spec[2];
          ctx.lineWidth = spec[3];
          for (var t = 0; t < nT; t++) {
            var v = spec[0][spec[1] + k * nT + t] / wscale;
            var X = pad + (t / (nT - 1)) * (c.w - 2 * pad);
            var Y = pad + lane * (k + 0.5) - v * lane * 0.4;
            if (t === 0) ctx.moveTo(X, Y);
            else ctx.lineTo(X, Y);
          }
          ctx.stroke();
        });
      }
    }

    function drawTable() {
      var cur = current();
      var res = P.results[cur.result];
      var g = res.gains;
      var html =
        "<tr><th></th><th>at source 1</th><th>at source 2</th></tr>";
      for (var i = 0; i < 2; i++) {
        html += "<tr><th>filter " + (i + 1) + "</th>";
        for (var j = 0; j < 2; j++) {
          var v = g[i][j];
          var mag = Math.min(1, Math.abs(v));
          var col =
            v >= 0
              ? "rgba(200,69,46," + (0.12 + 0.55 * mag) + ")"
              : "rgba(33,102,172," + (0.12 + 0.55 * mag) + ")";
          html +=
            '<td style="background:' + col + '">' + (v >= 0 ? "+" : "") + v.toFixed(3) + "</td>";
        }
        html += "</tr>";
      }
      root.querySelector("#cp-gains").innerHTML = html;

      var off = Math.abs(g[0][1]);
      var ratio = res.amplitude_ratio[0];
      var err = res.peak_errors[0];
      root.querySelector("#cp-readout").innerHTML =
        '<div><span>Off-diagonal gain</span><strong>' +
        (g[0][1] >= 0 ? "+" : "") + g[0][1].toFixed(3) +
        "</strong></div>" +
        '<div><span>Amplitude recovered</span><strong>' +
        (ratio * 100).toFixed(0) + "%</strong></div>" +
        '<div><span>Localisation error</span><strong>' +
        err.toFixed(0) + " mm</strong></div>" +
        '<div><span>Cancelling?</span><strong>' +
        (off > 0.2 ? "yes" : "no") + "</strong></div>";
    }

    function render() {
      P.methods.forEach(function (m, i) {
        methodsBox.children[i].setAttribute(
          "aria-pressed", i === state.method ? "true" : "false"
        );
      });
      var scene = P.scenes[current().scene];
      root.querySelector("#cp-corr-v").textContent =
        "r = " + scene.correlation.toFixed(2);
      root.querySelector("#cp-sep-v").textContent =
        (scene.separation * 100).toFixed(1) + " cm apart";
      root.querySelector("#cp-snr-v").textContent =
        "SNR " + P.snrs[state.snr];
      drawBrain();
      drawSensor();
      drawWave();
      drawTable();
    }

    var pending;
    window.addEventListener("resize", function () {
      clearTimeout(pending);
      pending = setTimeout(render, 120);
    });
    new MutationObserver(render).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    render();
  }

  function start() {
    var root = document.getElementById(MOUNT);
    if (!root) return;
    var P = window.CONSTRAINT_PANEL;
    if (!P) {
      root.innerHTML =
        '<div class="cp-status">The panel data file did not load.</div>';
      return;
    }
    root.innerHTML = '<div class="cp-status">Loading the precomputed scenes…</div>';
    decode(P.blob).then(
      function (buf) {
        try {
          build(root, P, buf);
        } catch (err) {
          root.innerHTML =
            '<div class="cp-status">The panel failed to start: ' +
            err.message +
            "</div>";
        }
      },
      function (err) {
        root.innerHTML =
          '<div class="cp-status">The panel could not decode its data (' +
          err.message +
          "). It needs a browser with DecompressionStream.</div>";
      }
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
