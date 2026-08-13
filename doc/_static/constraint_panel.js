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

  /* What each method is actually solving. Taken from the implementations:
   * LCMV and MCMV from _mcmv.py, ABMC from make_abmc's docstring (Shirani 2024,
   * Eqs. 14-19), ReciPSIICOS from _recipsiicos.py. The point of showing them
   * side by side is that only one line differs between the first two. */
  var EQUATIONS = {
    lcmv: {
      objective: "minimise&nbsp; w<sub>1</sub><sup>T</sup>R&thinsp;w<sub>1</sub>",
      subject: "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = 1",
      solution:
        "w<sub>1</sub> = R<sup>&minus;1</sup>g<sub>1</sub> &frasl; (g<sub>1</sub><sup>T</sup>R<sup>&minus;1</sup>g<sub>1</sub>)",
      note:
        "One constraint, on one number. Nothing is said about " +
        "w<sub>1</sub><sup>T</sup>g<sub>2</sub>, so the minimiser is free to " +
        "choose whatever value there lowers the output power most.",
    },
    mcmv: {
      objective: "minimise&nbsp; tr(W<sup>T</sup>R&thinsp;W)",
      subject: "W<sup>T</sup>G = I",
      solution:
        "W = R<sup>&minus;1</sup>G&thinsp;(G<sup>T</sup>R<sup>&minus;1</sup>G)<sup>&minus;1</sup>",
      note:
        "The same objective, but the constraint now covers the whole table: " +
        "w<sub>i</sub><sup>T</sup>g<sub>i</sub> = 1 <em>and</em> " +
        "w<sub>i</sub><sup>T</sup>g<sub>j</sub> = 0. The freedom LCMV used to " +
        "cancel with is gone.",
    },
    recipsiicos: {
      objective: "minimise&nbsp; w<sub>1</sub><sup>T</sup>R&#771;&thinsp;w<sub>1</sub>",
      subject: "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = 1",
      solution: "R&#771; = projection of R onto the retained power subspace (rank K)",
      note:
        "The constraint is LCMV's, unchanged. What changes is the covariance: " +
        "the correlation part of R is projected out first, so there is less " +
        "left for the minimiser to cancel against.",
    },
    abmc: {
      objective:
        "minimise&nbsp; &frac12;&thinsp;w<sub>1</sub><sup>T</sup>R&thinsp;w<sub>1</sub>" +
        "&nbsp; while maximising&nbsp; (w<sub>1</sub><sup>T</sup>X)&middot;u",
      subject: "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = 1,&nbsp; &beta;<sub>2</sub> = P&thinsp;&beta;<sub>1</sub>",
      solution: "localiser: |corr(w<sub>1</sub><sup>T</sup>X, u)| over the grid",
      note:
        "A second term rewards output that looks like a known waveform u, and " +
        "P sets how much that is worth relative to staying distortionless.",
    },
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

  var VIEWS = {
    Left: [Math.PI / 2, 0],
    Right: [-Math.PI / 2, 0],
    Top: [0, -Math.PI / 2 + 0.001],
    Front: [Math.PI, 0],
  };

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
    var sensorPos = view(P.sensor_pos);
    var topo = view(P.topography);
    var gscale = P.geometry_scale;
    var wscale = P.waveform_scale;
    var nT = P.n_times;
    var nSrc = P.n_sources;
    var nTraces = P.n_sensor_traces;
    var nCh = P.n_channels;
    var sfreq = P.sfreq;

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
      corr: Math.max(0, P.correlations.indexOf(0.95)),
      sep: 1,
      snr: 1,
      az: -0.9,
      el: 0.2,
    };

    root.innerHTML =
      '<div class="cp-controls">' +
      '<div class="cp-control"><label>Method</label><div class="cp-methods" id="cp-methods"></div></div>' +
      '<div class="cp-control"><label>Source correlation</label>' +
      '<input type="range" id="cp-corr" min="0" max="' + (P.correlations.length - 1) +
      '" step="1"><div class="cp-value" id="cp-corr-v"></div></div>' +
      '<div class="cp-control"><label>Separation</label>' +
      '<input type="range" id="cp-sep" min="0" max="' + (P.separations.length - 1) +
      '" step="1"><div class="cp-value" id="cp-sep-v"></div></div>' +
      '<div class="cp-control"><label>Sensor SNR</label>' +
      '<input type="range" id="cp-snr" min="0" max="' + (P.snrs.length - 1) +
      '" step="1"><div class="cp-value" id="cp-snr-v"></div></div>' +
      "</div>" +

      '<div class="cp-card cp-eq" id="cp-equations"></div>' +

      '<div class="cp-grid">' +
      '<div class="cp-card cp-wide"><h4>Where the method says the sources are</h4>' +
      '<p class="cp-hint">Rings are the two simulated sources. Crosses are the two ' +
      "strongest points of the localiser, which is what the error is measured " +
      "against; the dashed line joins each estimate to the source it missed. Drag " +
      "to rotate. Colour is the localiser value by <em>rank</em>, because the four " +
      "methods produce distributions no single scaling can show at once.</p>" +
      '<div class="cp-views" id="cp-views"></div>' +
      '<canvas id="cp-brain"></canvas>' +
      '<div class="cp-legend"><span>low</span><div class="cp-ramp" id="cp-ramp"></div><span>high</span>' +
      '<span class="cp-key"><i class="cp-ring"></i> simulated</span>' +
      '<span class="cp-key"><i class="cp-cross"></i> estimated</span></div></div>' +

      '<div class="cp-card"><h4>The constraint table</h4>' +
      '<p class="cp-hint">Row <em>i</em> is the filter for source <em>i</em>; column ' +
      "<em>j</em> is the gain it has at source <em>j</em>. The diagonal is what the " +
      "distortionless constraint pins to one. The off-diagonal is the part LCMV " +
      "leaves free, and it is where the cancellation lives.</p>" +
      '<table class="cp-table" id="cp-gains"></table>' +
      '<div class="cp-readout" id="cp-readout"></div></div>' +

      '<div class="cp-card cp-wide"><h4>What the sensors record</h4>' +
      '<p class="cp-hint">Left: the field over the array at the instant of strongest ' +
      "signal, marked on the traces at right. Right: twenty gradiometers over " +
      "1.25 s. This is the whole of what any beamformer sees.</p>" +
      '<div class="cp-sensor-row">' +
      '<canvas id="cp-topo"></canvas><canvas id="cp-sensor"></canvas></div></div>' +

      '<div class="cp-card"><h4>True and recovered source waveforms</h4>' +
      '<p class="cp-hint">Both sources on one shared scale, so a reconstruction that ' +
      "lost half its amplitude looks like it did. Grey is simulated, colour is " +
      "recovered.</p>" +
      '<canvas id="cp-wave"></canvas></div>' +

      '<div class="cp-card cp-wide"><h4>How it changes as the sources become correlated</h4>' +
      '<p class="cp-hint">The whole correlation axis at the current separation and ' +
      "SNR, for all four methods at once. The marker is where the sliders are. This " +
      "is the parameter dependence the sliders show one point at a time.</p>" +
      '<canvas id="cp-sweep"></canvas>' +
      '<div class="cp-legend" id="cp-sweep-key"></div></div>' +
      "</div>";

    var ramp = root.querySelector("#cp-ramp");
    ramp.style.background =
      "linear-gradient(to right," +
      RAMP.map(function (s) {
        return "rgb(" + s[1] + "," + s[2] + "," + s[3] + ") " + s[0] * 100 + "%";
      }).join(",") +
      ")";

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

    var viewsBox = root.querySelector("#cp-views");
    Object.keys(VIEWS).forEach(function (name) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = name;
      b.addEventListener("click", function () {
        state.az = VIEWS[name][0];
        state.el = VIEWS[name][1];
        drawBrain();
      });
      viewsBox.appendChild(b);
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
    var dragging = false, lastX = 0, lastY = 0;
    brain.addEventListener("pointerdown", function (e) {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      brain.setPointerCapture(e.pointerId);
    });
    brain.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      state.az += (e.clientX - lastX) * 0.01;
      state.el += (e.clientY - lastY) * 0.01;
      state.el = Math.max(-1.5, Math.min(1.5, state.el));
      lastX = e.clientX; lastY = e.clientY;
      drawBrain();
    });
    brain.addEventListener("pointerup", function (e) {
      dragging = false;
      brain.releasePointerCapture(e.pointerId);
    });

    function current() {
      var s = sceneOf[
        P.separations[state.sep] + "|" + P.correlations[state.corr] + "|" + P.snrs[state.snr]
      ];
      return { scene: s, result: resultOf[s + "|" + P.methods[state.method]] };
    }

    function rotated(x, y, z) {
      var ca = Math.cos(state.az), sa = Math.sin(state.az);
      var X = x * ca - y * sa, Y = x * sa + y * ca;
      var ce = Math.cos(state.el), se = Math.sin(state.el);
      return [X, Y * ce - z * se, Y * se + z * ce];
    }

    function at(arr, i) {
      return rotated(arr[i * 3] / gscale, arr[i * 3 + 1] / gscale, arr[i * 3 + 2] / gscale);
    }

    function drawBrain() {
      var c = fitCanvas(brain, 340);
      var ctx = c.ctx;
      var cur = current();
      var scene = P.scenes[cur.scene];
      var res = P.results[cur.result];
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);

      var pts = [], i, p;
      for (i = 0; i < cortex.length / 3; i++) {
        p = at(cortex, i);
        pts.push([p[0], p[2], p[1], -1, 0]);
      }
      var base = cur.result * nSrc;
      for (i = 0; i < nSrc; i++) {
        p = at(pos, i);
        pts.push([p[0], p[2], p[1], i, maps[base + i] / 255]);
      }

      var xs = pts.map(function (q) { return q[0]; });
      var ys = pts.map(function (q) { return q[1]; });
      var ds = pts.map(function (q) { return q[2]; });
      var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
      var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
      var minD = Math.min.apply(null, ds), maxD = Math.max.apply(null, ds);
      var scale = 0.82 * Math.min(c.w / (maxX - minX || 1), c.h / (maxY - minY || 1));
      var cx = c.w / 2 - ((minX + maxX) / 2) * scale;
      var cy = c.h / 2 + ((minY + maxY) / 2) * scale;
      var X = function (q) { return cx + q[0] * scale; };
      var Y = function (q) { return cy - q[1] * scale; };

      pts.sort(function (a, b) { return a[2] - b[2]; });
      var backdrop = css(root, "--cp-grid");
      for (i = 0; i < pts.length; i++) {
        var q = pts[i];
        /* Depth fade, so the backdrop reads as a solid head rather than a
         * uniform fog with the near and far surfaces indistinguishable. */
        var depth = (q[2] - minD) / (maxD - minD || 1);
        if (q[3] < 0) {
          ctx.fillStyle = backdrop;
          ctx.globalAlpha = 0.1 + 0.3 * depth;
          ctx.fillRect(X(q), Y(q), 1.5, 1.5);
        } else {
          var rgb = rampColour(q[4]);
          ctx.globalAlpha = (0.25 + 0.75 * q[4]) * (0.45 + 0.55 * depth);
          ctx.fillStyle = "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
          ctx.beginPath();
          ctx.arc(X(q), Y(q), 1.5 + 4.5 * q[4] * q[4] * q[4], 0, 6.2832);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;

      /* Estimates and truth, and the gap between them. Rings and crosses rather
       * than filled markers, so neither can hide the peak it points at. */
      var truth = scene.sources.map(function (s) { return at(pos, s); });
      var est = res.peaks.map(function (s) { return at(pos, s); });
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = css(root, "--cp-muted");
      ctx.lineWidth = 1;
      truth.forEach(function (t) {
        var best = null, bestD = Infinity;
        est.forEach(function (e) {
          var d = (X(e) - X(t)) * (X(e) - X(t)) + (Y(e) - Y(t)) * (Y(e) - Y(t));
          if (d < bestD) { bestD = d; best = e; }
        });
        if (best && bestD > 25) {
          ctx.beginPath();
          ctx.moveTo(X(t), Y(t));
          ctx.lineTo(X(best), Y(best));
          ctx.stroke();
        }
      });
      ctx.setLineDash([]);

      ctx.lineWidth = 2;
      ctx.strokeStyle = css(root, "--cp-warm");
      truth.forEach(function (t) {
        ctx.beginPath();
        ctx.arc(X(t), Y(t), 9, 0, 6.2832);
        ctx.stroke();
      });
      ctx.strokeStyle = css(root, "--cp-text");
      est.forEach(function (e) {
        ctx.beginPath();
        ctx.moveTo(X(e) - 6, Y(e) - 6); ctx.lineTo(X(e) + 6, Y(e) + 6);
        ctx.moveTo(X(e) + 6, Y(e) - 6); ctx.lineTo(X(e) - 6, Y(e) + 6);
        ctx.stroke();
      });

      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(
        "localisation error " + res.peak_errors.map(function (v) {
          return v.toFixed(0);
        }).join(" and ") + " mm",
        10, c.h - 10
      );
    }

    var topoCache = {};
    function drawTopo() {
      var cur = current();
      var canvas = root.querySelector("#cp-topo");
      var c = fitCanvas(canvas, 200);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var R = Math.min(c.w, c.h) / 2 - 12;
      var ox = c.w / 2, oy = c.h / 2;
      var key = cur.scene + "|" + Math.round(c.w) + "|" + Math.round(c.h);

      if (!topoCache[key]) {
        /* Inverse-distance interpolation over the sensor positions. Cheap
         * enough at 203 channels, and cached because the field depends on the
         * scene, not on which method is selected. */
        var img = ctx.createImageData(Math.round(c.w), Math.round(c.h));
        var vals = [];
        for (var k = 0; k < nCh; k++) vals.push(topo[cur.scene * nCh + k] / wscale);
        for (var py = 0; py < img.height; py++) {
          for (var px = 0; px < img.width; px++) {
            var dx = (px - ox) / R, dy = (py - oy) / R;
            var idx = (py * img.width + px) * 4;
            if (dx * dx + dy * dy > 1) continue;
            var num = 0, den = 0;
            for (var m = 0; m < nCh; m++) {
              var ex = dx - sensorPos[m * 2] / wscale;
              var ey = dy + sensorPos[m * 2 + 1] / wscale;
              var w = 1 / (ex * ex + ey * ey + 0.002);
              num += w * vals[m];
              den += w;
            }
            var v = Math.max(-1, Math.min(1, num / den));
            /* Diverging: blue for one polarity, red for the other, pale at
             * zero. A sequential map here would imply a sign that is not
             * there. */
            var t = (v + 1) / 2;
            img.data[idx] = Math.round(60 + 195 * t);
            img.data[idx + 1] = Math.round(90 + 130 * (1 - Math.abs(v)));
            img.data[idx + 2] = Math.round(255 - 195 * t);
            img.data[idx + 3] = 235;
          }
        }
        topoCache[key] = img;
      }
      ctx.putImageData(topoCache[key], 0, 0);

      ctx.strokeStyle = css(root, "--cp-border");
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(ox, oy, R, 0, 6.2832);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ox - 7, oy - R); ctx.lineTo(ox, oy - R - 9); ctx.lineTo(ox + 7, oy - R);
      ctx.stroke();
      ctx.fillStyle = css(root, "--cp-text");
      ctx.globalAlpha = 0.55;
      for (var s2 = 0; s2 < nCh; s2++) {
        ctx.beginPath();
        ctx.arc(
          ox + (sensorPos[s2 * 2] / wscale) * R,
          oy - (sensorPos[s2 * 2 + 1] / wscale) * R,
          1.1, 0, 6.2832
        );
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText("field at t = " +
        (P.topography_time[cur.scene] / sfreq * 1000).toFixed(0) + " ms", 6, c.h - 6);
    }

    function drawSensor() {
      var cur = current();
      var canvas = root.querySelector("#cp-sensor");
      var c = fitCanvas(canvas, 200);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var padL = 6, padR = 6, padT = 6, padB = 18;
      var lane = (c.h - padT - padB) / nTraces;
      var off = cur.scene * nTraces * nT;
      ctx.strokeStyle = css(root, "--cp-muted");
      ctx.lineWidth = 0.9;
      ctx.globalAlpha = 0.75;
      for (var k = 0; k < nTraces; k++) {
        ctx.beginPath();
        for (var t = 0; t < nT; t++) {
          var v = sensor[off + k * nT + t] / wscale;
          var x = padL + (t / (nT - 1)) * (c.w - padL - padR);
          var y = padT + lane * (k + 0.5) - v * lane * 1.6;
          if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      var tx = padL + (P.topography_time[cur.scene] / (nT - 1)) * (c.w - padL - padR);
      ctx.strokeStyle = css(root, "--cp-accent");
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(tx, padT); ctx.lineTo(tx, c.h - padB);
      ctx.stroke();

      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText("0", padL, c.h - 5);
      ctx.fillText((nT / sfreq * 1000).toFixed(0) + " ms", c.w - 44, c.h - 5);
      ctx.fillText("20 gradiometers", padL + 40, c.h - 5);
    }

    function drawWave() {
      var cur = current();
      var canvas = root.querySelector("#cp-wave");
      var c = fitCanvas(canvas, 200);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var pad = 8;
      var lane = (c.h - 2 * pad) / 2;
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
            var x = pad + (t / (nT - 1)) * (c.w - 2 * pad);
            var y = pad + lane * (k + 0.5) - v * lane * 0.4;
            if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
        });
      }
      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText("source 1", 8, pad + 12);
      ctx.fillText("source 2", 8, pad + lane + 12);
    }

    var SWEEP_COLOURS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7"];
    function drawSweep() {
      var canvas = root.querySelector("#cp-sweep");
      var c = fitCanvas(canvas, 230);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var padL = 44, padR = 10, padT = 16, padB = 26;
      var half = (c.h - padT - padB - 18) / 2;
      var n = P.correlations.length;
      var xOf = function (i) { return padL + (i / (n - 1)) * (c.w - padL - padR); };

      var series = [];
      var maxErr = 1;
      P.methods.forEach(function (m, mi) {
        var amp = [], err = [];
        for (var i = 0; i < n; i++) {
          var s = sceneOf[P.separations[state.sep] + "|" + P.correlations[i] + "|" + P.snrs[state.snr]];
          var r = P.results[resultOf[s + "|" + m]];
          amp.push(r.amplitude_ratio[0]);
          err.push(r.peak_errors[0]);
          if (r.peak_errors[0] > maxErr) maxErr = r.peak_errors[0];
        }
        series.push({ m: m, mi: mi, amp: amp, err: err });
      });

      function panel(top, height, key, maxV, label) {
        ctx.strokeStyle = css(root, "--cp-border");
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, top); ctx.lineTo(padL, top + height); ctx.lineTo(c.w - padR, top + height);
        ctx.stroke();
        ctx.fillStyle = css(root, "--cp-muted");
        ctx.font = "10px system-ui, sans-serif";
        ctx.fillText(maxV.toFixed(maxV > 3 ? 0 : 1), 6, top + 8);
        ctx.fillText("0", 6, top + height);
        ctx.save();
        ctx.translate(11, top + height / 2 + 24);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(label, 0, 0);
        ctx.restore();
        series.forEach(function (s) {
          var on = s.mi === state.method;
          ctx.strokeStyle = SWEEP_COLOURS[s.mi];
          ctx.globalAlpha = on ? 1 : 0.32;
          ctx.lineWidth = on ? 2.2 : 1.2;
          ctx.beginPath();
          for (var i = 0; i < n; i++) {
            var y = top + height - Math.min(1, s[key][i] / maxV) * height;
            if (i === 0) ctx.moveTo(xOf(i), y); else ctx.lineTo(xOf(i), y);
          }
          ctx.stroke();
          if (on) {
            var yc = top + height - Math.min(1, s[key][state.corr] / maxV) * height;
            ctx.fillStyle = SWEEP_COLOURS[s.mi];
            ctx.beginPath();
            ctx.arc(xOf(state.corr), yc, 4.5, 0, 6.2832);
            ctx.fill();
          }
          ctx.globalAlpha = 1;
        });
      }

      panel(padT, half, "amp", 1.2, "amplitude");
      panel(padT + half + 18, half, "err", Math.ceil(maxErr / 10) * 10, "error (mm)");

      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "10px system-ui, sans-serif";
      for (var i = 0; i < n; i++) {
        ctx.fillText(P.correlations[i].toFixed(2), xOf(i) - 10, c.h - 8);
      }
      ctx.fillText("source correlation", c.w / 2 - 40, c.h - 22 + 20);

      root.querySelector("#cp-sweep-key").innerHTML = P.methods
        .map(function (m, i) {
          return (
            '<span class="cp-key"><i style="background:' + SWEEP_COLOURS[i] +
            ';width:14px;height:3px;border-radius:2px"></i>' +
            (METHOD_LABEL[m] || m) + "</span>"
          );
        })
        .join("");
    }

    function drawTable() {
      var cur = current();
      var res = P.results[cur.result];
      var g = res.gains;
      var html = "<tr><th></th><th>at source 1</th><th>at source 2</th></tr>";
      for (var i = 0; i < 2; i++) {
        html += "<tr><th>filter " + (i + 1) + "</th>";
        for (var j = 0; j < 2; j++) {
          var v = g[i][j];
          var mag = Math.min(1, Math.abs(v));
          var col = v >= 0
            ? "rgba(200,69,46," + (0.12 + 0.55 * mag) + ")"
            : "rgba(33,102,172," + (0.12 + 0.55 * mag) + ")";
          html += '<td style="background:' + col + '">' +
            (v >= 0 ? "+" : "") + v.toFixed(3) + "</td>";
        }
        html += "</tr>";
      }
      root.querySelector("#cp-gains").innerHTML = html;

      var off = Math.abs(g[0][1]);
      root.querySelector("#cp-readout").innerHTML =
        '<div><span>Off-diagonal gain</span><strong>' +
        (g[0][1] >= 0 ? "+" : "") + g[0][1].toFixed(3) + "</strong></div>" +
        '<div><span>Amplitude recovered</span><strong>' +
        (res.amplitude_ratio[0] * 100).toFixed(0) + "%</strong></div>" +
        '<div><span>Localisation error</span><strong>' +
        res.peak_errors[0].toFixed(0) + " mm</strong></div>" +
        '<div><span>Cancelling?</span><strong>' +
        (off > 0.2 ? "yes" : "no") + "</strong></div>";
    }

    function drawEquations() {
      var cur = current();
      var res = P.results[cur.result];
      var eq = EQUATIONS[P.methods[state.method]];
      var g = res.gains;
      root.querySelector("#cp-equations").innerHTML =
        "<h4>What " + (METHOD_LABEL[P.methods[state.method]]) + " is solving</h4>" +
        '<div class="cp-eq-row"><span class="cp-eq-main">' + eq.objective +
        '</span><span class="cp-eq-sub">subject to</span><span class="cp-eq-main">' +
        eq.subject + "</span></div>" +
        '<div class="cp-eq-sol">' + eq.solution + "</div>" +
        '<p class="cp-hint">' + eq.note + "</p>" +
        '<div class="cp-eq-live">with the sliders where they are: ' +
        "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = <b>" + g[0][0].toFixed(3) +
        "</b>, &nbsp;w<sub>1</sub><sup>T</sup>g<sub>2</sub> = <b>" +
        (g[0][1] >= 0 ? "+" : "") + g[0][1].toFixed(3) +
        "</b> &nbsp;&rarr;&nbsp; " +
        (res.amplitude_ratio[0] * 100).toFixed(0) + "% of the amplitude survives, " +
        "peak " + res.peak_errors[0].toFixed(0) + " mm from the source</div>";
    }

    function render() {
      P.methods.forEach(function (m, i) {
        methodsBox.children[i].setAttribute(
          "aria-pressed", i === state.method ? "true" : "false"
        );
      });
      var scene = P.scenes[current().scene];
      root.querySelector("#cp-corr-v").textContent = "r = " + scene.correlation.toFixed(2);
      root.querySelector("#cp-sep-v").textContent =
        (scene.separation * 100).toFixed(1) + " cm apart";
      root.querySelector("#cp-snr-v").textContent = "SNR " + P.snrs[state.snr];
      drawEquations();
      drawBrain();
      drawTopo();
      drawSensor();
      drawWave();
      drawTable();
      drawSweep();
    }

    var pending;
    window.addEventListener("resize", function () {
      clearTimeout(pending);
      topoCache = {};
      pending = setTimeout(render, 120);
    });
    new MutationObserver(function () {
      topoCache = {};
      render();
    }).observe(document.documentElement, {
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
      root.innerHTML = '<div class="cp-status">The panel data file did not load.</div>';
      return;
    }
    root.innerHTML = '<div class="cp-status">Loading the precomputed scenes…</div>';
    decode(P.blob).then(
      function (buf) {
        try {
          build(root, P, buf);
        } catch (err) {
          root.innerHTML =
            '<div class="cp-status">The panel failed to start: ' + err.message + "</div>";
        }
      },
      function (err) {
        root.innerHTML =
          '<div class="cp-status">The panel could not decode its data (' + err.message +
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
