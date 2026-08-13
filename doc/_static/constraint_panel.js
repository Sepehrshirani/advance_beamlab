/* Interactive constraint panel.
 *
 * Every configuration was computed by tools/build_constraint_panel.py, which
 * calls advance_beamlab.constraint_demo. Nothing is simulated here; the page
 * only displays what that produced. See doc/panel.rst.
 *
 * The sensor recording is the one exception, and it is a reconstruction rather
 * than a simulation: the build stores the leadfield columns, the source
 * waveforms and one shared noise field, and this file recombines them exactly
 * as the build did. Storing 203 channels for every scene would have been
 * megabytes of noise, which does not compress.
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
  var MORPH_LABEL = { oscillation: "10 Hz rhythm", transient: "bursts" };

  var EQUATIONS = {
    lcmv: {
      objective: "minimise&nbsp; w<sub>1</sub><sup>T</sup>R&thinsp;w<sub>1</sub>",
      subject: "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = 1",
      solution:
        "w<sub>1</sub> = R<sup>&minus;1</sup>g<sub>1</sub> &frasl; " +
        "(g<sub>1</sub><sup>T</sup>R<sup>&minus;1</sup>g<sub>1</sub>)",
      note:
        "One constraint, on one number. Nothing is said about " +
        "w<sub>1</sub><sup>T</sup>g<sub>j</sub> for any other source, so the " +
        "minimiser is free to choose whatever value there lowers output power " +
        "most.",
    },
    mcmv: {
      objective: "minimise&nbsp; tr(W<sup>T</sup>R&thinsp;W)",
      subject: "W<sup>T</sup>G = I",
      solution:
        "W = R<sup>&minus;1</sup>G&thinsp;(G<sup>T</sup>R<sup>&minus;1</sup>G)" +
        "<sup>&minus;1</sup>",
      note:
        "The same objective with the constraint extended over the whole table: " +
        "w<sub>i</sub><sup>T</sup>g<sub>i</sub> = 1 <em>and</em> " +
        "w<sub>i</sub><sup>T</sup>g<sub>j</sub> = 0. The freedom LCMV used to " +
        "cancel with is gone. Note it needs to be told where the sources are.",
    },
    recipsiicos: {
      objective:
        "minimise&nbsp; w<sub>1</sub><sup>T</sup>R&#771;&thinsp;w<sub>1</sub>",
      subject: "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = 1",
      solution: "R&#771; = projection of R onto the retained power subspace",
      note:
        "The constraint is LCMV's, unchanged. What changes is the covariance: " +
        "the correlation part of R is projected out first, so there is less " +
        "left for the minimiser to cancel against.",
    },
    abmc: {
      objective:
        "minimise&nbsp; &frac12;&thinsp;w<sub>1</sub><sup>T</sup>R&thinsp;" +
        "w<sub>1</sub>&nbsp; while maximising&nbsp; " +
        "(w<sub>1</sub><sup>T</sup>X)&middot;u",
      subject:
        "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = 1,&nbsp; " +
        "&beta;<sub>2</sub> = P&thinsp;&beta;<sub>1</sub>",
      solution:
        "localiser: |corr(w<sub>1</sub><sup>T</sup>X, u)| over the grid",
      note:
        "A second term rewards output that looks like a known waveform u, and " +
        "P sets what that is worth against staying distortionless. Try it with " +
        "the burst morphology, which is the regime it was designed for.",
    },
  };

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
        var a = RAMP[i - 1], b = RAMP[i];
        var f = (t - a[0]) / (b[0] - a[0] || 1);
        return [
          Math.round(a[1] + f * (b[1] - a[1])),
          Math.round(a[2] + f * (b[2] - a[2])),
          Math.round(a[3] + f * (b[3] - a[3])),
        ];
      }
    }
    return [255, 240, 170];
  }

  /* Diverging, through near-white, because the field has a sign and a
   * sequential map would invent a polarity that is not there. */
  function fieldColour(v) {
    v = Math.max(-1, Math.min(1, v));
    var lo = [33, 102, 172], mid = [247, 247, 247], hi = [178, 24, 43];
    var a = v < 0 ? lo : hi, f = Math.abs(v);
    return [
      Math.round(mid[0] + f * (a[0] - mid[0])),
      Math.round(mid[1] + f * (a[1] - mid[1])),
      Math.round(mid[2] + f * (a[2] - mid[2])),
    ];
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
    return new Response(
      new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))
    ).arrayBuffer();
  }

  function build(root, P, buf) {
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
    var noise = view(P.noise);
    var mix = view(P.mix);
    var sensorPos = view(P.sensor_pos);
    var topo = view(P.topography);
    var gscale = P.geometry_scale;
    var wscale = P.waveform_scale;
    var nT = P.n_times;
    var nSrc = P.n_sources;
    var nCh = P.n_channels;
    var sfreq = P.sfreq;

    /* Variable-length blocks: scenes hold one, two or three sources, so the
     * offsets are cumulative rather than a fixed stride. */
    var sceneOffset = [], mixOffset = [], acc = 0, macc = 0;
    P.scenes.forEach(function (s) {
      sceneOffset.push(acc);
      mixOffset.push(macc);
      acc += s.sources.length * nT;
      macc += nCh * s.sources.length;
    });
    var resultOffset = [], racc = 0;
    P.results.forEach(function (r) {
      resultOffset.push(racc);
      racc += P.scenes[r.scene].sources.length * nT;
    });

    var sceneOf = {};
    P.scenes.forEach(function (s, i) {
      var q = s.requested;
      sceneOf[[q.sep, q.corr, q.snr, q.n, q.morph].join("|")] = i;
    });
    var resultOf = {};
    P.results.forEach(function (r, i) {
      resultOf[r.scene + "|" + r.method] = i;
    });

    var state = {
      method: 0,
      corr: Math.max(0, P.correlations.indexOf(0.95)),
      sep: P.separations.length - 1,
      snr: P.snrs.length - 1,
      count: Math.max(0, P.source_counts.indexOf(2)),
      morph: 0,
      az: -0.9,
      el: 0.2,
      chan0: 0,
      chanShown: 20,
      tStart: 0,
      tSpan: nT,
    };

    function chip(id, label) {
      return (
        '<div class="cp-control"><label>' + label + "</label>" +
        '<div class="cp-methods" id="' + id + '"></div></div>'
      );
    }
    function slider(id, label, n) {
      return (
        '<div class="cp-control"><label>' + label + "</label>" +
        '<input type="range" id="' + id + '" min="0" max="' + (n - 1) +
        '" step="1"><div class="cp-value" id="' + id + '-v"></div></div>'
      );
    }

    root.innerHTML =
      '<div class="cp-controls">' +
      chip("cp-methods", "Method") +
      chip("cp-count", "Sources") +
      chip("cp-morph", "Morphology") +
      slider("cp-corr", "Source correlation", P.correlations.length) +
      slider("cp-sep", "Separation", P.separations.length) +
      slider("cp-snr", "Sensor SNR", P.snrs.length) +
      "</div>" +

      '<div class="cp-card cp-eq" id="cp-equations"></div>' +

      '<div class="cp-grid">' +
      '<div class="cp-card cp-half"><h4>Where the method says the sources are</h4>' +
      '<p class="cp-hint">Rings are the simulated sources, crosses the strongest ' +
      "points of the localiser, dashed lines the miss. Colour is the map " +
      "normalised to its own range and cube-root compressed, which keeps the " +
      "four methods distinguishable. Drag to rotate.</p>" +
      '<div class="cp-views" id="cp-views"></div>' +
      '<canvas id="cp-brain"></canvas>' +
      '<div class="cp-legend"><span>low</span><div class="cp-ramp" id="cp-ramp"></div>' +
      '<span>high</span><span class="cp-key"><i class="cp-ring"></i> simulated</span>' +
      '<span class="cp-key"><i class="cp-cross"></i> estimated</span></div></div>' +

      '<div class="cp-card cp-half"><h4>The constraint table</h4>' +
      '<p class="cp-hint">Row <em>i</em> is the filter for source <em>i</em>, column ' +
      "<em>j</em> its gain at source <em>j</em>. The diagonal is what the " +
      "distortionless constraint pins. The off-diagonal is what LCMV leaves free, " +
      "and where the cancellation lives.</p>" +
      '<table class="cp-table" id="cp-gains"></table>' +
      '<div class="cp-readout" id="cp-readout"></div></div>' +

      '<div class="cp-card cp-wide"><h4>What the sensors record</h4>' +
      '<p class="cp-hint">All ' + nCh + " gradiometers. Drag the traces to scroll " +
      "through channels and time, or use the sliders. The map is the field at the " +
      "instant marked on the traces.</p>" +
      '<div class="cp-sensor-row">' +
      '<div><canvas id="cp-topo"></canvas></div>' +
      '<div><canvas id="cp-sensor"></canvas>' +
      '<div class="cp-scrollers">' +
      '<label>channels <input type="range" id="cp-chan" min="0" max="' +
      (nCh - 1) + '" step="1" value="0"></label>' +
      '<label>zoom <input type="range" id="cp-zoom" min="0" max="100" step="1" ' +
      'value="0"></label>' +
      '<span class="cp-value" id="cp-chan-v"></span></div></div></div></div>' +

      '<div class="cp-card cp-half"><h4>True and recovered source waveforms</h4>' +
      '<p class="cp-hint">One shared scale per scene, so a reconstruction that lost ' +
      "half its amplitude looks like it did. Grey is simulated, colour recovered.</p>" +
      '<canvas id="cp-wave"></canvas></div>' +

      '<div class="cp-card cp-half"><h4>How it changes with correlation</h4>' +
      '<p class="cp-hint">The whole correlation axis at the current settings, for ' +
      "all four methods. The marker is where the sliders are.</p>" +
      '<canvas id="cp-sweep"></canvas>' +
      '<div class="cp-legend" id="cp-sweep-key"></div></div>' +
      "</div>";

    root.querySelector("#cp-ramp").style.background =
      "linear-gradient(to right," +
      RAMP.map(function (s) {
        return "rgb(" + s[1] + "," + s[2] + "," + s[3] + ") " + s[0] * 100 + "%";
      }).join(",") + ")";

    function chips(id, items, key, label) {
      var box = root.querySelector("#" + id);
      items.forEach(function (m, i) {
        var b = document.createElement("button");
        b.type = "button";
        b.textContent = label ? label(m) : m;
        b.addEventListener("click", function () {
          state[key] = i;
          render();
        });
        box.appendChild(b);
      });
      return box;
    }
    var methodsBox = chips("cp-methods", P.methods, "method", function (m) {
      return METHOD_LABEL[m] || m;
    });
    var countBox = chips("cp-count", P.source_counts, "count", String);
    var morphBox = chips("cp-morph", P.morphologies, "morph", function (m) {
      return MORPH_LABEL[m] || m;
    });

    ["corr", "sep", "snr"].forEach(function (key) {
      var input = root.querySelector("#cp-" + key);
      input.value = state[key];
      input.addEventListener("input", function () {
        state[key] = parseInt(input.value, 10);
        render();
      });
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

    function drag(canvas, onMove) {
      var on = false, lx = 0, ly = 0;
      canvas.addEventListener("pointerdown", function (e) {
        on = true; lx = e.clientX; ly = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener("pointermove", function (e) {
        if (!on) return;
        onMove(e.clientX - lx, e.clientY - ly);
        lx = e.clientX; ly = e.clientY;
      });
      canvas.addEventListener("pointerup", function (e) {
        on = false;
        canvas.releasePointerCapture(e.pointerId);
      });
    }

    var brain = root.querySelector("#cp-brain");
    drag(brain, function (dx, dy) {
      state.az += dx * 0.01;
      state.el = Math.max(-1.5, Math.min(1.5, state.el + dy * 0.01));
      drawBrain();
    });

    var sensorCanvas = root.querySelector("#cp-sensor");
    drag(sensorCanvas, function (dx, dy) {
      state.chan0 = Math.max(
        0, Math.min(nCh - state.chanShown, state.chan0 - Math.round(dy / 8))
      );
      state.tStart = Math.max(
        0, Math.min(nT - state.tSpan, state.tStart - Math.round(dx * state.tSpan / 400))
      );
      root.querySelector("#cp-chan").value = state.chan0;
      drawSensor();
    });
    sensorCanvas.addEventListener("wheel", function (e) {
      e.preventDefault();
      var centre = state.tStart + state.tSpan / 2;
      state.tSpan = Math.max(25, Math.min(nT, Math.round(state.tSpan * (e.deltaY > 0 ? 1.15 : 0.87))));
      state.tStart = Math.max(0, Math.min(nT - state.tSpan, Math.round(centre - state.tSpan / 2)));
      root.querySelector("#cp-zoom").value = Math.round(
        100 * (1 - (state.tSpan - 25) / (nT - 25))
      );
      drawSensor();
    }, { passive: false });

    root.querySelector("#cp-chan").addEventListener("input", function (e) {
      state.chan0 = Math.min(nCh - state.chanShown, parseInt(e.target.value, 10));
      drawSensor();
    });
    root.querySelector("#cp-zoom").addEventListener("input", function (e) {
      var f = parseInt(e.target.value, 10) / 100;
      var centre = state.tStart + state.tSpan / 2;
      state.tSpan = Math.round(nT - f * (nT - 25));
      state.tStart = Math.max(0, Math.min(nT - state.tSpan, Math.round(centre - state.tSpan / 2)));
      drawSensor();
    });

    function current() {
      var key = [
        P.separations[state.sep],
        P.correlations[state.corr],
        P.snrs[state.snr],
        P.source_counts[state.count],
        P.morphologies[state.morph],
      ].join("|");
      var s = sceneOf[key];
      return { scene: s, result: resultOf[s + "|" + P.methods[state.method]] };
    }

    function nOf(scene) {
      return P.scenes[scene].sources.length;
    }

    /* The recording, rebuilt exactly as the build made it. */
    function sensorAt(scene, channel, t) {
      var n = nOf(scene);
      var v = 0;
      for (var i = 0; i < n; i++) {
        v += (mix[mixOffset[scene] + channel * n + i] / wscale) *
          (trueTcs[sceneOffset[scene] + i * nT + t] / wscale);
      }
      return v + P.noise_gain[scene] * (noise[channel * nT + t] / wscale);
    }

    function rotated(x, y, z) {
      var ca = Math.cos(state.az), sa = Math.sin(state.az);
      var X = x * ca - y * sa, Y = x * sa + y * ca;
      var ce = Math.cos(state.el), se = Math.sin(state.el);
      return [X, Y * ce - z * se, Y * se + z * ce];
    }
    function at(arr, i) {
      return rotated(
        arr[i * 3] / gscale, arr[i * 3 + 1] / gscale, arr[i * 3 + 2] / gscale
      );
    }

    function drawBrain() {
      var c = fitCanvas(brain, 300);
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
        var depth = (q[2] - minD) / (maxD - minD || 1);
        if (q[3] < 0) {
          ctx.fillStyle = backdrop;
          ctx.globalAlpha = 0.1 + 0.3 * depth;
          ctx.fillRect(X(q), Y(q), 1.5, 1.5);
        } else {
          var rgb = rampColour(q[4]);
          ctx.globalAlpha = (0.2 + 0.8 * q[4]) * (0.45 + 0.55 * depth);
          ctx.fillStyle = "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
          ctx.beginPath();
          ctx.arc(X(q), Y(q), 1.4 + 4.5 * q[4] * q[4], 0, 6.2832);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;

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
        "error " + res.peak_errors.map(function (v) { return v.toFixed(0); }).join(", ") +
        " mm", 10, c.h - 10
      );
    }

    var topoCache = {};
    function drawTopo() {
      var cur = current();
      var canvas = root.querySelector("#cp-topo");
      var c = fitCanvas(canvas, 230);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var R = Math.min(c.w, c.h) / 2 - 16;
      var ox = c.w / 2, oy = c.h / 2 + 4;
      var key = cur.scene + "|" + Math.round(c.w) + "|" + Math.round(c.h);

      if (!topoCache[key]) {
        var W = Math.round(c.w), H = Math.round(c.h);
        var img = ctx.createImageData(W, H);
        var grid = new Float32Array(W * H);
        var vals = [];
        for (var k = 0; k < nCh; k++) vals.push(topo[cur.scene * nCh + k] / wscale);
        for (var py = 0; py < H; py++) {
          for (var px = 0; px < W; px++) {
            var dx = (px - ox) / R, dy = (py - oy) / R;
            var gi = py * W + px;
            grid[gi] = NaN;
            if (dx * dx + dy * dy > 1) continue;
            var num = 0, den = 0;
            for (var m = 0; m < nCh; m++) {
              var ex = dx - sensorPos[m * 2] / wscale;
              var ey = dy + sensorPos[m * 2 + 1] / wscale;
              var w = 1 / (ex * ex + ey * ey + 0.0015);
              num += w * vals[m];
              den += w;
            }
            var v = Math.max(-1, Math.min(1, num / den));
            grid[gi] = v;
            var rgb = fieldColour(v);
            var idx = gi * 4;
            img.data[idx] = rgb[0];
            img.data[idx + 1] = rgb[1];
            img.data[idx + 2] = rgb[2];
            img.data[idx + 3] = 255;
          }
        }
        topoCache[key] = { img: img, grid: grid, W: W, H: H };
      }
      var cache = topoCache[key];
      ctx.putImageData(cache.img, 0, 0);

      /* Iso-contours. A smooth colour wash alone is hard to read a gradient
       * from; the lines are what make a field map legible. */
      ctx.strokeStyle = "rgba(0,0,0,0.42)";
      ctx.lineWidth = 0.9;
      [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75].forEach(function (level) {
        ctx.beginPath();
        for (var y = 0; y < cache.H - 1; y++) {
          for (var x = 0; x < cache.W - 1; x++) {
            var a = cache.grid[y * cache.W + x];
            var b = cache.grid[y * cache.W + x + 1];
            var d2 = cache.grid[(y + 1) * cache.W + x];
            if (!isNaN(a) && !isNaN(b) && (a - level) * (b - level) < 0) {
              ctx.moveTo(x + (level - a) / (b - a), y);
              ctx.lineTo(x + (level - a) / (b - a), y + 1);
            }
            if (!isNaN(a) && !isNaN(d2) && (a - level) * (d2 - level) < 0) {
              ctx.moveTo(x, y + (level - a) / (d2 - a));
              ctx.lineTo(x + 1, y + (level - a) / (d2 - a));
            }
          }
        }
        ctx.stroke();
      });

      ctx.strokeStyle = css(root, "--cp-text");
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(ox, oy, R, 0, 6.2832);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ox - 8, oy - R + 1);
      ctx.lineTo(ox, oy - R - 11);
      ctx.lineTo(ox + 8, oy - R + 1);
      ctx.stroke();
      ctx.beginPath();
      ctx.ellipse(ox - R, oy, 5, 12, 0, 0, 6.2832);
      ctx.moveTo(ox + R + 5, oy);
      ctx.ellipse(ox + R, oy, 5, 12, 0, 0, 6.2832);
      ctx.stroke();

      ctx.fillStyle = "rgba(0,0,0,0.55)";
      for (var s2 = 0; s2 < nCh; s2++) {
        ctx.beginPath();
        ctx.arc(
          ox + (sensorPos[s2 * 2] / wscale) * R,
          oy - (sensorPos[s2 * 2 + 1] / wscale) * R,
          1.2, 0, 6.2832
        );
        ctx.fill();
      }
      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(
        "field at " + ((P.topography_time[cur.scene] / sfreq) * 1000).toFixed(0) + " ms",
        4, c.h - 4
      );
    }

    function drawSensor() {
      var cur = current();
      var c = fitCanvas(sensorCanvas, 230);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var padL = 34, padR = 6, padT = 6, padB = 18;
      var shown = Math.min(state.chanShown, nCh - state.chan0);
      var lane = (c.h - padT - padB) / shown;
      var t0 = state.tStart, t1 = Math.min(nT, state.tStart + state.tSpan);

      // Common scale across the visible channels, so relative amplitude is real.
      var peak = 1e-9;
      for (var k = 0; k < shown; k++) {
        for (var t = t0; t < t1; t++) {
          var a = Math.abs(sensorAt(cur.scene, state.chan0 + k, t));
          if (a > peak) peak = a;
        }
      }
      ctx.strokeStyle = css(root, "--cp-muted");
      ctx.lineWidth = 0.9;
      ctx.globalAlpha = 0.8;
      for (k = 0; k < shown; k++) {
        ctx.beginPath();
        for (t = t0; t < t1; t++) {
          var v = sensorAt(cur.scene, state.chan0 + k, t) / peak;
          var x = padL + ((t - t0) / (t1 - t0 - 1 || 1)) * (c.w - padL - padR);
          var y = padT + lane * (k + 0.5) - v * lane * 0.48;
          if (t === t0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "9px system-ui, sans-serif";
      for (k = 0; k < shown; k += Math.ceil(shown / 8)) {
        ctx.fillText("ch " + (state.chan0 + k + 1), 2, padT + lane * (k + 0.5) + 3);
      }

      var tp = P.topography_time[cur.scene];
      if (tp >= t0 && tp < t1) {
        var tx = padL + ((tp - t0) / (t1 - t0 - 1 || 1)) * (c.w - padL - padR);
        ctx.strokeStyle = css(root, "--cp-accent");
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(tx, padT); ctx.lineTo(tx, c.h - padB);
        ctx.stroke();
      }
      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(((t0 / sfreq) * 1000).toFixed(0) + " ms", padL, c.h - 5);
      ctx.fillText(((t1 / sfreq) * 1000).toFixed(0) + " ms", c.w - 46, c.h - 5);
      root.querySelector("#cp-chan-v").textContent =
        "channels " + (state.chan0 + 1) + "–" + (state.chan0 + shown) +
        " of " + nCh;
    }

    var PALETTE = ["#d55e00", "#0072b2", "#009e73"];
    function drawWave() {
      var cur = current();
      var n = nOf(cur.scene);
      var c = fitCanvas(root.querySelector("#cp-wave"), 230);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var pad = 8;
      var lane = (c.h - 2 * pad) / n;
      for (var k = 0; k < n; k++) {
        [
          [trueTcs, sceneOffset[cur.scene], css(root, "--cp-true"), 1.1],
          [recon, resultOffset[cur.result], PALETTE[k % PALETTE.length], 1.6],
        ].forEach(function (spec) {
          ctx.beginPath();
          ctx.strokeStyle = spec[2];
          ctx.lineWidth = spec[3];
          for (var t = 0; t < nT; t++) {
            var v = spec[0][spec[1] + k * nT + t] / wscale;
            var x = pad + (t / (nT - 1)) * (c.w - 2 * pad);
            var y = pad + lane * (k + 0.5) - v * lane * 0.42;
            if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
        });
        ctx.fillStyle = css(root, "--cp-muted");
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText("source " + (k + 1), 8, pad + lane * k + 12);
      }
    }

    var SWEEP_COLOURS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7"];
    function drawSweep() {
      var c = fitCanvas(root.querySelector("#cp-sweep"), 230);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var padL = 42, padR = 8, padT = 14, padB = 26;
      var half = (c.h - padT - padB - 16) / 2;
      var n = P.correlations.length;
      var xOf = function (i) { return padL + (i / (n - 1)) * (c.w - padL - padR); };
      var series = [], maxErr = 1;
      P.methods.forEach(function (m, mi) {
        var amp = [], err = [];
        for (var i = 0; i < n; i++) {
          var key = [
            P.separations[state.sep], P.correlations[i], P.snrs[state.snr],
            P.source_counts[state.count], P.morphologies[state.morph],
          ].join("|");
          var r = P.results[resultOf[sceneOf[key] + "|" + m]];
          amp.push(r.amplitude_ratio[0]);
          err.push(r.peak_errors[0]);
          if (r.peak_errors[0] > maxErr) maxErr = r.peak_errors[0];
        }
        series.push({ mi: mi, amp: amp, err: err });
      });

      function panel(top, height, key, maxV, label) {
        ctx.strokeStyle = css(root, "--cp-border");
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, top); ctx.lineTo(padL, top + height);
        ctx.lineTo(c.w - padR, top + height);
        ctx.stroke();
        ctx.fillStyle = css(root, "--cp-muted");
        ctx.font = "10px system-ui, sans-serif";
        ctx.fillText(maxV.toFixed(maxV > 3 ? 0 : 1), 4, top + 8);
        ctx.fillText("0", 4, top + height);
        ctx.save();
        ctx.translate(11, top + height / 2 + 26);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(label, 0, 0);
        ctx.restore();
        series.forEach(function (s) {
          var on = s.mi === state.method;
          ctx.strokeStyle = SWEEP_COLOURS[s.mi];
          ctx.globalAlpha = on ? 1 : 0.3;
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
      panel(padT + half + 16, half, "err", Math.max(10, Math.ceil(maxErr / 10) * 10),
        "error (mm)");
      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "10px system-ui, sans-serif";
      for (var i = 0; i < n; i++) {
        ctx.fillText(P.correlations[i].toFixed(2), xOf(i) - 10, c.h - 8);
      }
      root.querySelector("#cp-sweep-key").innerHTML = P.methods
        .map(function (m, i) {
          return '<span class="cp-key"><i style="background:' + SWEEP_COLOURS[i] +
            ';width:14px;height:3px;border-radius:2px"></i>' +
            (METHOD_LABEL[m] || m) + "</span>";
        }).join("");
    }

    function drawTable() {
      var cur = current();
      var res = P.results[cur.result];
      var g = res.gains, n = g.length;
      var html = "<tr><th></th>";
      for (var j = 0; j < n; j++) html += "<th>at " + (j + 1) + "</th>";
      html += "</tr>";
      for (var i = 0; i < n; i++) {
        html += "<tr><th>filter " + (i + 1) + "</th>";
        for (j = 0; j < n; j++) {
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

      var worst = 0;
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          if (i !== j) worst = Math.max(worst, Math.abs(g[i][j]));
        }
      }
      root.querySelector("#cp-readout").innerHTML =
        '<div><span>Largest off-diagonal</span><strong>' +
        (n > 1 ? worst.toFixed(3) : "n/a") + "</strong></div>" +
        '<div><span>Amplitude recovered</span><strong>' +
        (res.amplitude_ratio[0] * 100).toFixed(0) + "%</strong></div>" +
        '<div><span>Localisation error</span><strong>' +
        res.peak_errors[0].toFixed(0) + " mm</strong></div>" +
        '<div><span>Cancelling?</span><strong>' +
        (n > 1 ? (worst > 0.2 ? "yes" : "no") : "nothing to cancel") +
        "</strong></div>";
    }

    function drawEquations() {
      var cur = current();
      var res = P.results[cur.result];
      var g = res.gains, n = g.length;
      var eq = EQUATIONS[P.methods[state.method]];
      var live = "w<sub>1</sub><sup>T</sup>g<sub>1</sub> = <b>" +
        g[0][0].toFixed(3) + "</b>";
      if (n > 1) {
        live += ", &nbsp;w<sub>1</sub><sup>T</sup>g<sub>2</sub> = <b>" +
          (g[0][1] >= 0 ? "+" : "") + g[0][1].toFixed(3) + "</b>";
      }
      root.querySelector("#cp-equations").innerHTML =
        "<h4>What " + (METHOD_LABEL[P.methods[state.method]]) + " is solving</h4>" +
        '<div class="cp-eq-row"><span class="cp-eq-main">' + eq.objective +
        '</span><span class="cp-eq-sub">subject to</span><span class="cp-eq-main">' +
        eq.subject + "</span></div>" +
        '<div class="cp-eq-sol">' + eq.solution + "</div>" +
        '<p class="cp-hint">' + eq.note + "</p>" +
        '<div class="cp-eq-live">with the controls where they are: ' + live +
        " &nbsp;&rarr;&nbsp; " + (res.amplitude_ratio[0] * 100).toFixed(0) +
        "% of the amplitude survives, peak " + res.peak_errors[0].toFixed(0) +
        " mm from the source</div>";
    }

    function render() {
      [[methodsBox, "method"], [countBox, "count"], [morphBox, "morph"]].forEach(
        function (pair) {
          for (var i = 0; i < pair[0].children.length; i++) {
            pair[0].children[i].setAttribute(
              "aria-pressed", i === state[pair[1]] ? "true" : "false"
            );
          }
        }
      );
      var cur = current();
      var scene = P.scenes[cur.scene];
      var single = nOf(cur.scene) < 2;
      root.querySelector("#cp-corr").disabled = single;
      root.querySelector("#cp-corr-v").textContent =
        single ? "one source, nothing to correlate" : "r = " + scene.correlation.toFixed(2);
      root.querySelector("#cp-sep").disabled = single;
      root.querySelector("#cp-sep-v").textContent =
        single ? "—" : (scene.separation * 100).toFixed(1) + " cm apart";
      root.querySelector("#cp-snr-v").textContent = "SNR " + P.snrs[state.snr];
      state.chan0 = Math.min(state.chan0, nCh - state.chanShown);
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
      attributes: true, attributeFilter: ["data-theme"],
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
