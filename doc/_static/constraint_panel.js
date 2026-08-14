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
  var HEAD_LABEL = { matched: "matched", realistic: "realistic" };
  var HEAD_NOTE = {
    matched:
      "<b>Matched head model.</b> The sources sit exactly on points the " +
      "beamformer scans, and the data are generated with the very leadfield " +
      "being inverted. That is an inverse crime: a single source usually " +
      "localises to exactly 0 mm, because little can move a matched filter off " +
      "its own node. Read the error here as a floor rather than a result. " +
      "What this setting buys is the only clean view of what the " +
      "<em>constraint</em> does, with every other source of error held at zero.",
    realistic:
      "<b>Realistic head model.</b> The sources are taken from a finer forward " +
      "than the one being scanned, so they sit a few millimetres from anything " +
      "any method can report, as on real data. Localisation error becomes " +
      "meaningful. The price is that every method now loses amplitude for the " +
      "same reason, so the constraint contrast largely disappears: a filter " +
      "pointed slightly wrong <em>nulls</em> a source rather than passing a " +
      "weakened copy of it. That is the practical lesson of this control.",
  };
  var MORPH_LABEL = {
    theta: "theta 6 Hz",
    alpha: "alpha 10 Hz",
    beta: "beta 20 Hz",
    transient: "bursts",
  };

  /* One colour system for the whole panel, so a colour means the same thing
   * wherever it appears. A method colour is always the selected method's
   * estimate; grey is always the truth. Sources are told apart by their row and
   * label rather than by colour, which would collide with the method colours. */
  var METHOD_COLOUR = {
    lcmv: "#2a6f97",
    mcmv: "#bc4b23",
    recipsiicos: "#17786b",
    abmc: "#7b4397",
  };

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
        "cancel with is gone. Note it has to be told where the sources are.",
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
      solution: "localiser: |corr(w<sub>1</sub><sup>T</sup>X, u)| over the grid",
      note:
        "A second term rewards output that looks like a known waveform u, and " +
        "P sets what that is worth against staying distortionless. Worth trying " +
        "with the burst morphology, the regime it was designed for.",
    },
  };

  /* Sizes written by the build with {C}-style placeholders, so the numbers
   * come from the model that was actually used rather than from this file. */
  function fillSizes(text, S) {
    return String(text).replace(/\{(\w+)\}/g, function (whole, key) {
      return Object.prototype.hasOwnProperty.call(S, key) ? S[key] : whole;
    });
  }

  /* What each symbol in the equation above actually is, at the current
   * selection. Readers ask this constantly and the algebra never says it.
   *
   * The first table holds only what all four methods share. Everything a
   * method introduces on top of that -- the matrix MCMV inverts, the space
   * ReciPSIICOS projects in, ABMC's template and its iteration counts -- comes
   * from the build in P.model.algorithms, because a table that did not change
   * with the method left those terms with no size at all. */
  function dimensions(P, n, method) {
    var C = P.n_channels, V = P.n_sources;
    var T = (P.model && P.model.n_times_simulated) || P.n_times_simulated;
    var rows = [
      ["X", "sensor recording", C + " &times; " + T, "channels &times; samples"],
      ["R", "data covariance", C + " &times; " + C, "channels &times; channels"],
      ["G", "leadfield, whole scan grid", C + " &times; " + V,
        "channels &times; scanned points"],
      ["g<sub>i</sub>", "leadfield of one point", C + " &times; 1",
        "what one unit dipole at that point would put on every channel. One " +
        "column here because this model is fixed orientation; see below"],
      ["w<sub>i</sub>", "one filter", C + " &times; 1",
        "one weight per channel. Its output at time t is a single number, " +
        "w<sub>i</sub><sup>T</sup>x(t)"],
      ["W", "the filters in play", C + " &times; " + n,
        "one column per <em>constrained</em> source; see below"],
      ["W<sup>T</sup>G<sub>s</sub>", "constraint table", n + " &times; " + n,
        "the table on the right: every filter's gain at every constrained source"],
      ["s&#770;", "reconstructed sources", n + " &times; " + T,
        "one row per source"],
      ["map", "localiser", V + " &times; 1", "one value per scanned point"],
    ];
    function table(list) {
      return (
        '<table class="cp-dims"><tr><th>symbol</th><th>what it is</th>' +
        "<th>size</th><th></th></tr>" +
        list
          .map(function (r) {
            return "<tr><td class=\"cp-sym\">" + r[0] + "</td><td>" + r[1] +
              '</td><td class="cp-size">' + r[2] + "</td><td>" + r[3] +
              "</td></tr>";
          })
          .join("") +
        "</table>"
      );
    }

    var algo = (P.model && P.model.algorithms && P.model.algorithms[method]) || null;
    var extra = "";
    if (algo) {
      var S = {
        C: C, V: V, T: T, n: n, n2: n * n,
        K: (P.model && P.model.recipsiicos_rank) || 0,
        q: (P.model && P.model.recipsiicos_virtual) || 0,
      };
      S.q2 = S.q * S.q;
      extra =
        "<h5>What " + METHOD_LABEL[method] + " adds on top</h5>" +
        '<p class="cp-hint">' + fillSizes(algo.summary, S) + "</p>" +
        table(
          algo.rows.map(function (r) {
            return [r[0], r[1], fillSizes(r[2], S), fillSizes(r[3], S)];
          })
        );
    }

    return (
      table(rows) +
      extra +
      '<p class="cp-note-block"><b>&ldquo;Constrained&rdquo; source.</b> A source ' +
      "the method is <em>told about</em> and writes an equation for. The " +
      "distinction is the whole difference between the first two methods. LCMV " +
      "constrains one at a time: it builds " + n + " separate filters, each " +
      "knowing only its own location, so each carries a single equation and " +
      "the scan grid's other " + (V - n) + " points are simply not mentioned. " +
      "MCMV constrains all " + n + " jointly in one system, so it carries " +
      n * n + " equations and has to be handed the locations in advance. " +
      "Everything outside the constrained set is unconstrained for every method " +
      "here, which is why the localiser is still free to peak in the wrong " +
      "place.</p>" +
      '<p class="cp-note-block"><b>Fixed against free orientation.</b> A dipole ' +
      "has a direction as well as a position. <em>Fixed</em> assumes the " +
      "direction is known, so a point contributes one leadfield column (" + C +
      " &times; 1), its filter is one column, and its output is one number per " +
      "sample. <em>Free</em> leaves the direction to be estimated, so the point " +
      "contributes three orthogonal components (" + C + " &times; 3), the " +
      "filter becomes " + C + " &times; 3, and the output is a three-vector " +
      "that has to be reduced to a scalar, usually by taking the direction of " +
      "maximum output power or the norm across the three.<br><br>" +
      "Fixed is the better choice when the anatomy supplies a direction you " +
      "trust. On a cortical surface it does: pyramidal cells run perpendicular " +
      "to the sheet, so the surface normal is a physiological statement rather " +
      "than a convenience. You get a third of the unknowns, a better " +
      "conditioned problem, and a constraint table of " + n + " &times; " + n +
      " instead of " + 3 * n + " &times; " + 3 * n + ".<br><br>" +
      "Free is safer when that direction is not trustworthy: a coarse or " +
      "poorly segmented surface, imperfect coregistration, or a volume source " +
      "that has no normal at all. It costs three times the parameters and an " +
      "orientation search that can chase noise at low signal-to-noise, and the " +
      "amplitude it reports is a maximum over directions rather than a " +
      "projection onto one. This panel is fixed throughout so that all four " +
      "methods can be read off a single table of the same size, and the " +
      "subcortical points, which genuinely have no normal, were each given the " +
      "short principal axis of their own structure.</p>"
    );
  }

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
    // Positive elevation puts the camera above the head. Negative looked
    // plausible but rendered an inferior view with the nose at the bottom.
    Top: [0, Math.PI / 2 - 0.001],
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
   * sequential map would invent a polarity that is not there.
   *
   * The exponent matters more than the endpoints. Interpolating between sensors
   * pulls values towards the mean, so a linear mapping leaves most of the head
   * washed out and two small saturated spots; raising |v| to a power below one
   * expands the mid range, which is where the pattern actually lives and where a
   * reader compares one source configuration against another. */
  function fieldColour(v) {
    v = Math.max(-1, Math.min(1, v));
    var lo = [21, 76, 149], mid = [252, 252, 250], hi = [158, 12, 30];
    var a = v < 0 ? lo : hi, f = Math.pow(Math.abs(v), 0.55);
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
    var TYPES = { int16: Int16Array, int8: Int8Array, uint8: Uint8Array };
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
    var truePos = view(P.true_positions);
    var gscale = P.geometry_scale;
    var wscale = P.waveform_scale;
    // Waveforms are stored as bytes; everything else keeps 16 bits. The mix
    // block carries its own scale, chosen from the data so it cannot saturate.
    var bscale = P.byte_scale;
    var mscale = P.mix_scale || wscale;
    var nT = P.n_times;
    var nW = P.wave_times;
    var nSrc = P.n_sources;
    var nCh = P.n_channels;
    var sfreq = P.sfreq;
    var COLOURS = P.methods.map(function (m) {
      return METHOD_COLOUR[m] || "#666";
    });

    /* Variable-length blocks: layouts hold one, two or three sources, so the
     * offsets are cumulative rather than a fixed stride. */
    var trueOff = [], mixOff = [], posOff = [];
    var a = 0, m = 0, q = 0;
    P.scenes.forEach(function (s) {
      trueOff.push(a); mixOff.push(m); posOff.push(q);
      a += s.sources.length * nT;
      m += nCh * s.sources.length;
      q += s.sources.length * 3;
    });
    var reconOff = [], r0 = 0;
    P.results.forEach(function (r) {
      reconOff.push(r0);
      r0 += P.scenes[r.scene].sources.length * nW;
    });

    var sceneOf = {};
    P.scenes.forEach(function (s, i) {
      var w = s.requested;
      sceneOf[[w.layout, w.corr, w.trials, w.morph, w.head].join("|")] = i;
    });
    var resultOf = {};
    P.results.forEach(function (r, i) {
      resultOf[r.scene + "|" + r.method] = i;
    });

    var state = {
      method: 0,
      layout: Math.max(0, P.layouts.map(function (l) { return l.key; }).indexOf("far")),
      morph: Math.max(0, P.morphologies.indexOf("alpha")),
      head: 0,
      corr: P.correlations.length - 2,
      trials: P.trials.length - 1,
      az: -0.9,
      el: 0.2,
      chan0: 0,
      chanShown: 16,
      tStart: 0,
      tSpan: Math.min(250, nT),
    };

    function chip(id, label, wide) {
      return '<div class="cp-control' + (wide ? " cp-control-wide" : "") +
        '"><label>' + label + "</label>" +
        '<div class="cp-methods" id="' + id + '"></div></div>';
    }
    function slider(id, label, n) {
      return '<div class="cp-control"><label>' + label + "</label>" +
        '<input type="range" id="' + id + '" min="0" max="' + (n - 1) +
        '" step="1"><div class="cp-value" id="' + id + '-v"></div></div>';
    }

    root.innerHTML =
      '<div class="cp-controls">' +
      chip("cp-methods", "Method") +
      chip("cp-layout", "Sources", true) +
      chip("cp-morph", "Activity") +
      chip("cp-head", "Head model") +
      slider("cp-corr", "Source correlation", P.correlations.length) +
      slider("cp-trials", "Trials averaged", P.trials.length) +
      "</div>" +

      '<div class="cp-card cp-eq" id="cp-equations"></div>' +

      '<div class="cp-grid">' +
      '<div class="cp-card cp-half"><h4>Where the method says the sources are</h4>' +
      '<p class="cp-hint">Grey rings are the simulated sources, at their real ' +
      "positions, which are not points the beamformer can scan. Coloured crosses " +
      "are the strongest points of its localiser, dashed lines the miss. Drag to " +
      "rotate.</p>" +
      '<div class="cp-views" id="cp-views"></div>' +
      '<canvas id="cp-brain"></canvas>' +
      '<div class="cp-legend"><span>low</span><div class="cp-ramp" id="cp-ramp"></div>' +
      '<span>high</span><span class="cp-key"><i class="cp-ring"></i> simulated</span>' +
      '<span class="cp-key"><i class="cp-cross" id="cp-cross-key"></i> estimated' +
      "</span></div></div>" +

      '<div class="cp-card cp-half"><h4>The constraint table</h4>' +
      '<p class="cp-hint">Every entry is ' +
      "w<sub>i</sub><sup>T</sup>g<sub>j</sub>: what the filter built for source " +
      "<em>i</em> passes of source <em>j</em>. Row <em>i</em> is therefore one " +
      "filter, read across all the sources in the scene.<br><br>" +
      "<b>The diagonal is the distortionless constraint.</b> Every method here " +
      "pins it to one, which is what makes their outputs comparable at all: a " +
      "filter that returns the source at unit gain is measuring amplitude in " +
      "the source's own units.<br><br>" +
      "<b>The off-diagonal is the part that is left free</b>, and it is where " +
      "the whole correlated-source problem lives. LCMV never mentions it, so " +
      "the minimiser sets it to whatever lowers output power most. When two " +
      "sources are correlated, subtracting a scaled copy of one from the other " +
      "does exactly that, so the entry runs to a large negative value and takes " +
      "the target's amplitude with it. MCMV adds the equations that pin it to " +
      "zero. ReciPSIICOS leaves the constraint alone and removes the " +
      "correlation from the covariance instead, so cancelling stops paying. " +
      "ABMC trades the diagonal itself against a template match.<br><br>" +
      "Red is positive, blue negative, and stronger colour means further from " +
      "zero. The numbers are measured by passing a scene containing only source " +
      "<em>j</em> through the finished filter, not read out of the stored " +
      "weights, because the methods keep their weights in different spaces.</p>" +
      '<table class="cp-table" id="cp-gains"></table>' +
      '<div class="cp-readout" id="cp-readout"></div></div>' +

      '<div class="cp-card cp-wide"><h4>What the sensors record</h4>' +
      '<p class="cp-hint">All ' + nCh + " gradiometers. Drag the traces to move " +
      "through channels and time, scroll to zoom, or use the sliders. The map is " +
      "the field at the instant marked on the traces.</p>" +
      '<div class="cp-sensor-row">' +
      '<div><canvas id="cp-topo"></canvas></div>' +
      '<div><canvas id="cp-sensor"></canvas>' +
      '<div class="cp-scrollers">' +
      '<label>channel <input type="range" id="cp-chan" min="0" max="' +
      (nCh - 1) + '" step="1" value="0"></label>' +
      '<label>zoom <input type="range" id="cp-zoom" min="0" max="100" step="1" ' +
      'value="50"></label>' +
      '<span class="cp-value" id="cp-chan-v"></span></div></div></div></div>' +

      '<div class="cp-card cp-half"><h4>True and recovered source waveforms</h4>' +
      '<p class="cp-hint">One shared scale per scene, so a reconstruction that lost ' +
      "half its amplitude looks like it did. Grey is simulated, colour is the " +
      "selected method.</p>" +
      '<canvas id="cp-wave"></canvas>' +
      '<div class="cp-legend" id="cp-wave-key"></div></div>' +

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
      items.forEach(function (item, i) {
        var b = document.createElement("button");
        b.type = "button";
        b.textContent = label(item);
        b.addEventListener("click", function () { state[key] = i; render(); });
        box.appendChild(b);
      });
      return box;
    }
    var methodsBox = chips("cp-methods", P.methods, "method", function (m) {
      return METHOD_LABEL[m] || m;
    });
    /* Grouped rather than one long row. Seventeen entries in a flat list is a
     * wall of buttons; the groups say what kind of question each answers. */
    var GROUP_LABEL = {
      geometry: "geometry",
      bilateral: "same region, both sides",
      circuit: "circuits",
    };
    var layoutBox = (function () {
      var box = root.querySelector("#cp-layout");
      var buttons = [];
      var order = ["geometry", "bilateral", "circuit"];
      order.forEach(function (group) {
        var members = P.layouts
          .map(function (l, i) { return { l: l, i: i }; })
          .filter(function (e) { return e.l.group === group; });
        if (!members.length) return;
        var head = document.createElement("span");
        head.className = "cp-group-label";
        head.textContent = GROUP_LABEL[group] || group;
        box.appendChild(head);
        members.forEach(function (e) {
          var b = document.createElement("button");
          b.type = "button";
          b.textContent = e.l.label;
          b.addEventListener("click", function () {
            state.layout = e.i;
            render();
          });
          box.appendChild(b);
          buttons[e.i] = b;
        });
      });
      return { children: buttons };
    })();
    var morphBox = chips("cp-morph", P.morphologies, "morph", function (m) {
      return MORPH_LABEL[m] || m;
    });
    var headBox = chips("cp-head", P.head_models, "head", function (m) {
      return HEAD_LABEL[m] || m;
    });

    ["corr", "trials"].forEach(function (key) {
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
        0, Math.min(nCh - state.chanShown, state.chan0 - Math.round(dy / 10))
      );
      state.tStart = Math.max(
        0,
        Math.min(nT - state.tSpan, state.tStart - Math.round((dx * state.tSpan) / 500))
      );
      root.querySelector("#cp-chan").value = state.chan0;
      drawSensor();
    });
    sensorCanvas.addEventListener("wheel", function (e) {
      e.preventDefault();
      var centre = state.tStart + state.tSpan / 2;
      state.tSpan = Math.max(
        40, Math.min(nT, Math.round(state.tSpan * (e.deltaY > 0 ? 1.15 : 0.87)))
      );
      state.tStart = Math.max(
        0, Math.min(nT - state.tSpan, Math.round(centre - state.tSpan / 2))
      );
      root.querySelector("#cp-zoom").value =
        Math.round(100 * (1 - (state.tSpan - 40) / (nT - 40)));
      drawSensor();
    }, { passive: false });

    root.querySelector("#cp-chan").addEventListener("input", function (e) {
      state.chan0 = Math.min(nCh - state.chanShown, parseInt(e.target.value, 10));
      drawSensor();
    });
    root.querySelector("#cp-zoom").addEventListener("input", function (e) {
      var f = parseInt(e.target.value, 10) / 100;
      var centre = state.tStart + state.tSpan / 2;
      state.tSpan = Math.round(nT - f * (nT - 40));
      state.tStart = Math.max(
        0, Math.min(nT - state.tSpan, Math.round(centre - state.tSpan / 2))
      );
      drawSensor();
    });

    function current() {
      var key = [
        P.layouts[state.layout].key,
        P.correlations[state.corr],
        P.trials[state.trials],
        P.morphologies[state.morph],
        P.head_models[state.head],
      ].join("|");
      var s = sceneOf[key];
      return { scene: s, result: resultOf[s + "|" + P.methods[state.method]] };
    }
    function nOf(scene) { return P.scenes[scene].sources.length; }
    function methodColour() { return COLOURS[state.method]; }

    /* The recording, rebuilt exactly as the build made it. */
    function sensorAt(scene, channel, t) {
      var n = nOf(scene), v = 0;
      for (var i = 0; i < n; i++) {
        v += (mix[mixOff[scene] + channel * n + i] / mscale) *
          (trueTcs[trueOff[scene] + i * nT + t] / bscale);
      }
      return v + P.noise_gain[scene] * (noise[channel * nT + t] / bscale);
    }

    function rotated(x, y, z) {
      var ca = Math.cos(state.az), sa = Math.sin(state.az);
      var X = x * ca - y * sa, Y = x * sa + y * ca;
      var ce = Math.cos(state.el), se = Math.sin(state.el);
      return [X, Y * ce - z * se, Y * se + z * ce];
    }
    /* Screen x, screen y, depth -- in that order, for everything that gets
     * drawn. ``rotated`` returns x, depth, height, and the point cloud used to
     * reorder it on the way into its own array while the markers did not, so
     * every ring and cross was positioned vertically by its *depth*. An
     * anterior structure like vmPFC then drew above the top of the head. */
    function at(arr, i) {
      var p = rotated(
        arr[i * 3] / gscale, arr[i * 3 + 1] / gscale, arr[i * 3 + 2] / gscale
      );
      return [p[0], p[2], p[1]];
    }

    function drawBrain() {
      var c = fitCanvas(brain, 340);
      var ctx = c.ctx;
      var cur = current();
      var res = P.results[cur.result];
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);

      var pts = [], i, p;
      for (i = 0; i < cortex.length / 3; i++) {
        p = at(cortex, i);
        pts.push([p[0], p[1], p[2], -1, 0]);
      }
      var base = cur.result * nSrc;
      for (i = 0; i < nSrc; i++) {
        p = at(pos, i);
        pts.push([p[0], p[1], p[2], i, maps[base + i] / 255]);
      }
      var xs = pts.map(function (v) { return v[0]; });
      var ys = pts.map(function (v) { return v[1]; });
      var ds = pts.map(function (v) { return v[2]; });
      var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
      var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
      var minD = Math.min.apply(null, ds), maxD = Math.max.apply(null, ds);
      var scale = 0.82 * Math.min(c.w / (maxX - minX || 1), c.h / (maxY - minY || 1));
      var cx = c.w / 2 - ((minX + maxX) / 2) * scale;
      var cy = c.h / 2 + ((minY + maxY) / 2) * scale;
      var X = function (v) { return cx + v[0] * scale; };
      var Y = function (v) { return cy - v[1] * scale; };

      /* Far first, so nearer points paint over them. The third component is
       * distance *from* the viewer, so this sorts descending; ascending drew
       * the far half of the head on top of the near half. */
      pts.sort(function (u, v) { return v[2] - u[2]; });
      var backdrop = css(root, "--cp-grid");
      for (i = 0; i < pts.length; i++) {
        var pt = pts[i];
        var near = 1 - (pt[2] - minD) / (maxD - minD || 1);
        if (pt[3] < 0) {
          /* Strong depth contrast, because a flat wash of dots reads as a
           * cloud rather than a head and gives no clue which end is frontal.
           * Near points are brighter and larger than far ones. */
          ctx.fillStyle = backdrop;
          ctx.globalAlpha = 0.08 + 0.62 * near * near;
          var sz = 1.3 + 1.3 * near;
          ctx.fillRect(X(pt), Y(pt), sz, sz);
        } else {
          var rgb = rampColour(pt[4]);
          ctx.globalAlpha = (0.2 + 0.8 * pt[4]) * (0.45 + 0.55 * near);
          ctx.fillStyle = "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
          ctx.beginPath();
          ctx.arc(X(pt), Y(pt), 1.4 + 4.5 * pt[4] * pt[4], 0, 6.2832);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;

      // Truth at its real position, not at the grid node standing in for it.
      var n = nOf(cur.scene);
      var truth = [];
      for (i = 0; i < n; i++) truth.push(at(truePos, posOff[cur.scene] / 3 + i));
      var est = res.peaks.map(function (s) { return at(pos, s); });

      /* Markers are drawn at their own depth rather than on top of everything.
       * Painted last regardless of depth, a hippocampus reads as sitting on the
       * surface of the brain and the left and right of a pair look inconsistent
       * as you rotate, because whichever is behind is still drawn in front. */
      // Nearness, not depth: one at the front of the head, zero at the back.
      function nearnessOf(d) {
        return 1 - Math.max(0, Math.min(1, (d - minD) / (maxD - minD || 1)));
      }
      var markers = [];
      truth.forEach(function (t) {
        markers.push({ d: t[2], kind: "truth", p: t });
      });
      est.forEach(function (e) {
        markers.push({ d: e[2], kind: "estimate", p: e });
      });
      truth.forEach(function (t) {
        var best = null, bestD = Infinity;
        est.forEach(function (e) {
          var dd = (X(e) - X(t)) * (X(e) - X(t)) + (Y(e) - Y(t)) * (Y(e) - Y(t));
          if (dd < bestD) { bestD = dd; best = e; }
        });
        if (best && bestD > 30) {
          markers.push({ d: Math.min(t[2], best[2]), kind: "link", p: t, q: best });
        }
      });
      markers.sort(function (u, v) { return v.d - u.d; });
      markers.forEach(function (mk) {
        // Behind the middle of the head: dimmer and thinner, so depth reads.
        var front = nearnessOf(mk.d);
        ctx.globalAlpha = 0.3 + 0.7 * front;
        if (mk.kind === "link") {
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = css(root, "--cp-muted");
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(X(mk.p), Y(mk.p));
          ctx.lineTo(X(mk.q), Y(mk.q));
          ctx.stroke();
          ctx.setLineDash([]);
        } else if (mk.kind === "truth") {
          ctx.strokeStyle = css(root, "--cp-true");
          ctx.lineWidth = 1.4 + 1.4 * front;
          ctx.beginPath();
          ctx.arc(X(mk.p), Y(mk.p), 6 + 4 * front, 0, 6.2832);
          ctx.stroke();
        } else {
          ctx.strokeStyle = methodColour();
          ctx.lineWidth = 1.6 + 1.4 * front;
          var r0 = 4 + 3 * front;
          ctx.beginPath();
          ctx.moveTo(X(mk.p) - r0, Y(mk.p) - r0);
          ctx.lineTo(X(mk.p) + r0, Y(mk.p) + r0);
          ctx.moveTo(X(mk.p) + r0, Y(mk.p) - r0);
          ctx.lineTo(X(mk.p) - r0, Y(mk.p) + r0);
          ctx.stroke();
        }
      });
      ctx.globalAlpha = 1;
      /* An orientation gizmo, because a rotated point cloud gives a reader no
       * way to tell the frontal pole from the occipital one. The three head
       * axes are drawn through the same rotation as the brain: +x to the right
       * ear, +y to the nose, +z to the vertex. Whichever end points towards the
       * viewer is drawn solid and labelled; the other end is faint. */
      (function gizmo() {
        var ox = 34, oy = c.h - 34, len = 22;
        var axes = [
          [[1, 0, 0], "R", "L"],
          [[0, 1, 0], "A", "P"],
          [[0, 0, 1], "S", "I"],
        ];
        ctx.font = "bold 10px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        axes.forEach(function (a) {
          var p3 = rotated(a[0][0], a[0][1], a[0][2]);
          var vx = p3[0], vy = p3[2], vd = p3[1];
          [1, -1].forEach(function (sign) {
            var towards = sign * vd <= 0;
            ctx.globalAlpha = towards ? 0.95 : 0.28;
            ctx.strokeStyle = css(root, "--cp-muted");
            ctx.lineWidth = towards ? 1.6 : 1;
            ctx.beginPath();
            ctx.moveTo(ox, oy);
            ctx.lineTo(ox + sign * vx * len, oy - sign * vy * len);
            ctx.stroke();
            ctx.fillStyle = towards
              ? css(root, "--cp-text")
              : css(root, "--cp-muted");
            ctx.fillText(
              sign > 0 ? a[1] : a[2],
              ox + sign * vx * (len + 8),
              oy - sign * vy * (len + 8)
            );
          });
        });
        ctx.globalAlpha = 1;
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
      })();

      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(
        "error " + res.peak_errors.map(function (v) { return v.toFixed(0); }).join(", ") +
        " mm", 72, c.h - 12
      );
    }

    var topoCache = {};
    function drawTopo() {
      var cur = current();
      var canvas = root.querySelector("#cp-topo");
      var c = fitCanvas(canvas, 260);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var R = Math.min(c.w, c.h) / 2 - 18;
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
            for (var mm = 0; mm < nCh; mm++) {
              var ex = dx - sensorPos[mm * 2] / wscale;
              var ey = dy + sensorPos[mm * 2 + 1] / wscale;
              var w = 1 / (ex * ex + ey * ey + 0.0006);
              num += w * vals[mm];
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

      /* Iso-contours. A smooth wash of colour is hard to read a gradient from;
       * the lines are what make a field map legible. */
      ctx.strokeStyle = "rgba(0,0,0,0.4)";
      ctx.lineWidth = 0.9;
      [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8].forEach(function (level) {
        ctx.beginPath();
        for (var y = 0; y < cache.H - 1; y++) {
          for (var x = 0; x < cache.W - 1; x++) {
            var v0 = cache.grid[y * cache.W + x];
            var vx = cache.grid[y * cache.W + x + 1];
            var vy = cache.grid[(y + 1) * cache.W + x];
            if (!isNaN(v0) && !isNaN(vx) && (v0 - level) * (vx - level) < 0) {
              ctx.moveTo(x + (level - v0) / (vx - v0), y);
              ctx.lineTo(x + (level - v0) / (vx - v0), y + 1);
            }
            if (!isNaN(v0) && !isNaN(vy) && (v0 - level) * (vy - level) < 0) {
              ctx.moveTo(x, y + (level - v0) / (vy - v0));
              ctx.lineTo(x + 1, y + (level - v0) / (vy - v0));
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
      ctx.stroke();
      ctx.beginPath();
      ctx.ellipse(ox + R, oy, 5, 12, 0, 0, 6.2832);
      ctx.stroke();

      ctx.fillStyle = "rgba(0,0,0,0.5)";
      for (var s2 = 0; s2 < nCh; s2++) {
        ctx.beginPath();
        ctx.arc(
          ox + (sensorPos[s2 * 2] / wscale) * R,
          oy - (sensorPos[s2 * 2 + 1] / wscale) * R, 1.2, 0, 6.2832
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
      var c = fitCanvas(sensorCanvas, 260);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var padL = 38, padR = 6, padT = 6, padB = 18;
      var shown = Math.min(state.chanShown, nCh - state.chan0);
      var lane = (c.h - padT - padB) / shown;
      var t0 = state.tStart, t1 = Math.min(nT, state.tStart + state.tSpan);

      // One scale across the visible channels, so relative amplitude is real.
      var peak = 1e-9, k, t;
      for (k = 0; k < shown; k++) {
        for (t = t0; t < t1; t++) {
          var v0 = Math.abs(sensorAt(cur.scene, state.chan0 + k, t));
          if (v0 > peak) peak = v0;
        }
      }
      ctx.strokeStyle = css(root, "--cp-muted");
      ctx.lineWidth = 0.9;
      ctx.globalAlpha = 0.85;
      for (k = 0; k < shown; k++) {
        ctx.beginPath();
        for (t = t0; t < t1; t++) {
          var v = sensorAt(cur.scene, state.chan0 + k, t) / peak;
          var x = padL + ((t - t0) / (t1 - t0 - 1 || 1)) * (c.w - padL - padR);
          var y = padT + lane * (k + 0.5) - v * lane * 0.46;
          if (t === t0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "9px system-ui, sans-serif";
      for (k = 0; k < shown; k += Math.max(1, Math.ceil(shown / 8))) {
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
      ctx.fillText(((t1 / sfreq) * 1000).toFixed(0) + " ms", c.w - 50, c.h - 5);
      root.querySelector("#cp-chan-v").textContent =
        "channels " + (state.chan0 + 1) + "–" + (state.chan0 + shown) +
        " of " + nCh;
    }

    function drawWave() {
      var cur = current();
      var n = nOf(cur.scene);
      var c = fitCanvas(root.querySelector("#cp-wave"), 260);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var pad = 8;
      var lane = (c.h - 2 * pad) / n;
      /* Each block is stored at its own scale so neither loses precision; they
       * are put back on one scale here, which is what makes a reconstruction
       * that lost half its amplitude look like it did. */
      var tScale = P.true_scale[cur.scene];
      var rScale = P.recon_scale[cur.result];
      var shared = Math.max(tScale, rScale);
      for (var k = 0; k < n; k++) {
        [
          [trueTcs, trueOff[cur.scene] + k * nT, css(root, "--cp-true"), 1.1,
            tScale / shared],
          [recon, reconOff[cur.result] + k * nW, methodColour(), 1.7,
            rScale / shared],
        ].forEach(function (spec) {
          ctx.beginPath();
          ctx.strokeStyle = spec[2];
          ctx.lineWidth = spec[3];
          for (var t = 0; t < nW; t++) {
            var v = (spec[0][spec[1] + t] / bscale) * spec[4];
            var x = pad + (t / (nW - 1)) * (c.w - 2 * pad);
            var y = pad + lane * (k + 0.5) - v * lane * 0.42;
            if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
        });
        ctx.fillStyle = css(root, "--cp-muted");
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText("source " + (k + 1), 8, pad + lane * k + 12);
      }
      root.querySelector("#cp-wave-key").innerHTML =
        '<span class="cp-key"><i style="background:' + css(root, "--cp-true") +
        ';width:14px;height:2px"></i>simulated</span>' +
        '<span class="cp-key"><i style="background:' + methodColour() +
        ';width:14px;height:3px"></i>recovered by ' +
        (METHOD_LABEL[P.methods[state.method]]) + "</span>" +
        '<span class="cp-key">' + (nW / sfreq).toFixed(2) + " s shown</span>";
    }

    function drawSweep() {
      var c = fitCanvas(root.querySelector("#cp-sweep"), 260);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var padL = 42, padR = 8, padT = 14, padB = 26;
      var half = (c.h - padT - padB - 16) / 2;
      var n = P.correlations.length;
      var xOf = function (i) { return padL + (i / (n - 1)) * (c.w - padL - padR); };
      var series = [], maxErr = 1;
      P.methods.forEach(function (mm, mi) {
        var amp = [], err = [];
        for (var i = 0; i < n; i++) {
          var key = [
            P.layouts[state.layout].key, P.correlations[i],
            P.trials[state.trials], P.morphologies[state.morph],
            P.head_models[state.head],
          ].join("|");
          var r = P.results[resultOf[sceneOf[key] + "|" + mm]];
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
          ctx.strokeStyle = COLOURS[s.mi];
          ctx.globalAlpha = on ? 1 : 0.35;
          ctx.lineWidth = on ? 2.4 : 1.3;
          ctx.beginPath();
          for (var i = 0; i < n; i++) {
            var y = top + height - Math.min(1, s[key][i] / maxV) * height;
            if (i === 0) ctx.moveTo(xOf(i), y); else ctx.lineTo(xOf(i), y);
          }
          ctx.stroke();
          if (on) {
            var yc = top + height - Math.min(1, s[key][state.corr] / maxV) * height;
            ctx.fillStyle = COLOURS[s.mi];
            ctx.beginPath();
            ctx.arc(xOf(state.corr), yc, 4.5, 0, 6.2832);
            ctx.fill();
          }
          ctx.globalAlpha = 1;
        });
      }
      var ampTop = 1.2;
      series.forEach(function (x) {
        x.amp.forEach(function (v) { ampTop = Math.max(ampTop, v); });
      });
      ampTop = Math.ceil(ampTop * 10) / 10;
      panel(padT, half, "amp", ampTop, "amplitude");
      panel(padT + half + 16, half, "err",
        Math.max(10, Math.ceil(maxErr / 10) * 10), "error (mm)");
      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "10px system-ui, sans-serif";
      for (var i = 0; i < n; i++) {
        ctx.fillText(P.correlations[i].toFixed(2), xOf(i) - 10, c.h - 8);
      }
      root.querySelector("#cp-sweep-key").innerHTML = P.methods
        .map(function (mm, i) {
          return '<span class="cp-key"><i style="background:' + COLOURS[i] +
            ';width:14px;height:3px;border-radius:2px"></i>' +
            (METHOD_LABEL[mm] || mm) + "</span>";
        }).join("");
    }

    function drawTable() {
      var cur = current();
      var res = P.results[cur.result];
      var g = res.gains, n = g.length, i, j;
      var html = "<tr><th></th>";
      for (j = 0; j < n; j++) html += "<th>at " + (j + 1) + "</th>";
      html += "</tr>";
      for (i = 0; i < n; i++) {
        html += "<tr><th>filter " + (i + 1) + "</th>";
        for (j = 0; j < n; j++) {
          var v = g[i][j];
          var mag = Math.min(1, Math.abs(v));
          var col = v >= 0
            ? "rgba(188,75,35," + (0.12 + 0.55 * mag) + ")"
            : "rgba(42,111,151," + (0.12 + 0.55 * mag) + ")";
          html += '<td style="background:' + col + '">' +
            (v >= 0 ? "+" : "") + v.toFixed(3) + "</td>";
        }
        html += "</tr>";
      }
      root.querySelector("#cp-gains").innerHTML = html;

      /* Signed, because the sign is the whole distinction: a negative
       * off-diagonal is the filter subtracting a neighbour, a positive one is
       * the neighbour leaking in. Reporting the magnitude called both
       * "cancelling". */
      var worst = 0;
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          if (i !== j && Math.abs(g[i][j]) > Math.abs(worst)) worst = g[i][j];
        }
      }
      var behaviour = "none";
      if (n > 1) {
        behaviour =
          Math.abs(worst) <= 0.2 ? "controlled" : worst < 0 ? "cancelling" : "leaking";
      }
      root.querySelector("#cp-readout").innerHTML =
        '<div><span>Largest off-diagonal</span><strong>' +
        (n > 1 ? (worst >= 0 ? "+" : "") + worst.toFixed(3) : "n/a") +
        "</strong></div>" +
        '<div><span>Amplitude delivered</span><strong>' +
        (res.amplitude_ratio[0] * 100).toFixed(0) + "%</strong></div>" +
        '<div><span>Localisation error</span><strong>' +
        res.peak_errors[0].toFixed(0) + " mm</strong></div>" +
        '<div><span>Neighbour</span><strong>' + behaviour + "</strong></div>";
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
      var m = P.model || null;
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
        " mm from the source</div>" +
        '<p class="cp-note-block"><b>What &ldquo;amplitude delivered&rdquo; ' +
        "means.</b> It is what this filter does to the source's amplitude: each " +
        "constrained source's gain, weighted by how much of this source's " +
        "waveform it carries. There is no noise in it, so it says what the " +
        "filter <em>does</em> rather than how well one short recording measures " +
        "it.<br><br>" +
        "Below 100 per cent is cancellation: the filter is subtracting a " +
        "correlated neighbour and taking the target with it, and the " +
        "off-diagonal is negative. Above 100 per cent is the opposite failure, " +
        "leakage, where a positive off-diagonal adds the neighbour instead. The " +
        "readout beside the constraint table gives that entry with its sign, so " +
        "the two cases can be told apart at a glance.<br><br>" +
        "The obvious measure, the output's own amplitude over the truth's, was " +
        "tried and rejected: the output is the source plus whatever " +
        "interference survives, so at one trial it reads about 570 per cent for " +
        "both LCMV and MCMV, separating them by two per cent where the filters " +
        "themselves differ by a factor of seven.</p>" +
        '<div class="cp-eq-note">' + HEAD_NOTE[P.head_models[state.head]] +
        "</div>" +
        "<h5>The sizes of everything in that equation</h5>" +
        dimensions(P, n, P.methods[state.method]) +
        (m
          ? '<p class="cp-provenance">Head model: MNE\u2019s <code>' + m.subject +
            "</code> subject, " + m.bem + ", " + m.channels + " " +
            m.channel_type + ". Scan grid " + m.n_scan + " points: " +
            m.n_cortical + " on an " + m.surface + " cortical surface at " +
            m.cortical_spacing_mm + " mm, plus " + m.n_subcortical + " inside " +
            m.subcortical_structures.length + " subcortical structures (" +
            m.subcortical_structures.join(", ").replace(/-Proper/g, "") +
            "). Truth drawn from " + m.n_truth + " points. Orientation " +
            m.orientation + ", covariance estimated from " + m.n_times_simulated +
            " samples at " + m.sfreq + " Hz.</p>"
          : "");
    }

    function render() {
      [[methodsBox, "method"], [layoutBox, "layout"], [morphBox, "morph"],
       [headBox, "head"]].forEach(
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
      root.querySelector("#cp-corr-v").textContent = single
        ? "one source, nothing to correlate"
        : "r = " + scene.correlation.toFixed(2) +
          ", " + (scene.separation * 100).toFixed(1) + " cm apart";
      root.querySelector("#cp-trials-v").textContent =
        P.trials[state.trials] + (P.trials[state.trials] === 1 ? " trial" : " trials") +
        " (SNR " + P.snrs[state.trials].toFixed(2) + ")";
      root.querySelector("#cp-cross-key").style.setProperty(
        "--cp-cross-colour", methodColour()
      );
      state.chan0 = Math.min(state.chan0, nCh - state.chanShown);
      /* Keep the instant the field map refers to inside the drawn window. The
       * card says the map is "the field at the instant marked on the traces",
       * and with a fixed default window that instant fell outside it for most
       * scenes, so no marker was drawn and the map referred to something not on
       * screen. */
      var tp = P.topography_time[cur.scene];
      if (tp < state.tStart || tp >= state.tStart + state.tSpan) {
        state.tStart = Math.max(
          0, Math.min(nT - state.tSpan, Math.round(tp - state.tSpan / 2))
        );
      }
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
