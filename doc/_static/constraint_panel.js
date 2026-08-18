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

  /* Not "constraint-panel": the page carries a Sphinx cross-reference label of
 * that name, which renders as an empty <span> with the same id *earlier* in the
 * document. getElementById returns the first match, so the panel built itself
 * inside that span -- an inline element with no cp-root class. Every palette
 * token is defined on .cp-root, so each var() silently fell back to its light
 * default in both themes, and the whole panel laid out inline. */
  var MOUNT = "advance-beamlab-panel";
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


  /* The same equations again, with the sizes put in and multiplied out.
   *
   * The table above says how big each symbol is; this says why those sizes are
   * the ones that work. Readers get the dimensions of the parts and still have
   * to do the conformability in their head, which is exactly the step the
   * algebra is usually skipped over at: that w'Rw collapses to one number is
   * the reason "minimise" is meaningful, and that W'G is n by n is the reason
   * MCMV writes n squared equations where LCMV writes one.
   *
   * Same {C}-style placeholders as the sizes table, filled from the model. */
  var SHAPES = {
    lcmv: [
      ["w<sub>1</sub><sup>T</sup>R&thinsp;w<sub>1</sub>",
       "(1&times;{C})({C}&times;{C})({C}&times;1)", "1&times;1",
       "one number, which is what there is to minimise"],
      ["w<sub>1</sub><sup>T</sup>g<sub>1</sub> = 1",
       "(1&times;{C})({C}&times;1)", "1&times;1",
       "one equation, the whole constraint"],
      ["R<sup>&minus;1</sup>g<sub>1</sub>",
       "({C}&times;{C})({C}&times;1)", "{C}&times;1",
       "a filter, one weight per channel"],
      ["g<sub>1</sub><sup>T</sup>R<sup>&minus;1</sup>g<sub>1</sub>",
       "(1&times;{C})({C}&times;{C})({C}&times;1)", "1&times;1",
       "a scalar, so dividing by it leaves the filter's shape alone"],
    ],
    mcmv: [
      ["W<sup>T</sup>R&thinsp;W",
       "({n}&times;{C})({C}&times;{C})({C}&times;{n})", "{n}&times;{n}",
       "a table, whose trace is the single number being minimised"],
      ["W<sup>T</sup>G = I",
       "({n}&times;{C})({C}&times;{n})", "{n}&times;{n}",
       "{n2} equations, not one -- the whole table is pinned"],
      ["G<sup>T</sup>R<sup>&minus;1</sup>G",
       "({n}&times;{C})({C}&times;{C})({C}&times;{n})", "{n}&times;{n}",
       "small, and inverted; this is what fails when two sources coincide"],
      ["R<sup>&minus;1</sup>G&thinsp;(G<sup>T</sup>R<sup>&minus;1</sup>G)<sup>&minus;1</sup>",
       "({C}&times;{C})({C}&times;{n})({n}&times;{n})", "{C}&times;{n}",
       "all {n} filters at once, which is why they cannot be built separately"],
    ],
    recipsiicos: [
      ["vec(R)", "({q}&times;{q}) unrolled", "{q2}&times;1",
       "the covariance as a vector, which is the space the edit happens in"],
      ["U<sub>K</sub><sup>T</sup>vec(R)",
       "({K}&times;{q2})({q2}&times;1)", "{K}&times;1",
       "the covariance's coordinates in the retained subspace"],
      ["U<sub>K</sub>(U<sub>K</sub><sup>T</sup>vec(R))",
       "({q2}&times;{K})({K}&times;1)", "{q2}&times;1",
       "back to a vectorised covariance, minus what was projected away"],
      ["w<sub>1</sub><sup>T</sup>R&#771;&thinsp;w<sub>1</sub>",
       "(1&times;{C})({C}&times;{C})({C}&times;1)", "1&times;1",
       "LCMV's objective again, on the edited covariance"],
    ],
    abmc: [
      ["w<sub>1</sub><sup>T</sup>X",
       "(1&times;{C})({C}&times;{T})", "1&times;{T}",
       "the filter's output, a time course the template is compared with"],
      ["(w<sub>1</sub><sup>T</sup>X)&middot;u",
       "(1&times;{T})({T}&times;1)", "1&times;1",
       "one number: how much the output looks like the template"],
      ["g<sub>1</sub> + P&thinsp;c<sub>1</sub>",
       "({C}&times;1) + ({C}&times;1)", "{C}&times;1",
       "the leadfield steered towards the template, P setting how far"],
      ["R<sup>&minus;1</sup>(g<sub>1</sub> + P&thinsp;c<sub>1</sub>)",
       "({C}&times;{C})({C}&times;1)", "{C}&times;1",
       "a filter again, the same shape LCMV's has"],
    ],
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
  function dimensions(P, n, method, isRecorded) {
    /* The recorded half has its own grid and its own time axis, so the sizes
     * have to follow the dataset. They did not: selecting Recorded MEG left the
     * table reporting the simulated grid and a sample count from a simulation
     * that is not on screen. */
    var C = P.n_channels;
    var V = isRecorded ? P.n_real_sources : P.n_sources;
    var T = isRecorded
      ? P.real_n_times
      : (P.model && P.model.n_times_simulated) || P.n_times_simulated;
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

    /* And the equations again, with those sizes multiplied out. Knowing how big
     * each symbol is still leaves the reader to check conformability in their
     * head, which is the step the algebra is usually skipped over at. */
    var shapes = SHAPES[method];
    var check = "";
    if (shapes) {
      var S2 = {
        C: C, V: V, T: T, n: n, n2: n * n,
        K: (P.model && P.model.recipsiicos_rank) || 0,
        q: (P.model && P.model.recipsiicos_virtual) || 0,
      };
      S2.q2 = S2.q * S2.q;
      check =
        "<h5>The same equations, with the sizes put in</h5>" +
        '<p class="cp-hint">Every line multiplies out to the shape on its ' +
        "right. It is worth reading down the last column: an objective has to " +
        "collapse to a single number before &ldquo;minimise&rdquo; means " +
        "anything, and a constraint is as many equations as its result has " +
        "entries.</p>" +
        '<table class="cp-dims cp-shapes"><tr><th>expression</th>' +
        "<th>sizes</th><th>gives</th><th></th></tr>" +
        shapes
          .map(function (r) {
            return '<tr><td class="cp-sym">' + fillSizes(r[0], S2) +
              '</td><td class="cp-size">' + fillSizes(r[1], S2) +
              '</td><td class="cp-size">' + fillSizes(r[2], S2) +
              "</td><td>" + fillSizes(r[3], S2) + "</td></tr>";
          })
          .join("") +
        "</table>";
      extra += check;
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

  /* Which palette to use, decided by looking rather than by asking.
   *
   * The host signals its mode through attributes, and the panel guessed wrong
   * for a long time because the theme keeps the user's *setting* in data-theme
   * ("auto" by default) and the resolved mode in data-mode. Reading either is
   * guessing at a convention. Measuring the page's own background colour is
   * not: whatever attribute a host uses, and whether it uses one at all, the
   * page is either dark or it is not, and the panel has to match it.
   *
   * The class is applied to the root element, so the whole palette -- including
   * every colour the canvases read at draw time -- follows from one decision. */
  function applyPalette(root) {
    var probe = document.body || document.documentElement;
    var colour = getComputedStyle(probe).backgroundColor || "";
    var parts = colour.match(/[\d.]+/g);
    var dark = false;
    if (parts && parts.length >= 3 && (parts.length < 4 || parseFloat(parts[3]) > 0)) {
      // Relative luminance, the same weighting the contrast ratios use.
      var lin = [0, 1, 2].map(function (i) {
        var v = parseFloat(parts[i]) / 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
      });
      dark = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.5;
    } else if (window.matchMedia) {
      // A transparent body tells us nothing; fall back to the system setting.
      dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    root.classList.toggle("cp-dark", dark);
    return dark;
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
      resultOf[r.scene + "|" + r.method + "|" + (r.template || "")] = i;
    });

    /* The recorded half. Everything about it is optional: a payload built with
     * --simulation-only has none of these keys, and the panel then runs exactly
     * as it did before the recordings were added. */
    var HAS_REAL = !!(P.real_scenes && P.real_scenes.length);
    var realPos = HAS_REAL ? view(P.real_positions) : null;
    var realMaps = HAS_REAL ? view(P.real_maps) : null;
    var realRecording = HAS_REAL ? view(P.real_recording) : null;
    var realRecon = HAS_REAL ? view(P.real_recon) : null;
    var nRealSrc = HAS_REAL ? P.n_real_sources : 0;
    var nRealT = HAS_REAL ? P.real_n_times : 0;
    var realSceneOf = {};
    if (HAS_REAL) {
      P.real_scenes.forEach(function (s, i) {
        realSceneOf[[s.condition, s.window.join(","), s.trials].join("|")] = i;
      });
    }
    var realResultOf = {};
    if (HAS_REAL) {
      P.real_results.forEach(function (r, i) {
        realResultOf[r.scene + "|" + r.method + "|" + (r.template || "")] = i;
      });
    }
    /* Trial counts are per condition -- a scene records what it actually
     * averaged, which is capped by how many epochs survived rejection -- so the
     * control lists what the chosen condition really offers. */
    function realTrialsFor(conditionIndex) {
      var key = P.real_conditions[conditionIndex].key;
      var out = [];
      P.real_scenes.forEach(function (s) {
        if (s.condition === key && out.indexOf(s.trials) < 0) out.push(s.trials);
      });
      return out.sort(function (a, b) { return a - b; });
    }
    function recorded() { return HAS_REAL && state.dataset === 1; }

    var state = {
      dataset: 0,
      template: 0,
      cond: 0,
      win: 0,
      rtrials: 0,
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
      (HAS_REAL ? chip("cp-dataset", "Dataset") : "") +
      chip("cp-methods", "Method") +
      '<span class="cp-abmc-only">' + chip("cp-template", "Template ABMC seeks") +
      "</span>" +
      '<span class="cp-sim-only">' +
      chip("cp-layout", "Sources", true) +
      chip("cp-morph", "Activity") +
      chip("cp-head", "Head model") +
      slider("cp-corr", "Source correlation", P.correlations.length) +
      slider("cp-trials", "Trials averaged", P.trials.length) +
      "</span>" +
      (HAS_REAL
        ? '<span class="cp-real-only">' +
          chip("cp-cond", "Condition", true) +
          chip("cp-win", "Covariance window") +
          chip("cp-rtrials", "Trials averaged") +
          "</span>"
        : "") +
      "</div>" +

      '<div class="cp-card cp-eq" id="cp-equations"></div>' +

      '<div class="cp-grid">' +
      '<div class="cp-card cp-half"><h4>Localiser peaks against the reference positions</h4>' +
      '<p class="cp-hint">Rings mark the reference positions: the simulated ' +
      "sources at their true coordinates, or, on the recording, an independent " +
      "dipole fit rather than a ground truth. Neither is a point the beamformer " +
      "can scan. Coloured crosses are the localiser peaks, and dashed lines the " +
      "displacement between the two. Drag to rotate.</p>" +
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

      '<div class="cp-card cp-wide"><h4>Sensor time courses and field topography</h4>' +
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

      '<div class="cp-card cp-half"><h4>Reconstructed source time courses</h4>' +
      '<p class="cp-hint">One shared scale per scene, so a reconstruction that lost ' +
      "half its amplitude looks like it did. Where a simulated time course exists " +
      "it is drawn in grey behind the reconstruction, which is in colour.</p>" +
      '<canvas id="cp-wave"></canvas>' +
      '<div class="cp-legend" id="cp-wave-key"></div></div>' +

      '<div class="cp-card cp-half"><h4>Amplitude and localisation error against correlation</h4>' +
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
        b.className = "cp-chip";
        b.textContent = label(item);
        b.addEventListener("click", function () { state[key] = i; render(); });
        box.appendChild(b);
      });
      return box;
    }
    var methodsBox = chips("cp-methods", P.methods, "method", function (m) {
      return METHOD_LABEL[m] || m;
    });
    var datasetBox = null, condBox = null, winBox = null, rtrialsBox = null;
    var TEMPLATE_LABEL = {
      truth: "the source itself",
      matched: "same band",
      mismatched: "wrong band",
      estimate: "first-pass estimate",
    };
    var TEMPLATE_NOTE = {
      truth:
        "The target's own waveform. No experiment has this; it is here to show " +
        "what a perfect template would be worth.",
      matched:
        "An independent signal from the same band as the target. This is the " +
        "realistic best case, and what looking for hippocampal theta means in " +
        "practice: you know the rhythm, not the trace.",
      mismatched:
        "A signal from a different band. A wrong guess about what you are " +
        "looking for, which is the failure this control exists to show.",
      estimate:
        "A first-pass single-source LCMV reconstruction, which is what a " +
        "recording can actually supply: an estimate of the time course, used " +
        "to sharpen the second pass.",
    };
    var templateBox = null;
    templateBox = root.querySelector("#cp-template");
    if (HAS_REAL) {
      datasetBox = chips(
        "cp-dataset",
        ["Simulation", "Recorded MEG"],
        "dataset",
        function (d) { return d; }
      );
      condBox = chips(
        "cp-cond",
        P.real_conditions,
        "cond",
        function (c) { return c.label; }
      );
      winBox = chips("cp-win", P.real_windows, "win", function (w) {
        return Math.round(w[0] * 1000) + "\u2013" + Math.round(w[1] * 1000) + " ms";
      });
      rtrialsBox = root.querySelector("#cp-rtrials");
    }
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
          b.className = "cp-chip";
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
      b.className = "cp-chip";
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
        Math.min(timeCount() - state.tSpan,
                 state.tStart - Math.round((dx * state.tSpan) / 500))
      );
      root.querySelector("#cp-chan").value = state.chan0;
      drawSensor();
    });
    sensorCanvas.addEventListener("wheel", function (e) {
      e.preventDefault();
      var centre = state.tStart + state.tSpan / 2;
      state.tSpan = Math.max(
        40, Math.min(timeCount(), Math.round(state.tSpan * (e.deltaY > 0 ? 1.15 : 0.87)))
      );
      state.tStart = Math.max(
        0, Math.min(timeCount() - state.tSpan, Math.round(centre - state.tSpan / 2))
      );
      root.querySelector("#cp-zoom").value =
        Math.round(100 * (1 - (state.tSpan - 40) / (timeCount() - 40)));
      drawSensor();
    }, { passive: false });

    root.querySelector("#cp-chan").addEventListener("input", function (e) {
      state.chan0 = Math.min(nCh - state.chanShown, parseInt(e.target.value, 10));
      drawSensor();
    });
    root.querySelector("#cp-zoom").addEventListener("input", function (e) {
      var f = parseInt(e.target.value, 10) / 100;
      var centre = state.tStart + state.tSpan / 2;
      state.tSpan = Math.round(timeCount() - f * (timeCount() - 40));
      state.tStart = Math.max(
        0, Math.min(timeCount() - state.tSpan, Math.round(centre - state.tSpan / 2))
      );
      drawSensor();
    });

    /* The recorded trial counts depend on the condition: a scene records what
     * it actually averaged, and rejection leaves each condition a different
     * number of epochs. Rebuilt whenever the condition changes. */
    /* Rebuilt on a dataset change: the simulated half can offer the target's own
     * waveform and the recorded half cannot, so the choices are not the same. */
    function rebuildTemplates() {
      var list = templateList();
      state.template = Math.min(state.template, Math.max(0, list.length - 1));
      templateBox.innerHTML = "";
      list.forEach(function (name, i) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "cp-chip";
        b.textContent = TEMPLATE_LABEL[name] || name;
        b.addEventListener("click", function () { state.template = i; render(); });
        templateBox.appendChild(b);
      });
    }

    function rebuildRealTrials() {
      if (!HAS_REAL) return;
      var counts = realTrialsFor(state.cond);
      state.rtrials = Math.min(state.rtrials, counts.length - 1);
      rtrialsBox.innerHTML = "";
      counts.forEach(function (n, i) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "cp-chip";
        b.textContent = n === 1 ? "1 trial" : n + " trials";
        b.addEventListener("click", function () { state.rtrials = i; render(); });
        rtrialsBox.appendChild(b);
      });
    }

    /* Only ABMC has a template; the other three were run once and stored with
     * an empty one, so the key is empty for them. */
    function templateList() {
      return recorded() ? (P.real_templates || []) : (P.templates || []);
    }
    /* The sweep draws all four methods at once, so it needs the template key of
     * a named method rather than of the selected one. */
    function templateFor(method) {
      if (method !== "abmc") return "";
      var list = templateList();
      return list.length ? list[Math.min(state.template, list.length - 1)] : "";
    }
    function templateKey() {
      if (P.methods[state.method] !== "abmc") return "";
      var list = templateList();
      return list.length ? list[Math.min(state.template, list.length - 1)] : "";
    }
    function current() {
      if (recorded()) {
        var counts = realTrialsFor(state.cond);
        var rk = [
          P.real_conditions[state.cond].key,
          P.real_windows[state.win].join(","),
          counts[Math.min(state.rtrials, counts.length - 1)],
        ].join("|");
        var rs = realSceneOf[rk];
        return {
          scene: rs,
          result:
            realResultOf[rs + "|" + P.methods[state.method] + "|" + templateKey()],
        };
      }
      var key = [
        P.layouts[state.layout].key,
        P.correlations[state.corr],
        P.trials[state.trials],
        P.morphologies[state.morph],
        P.head_models[state.head],
      ].join("|");
      var s = sceneOf[key];
      return {
        scene: s,
        result: resultOf[s + "|" + P.methods[state.method] + "|" + templateKey()],
      };
    }
    /* The active scene and result, whichever half is showing. */
    function sceneRec(i) { return recorded() ? P.real_scenes[i] : P.scenes[i]; }
    function resultRec(i) { return recorded() ? P.real_results[i] : P.results[i]; }
    function nOf(scene) { return sceneRec(scene).sources.length; }
    function gridPos() { return recorded() ? realPos : pos; }
    function gridN() { return recorded() ? nRealSrc : nSrc; }
    function gridMaps() { return recorded() ? realMaps : maps; }
    function methodColour() { return COLOURS[state.method]; }

    /* The recording. A simulated one is rebuilt from its ingredients, which is
     * what keeps the payload small; a recorded one has no such recipe and is
     * stored. All the recorded traces share one scale, so averaging more trials
     * visibly lowers the noise instead of each scene being renormalised to its
     * own peak -- which is the whole point of that control. */
    function sensorAt(scene, channel, t) {
      if (recorded()) {
        var rec = P.real_scenes[scene].recording;
        return (
          realRecording[(rec * nCh + channel) * nRealT + t] /
          P.real_recording_scale
        );
      }
      var n = nOf(scene), v = 0;
      for (var i = 0; i < n; i++) {
        v += (mix[mixOff[scene] + channel * n + i] / mscale) *
          (trueTcs[trueOff[scene] + i * nT + t] / bscale);
      }
      return v + P.noise_gain[scene] * (noise[channel * nT + t] / bscale);
    }
    /* How many samples the active recording has, and where its time axis starts. */
    function timeCount() { return recorded() ? nRealT : nT; }
    function timeAt(i) {
      return recorded() ? P.real_time0 + i / P.real_sfreq : i / P.sfreq;
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
    /* The same projection for a position that is not in a packed array. The
     * dipole reference is stored per scene in the header, already centred and
     * in millimetres, so it arrives as three numbers rather than an index. */
    function atPoint(x, y, z) {
      var p = rotated(x, y, z);
      return [p[0], p[2], p[1]];
    }

    function drawBrain() {
      var c = fitCanvas(brain, 340);
      var ctx = c.ctx;
      var cur = current();
      var res = resultRec(cur.result);
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);

      var pts = [], i, p;
      for (i = 0; i < cortex.length / 3; i++) {
        p = at(cortex, i);
        pts.push([p[0], p[1], p[2], -1, 0]);
      }
      var gpos = gridPos(), gmaps = gridMaps(), gn = gridN();
      var base = cur.result * gn;
      for (i = 0; i < gn; i++) {
        p = at(gpos, i);
        pts.push([p[0], p[1], p[2], i, gmaps[base + i] / 255]);
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

      /* Simulated: the truth, at its real position rather than at the grid node
       * standing in for it. Recorded: there is no truth, so the rings mark an
       * independent dipole fit instead -- another method's answer, not an
       * answer key. Both are drawn the same way because they play the same
       * role on the page: the thing the crosses are compared against. */
      var n = nOf(cur.scene);
      var truth = [];
      if (recorded()) {
        var ref = P.real_scenes[cur.scene].reference;
        for (i = 0; i < ref.length; i++) {
          truth.push(atPoint(ref[i][0], ref[i][1], ref[i][2]));
        }
      } else {
        for (i = 0; i < n; i++) truth.push(at(truePos, posOff[cur.scene] / 3 + i));
      }
      var est = res.peaks.map(function (s) { return at(gridPos(), s); });

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
      /* Simulated: distance to a known source, which is an error. Recorded:
       * distance to a dipole fit, which is two methods disagreeing. The wording
       * changes because the quantity does. */
      ctx.fillText(
        recorded()
          ? "to dipole fit " +
            res.reference_distance.map(function (v) { return v.toFixed(0); }).join(", ") +
            " mm"
          : "error " +
            res.peak_errors.map(function (v) { return v.toFixed(0); }).join(", ") +
            " mm",
        72, c.h - 12
      );
    }

    /* The instant the field map refers to. The simulated half has it
     * precomputed; the recorded half carries its whole recording, so the same
     * quantity -- the peak of the global field power -- is found here and
     * cached, rather than stored twice. */
    var realPeakCache = {};
    function topoTime(scene) {
      if (!recorded()) return P.topography_time[scene];
      if (realPeakCache[scene] === undefined) {
        var best = 0, bestV = -1;
        for (var t = 0; t < nRealT; t++) {
          var v = 0;
          for (var k = 0; k < nCh; k++) {
            var x = sensorAt(scene, k, t);
            v += x * x;
          }
          if (v > bestV) { bestV = v; best = t; }
        }
        realPeakCache[scene] = best;
      }
      return realPeakCache[scene];
    }
    /* Normalised per scene, both halves. The build stores each simulated field
     * divided by its own peak, so those maps always span the full range. The
     * recorded field is read out of the recording, which is scaled to the
     * loudest sample in the whole recorded half -- so without this the quietest
     * scenes never reach the first iso-contour and the map draws blank. That is
     * backwards: averaging 143 trials gave an emptier map than a single noisy
     * one. The peak is per scene and per instant, so it is cached with the
     * instant it belongs to. */
    var topoPeakCache = {};
    function topoPeak(scene) {
      if (topoPeakCache[scene] === undefined) {
        var t = topoTime(scene), hi = 0;
        for (var k = 0; k < nCh; k++) {
          var v = Math.abs(sensorAt(scene, k, t));
          if (v > hi) hi = v;
        }
        topoPeakCache[scene] = hi > 0 ? hi : 1;
      }
      return topoPeakCache[scene];
    }
    function topoValue(scene, channel) {
      return recorded()
        ? sensorAt(scene, channel, topoTime(scene)) / topoPeak(scene)
        : topo[scene * nCh + channel] / wscale;
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
      var key = state.dataset + "|" + cur.scene + "|" +
        Math.round(c.w) + "|" + Math.round(c.h);

      if (!topoCache[key]) {
        var W = Math.round(c.w), H = Math.round(c.h);
        var img = ctx.createImageData(W, H);
        var grid = new Float32Array(W * H);
        var vals = [];
        for (var k = 0; k < nCh; k++) vals.push(topoValue(cur.scene, k));
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
        "field at " + (timeAt(topoTime(cur.scene)) * 1000).toFixed(0) + " ms",
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
      var t0 = state.tStart, t1 = Math.min(timeCount(), state.tStart + state.tSpan);

      // One scale across the visible channels, so relative amplitude is real.
      // Seeded at zero, not at a small constant. A floor of 1e-9 is larger than
      // a gradiometer ever reads, so every real trace was divided by the floor
      // rather than by its own peak and drew as a flat line.
      var peak = 0, k, t;
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
          var v = peak > 0 ? sensorAt(cur.scene, state.chan0 + k, t) / peak : 0;
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

      var tp = topoTime(cur.scene);
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
      // Recorded time is relative to stimulus onset and starts negative, so the
      // axis has to come from the dataset rather than from a sample count.
      ctx.fillText((timeAt(t0) * 1000).toFixed(0) + " ms", padL, c.h - 5);
      ctx.fillText((timeAt(t1) * 1000).toFixed(0) + " ms", c.w - 50, c.h - 5);
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
      /* Recorded data has no simulated trace to draw beside the recovered one,
       * so only the reconstruction is drawn. The build normalises the pair
       * jointly, by the larger of the two peaks, so the quieter hemisphere
       * genuinely draws quieter -- which is the point, and is why this does not
       * renormalise per source. There is nothing to compare an amplitude
       * against here; what the filter did to it is the constraint table's
       * business, and real_recon_scale carries the physical peaks if a reader
       * of the payload wants them back. */
      var samples = recorded() ? nRealT : nW;
      var tScale = recorded() ? 1 : P.true_scale[cur.scene];
      var rScale = recorded() ? 1 : P.recon_scale[cur.result];
      var shared = Math.max(tScale, rScale);
      for (var k = 0; k < n; k++) {
        (recorded()
          ? [[realRecon, (cur.result * n + k) * nRealT, methodColour(), 1.7, 1]]
          : [
              [trueTcs, trueOff[cur.scene] + k * nT, css(root, "--cp-true"), 1.1,
                tScale / shared],
              [recon, reconOff[cur.result] + k * nW, methodColour(), 1.7,
                rScale / shared],
            ]
        ).forEach(function (spec) {
          ctx.beginPath();
          ctx.strokeStyle = spec[2];
          ctx.lineWidth = spec[3];
          for (var t = 0; t < samples; t++) {
            var v = (spec[0][spec[1] + t] / bscale) * spec[4];
            var x = pad + (t / (samples - 1)) * (c.w - 2 * pad);
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
        (recorded()
          ? ""
          : '<span class="cp-key"><i style="background:' + css(root, "--cp-true") +
            ';width:14px;height:2px"></i>simulated</span>') +
        '<span class="cp-key"><i style="background:' + methodColour() +
        ';width:14px;height:3px"></i>recovered by ' +
        (METHOD_LABEL[P.methods[state.method]]) + "</span>" +
        '<span class="cp-key">' +
        (recorded()
          ? (nRealT / P.real_sfreq).toFixed(2) + " s shown, no truth to compare"
          : (nW / sfreq).toFixed(2) + " s shown") +
        "</span>";
    }

    function drawSweep() {
      var c = fitCanvas(root.querySelector("#cp-sweep"), 260);
      var ctx = c.ctx;
      ctx.fillStyle = css(root, "--cp-bg");
      ctx.fillRect(0, 0, c.w, c.h);
      var padL = 42, padR = 8, padT = 14, padB = 26;
      var half = (c.h - padT - padB - 16) / 2;
      /* Simulated: the correlation axis, which is the page's main variable.
       * Recorded: there is no correlation to set -- it is measured, not chosen
       * -- so the axis becomes the one real control that does sweep, the number
       * of trials averaged. */
      var counts = recorded() ? realTrialsFor(state.cond) : null;
      var n = recorded() ? counts.length : P.correlations.length;
      var here = recorded() ? Math.min(state.rtrials, n - 1) : state.corr;
      var xOf = function (i) { return padL + (i / (n - 1 || 1)) * (c.w - padL - padR); };
      var series = [], maxErr = 1;
      P.methods.forEach(function (mm, mi) {
        var amp = [], err = [];
        for (var i = 0; i < n; i++) {
          var r;
          if (recorded()) {
            var rk = [
              P.real_conditions[state.cond].key,
              P.real_windows[state.win].join(","),
              counts[i],
            ].join("|");
            r = P.real_results[
              realResultOf[realSceneOf[rk] + "|" + mm + "|" + templateFor(mm)]
            ];
          } else {
            var key = [
              P.layouts[state.layout].key, P.correlations[i],
              P.trials[state.trials], P.morphologies[state.morph],
              P.head_models[state.head],
            ].join("|");
            r = P.results[resultOf[sceneOf[key] + "|" + mm + "|" + templateFor(mm)]];
          }
          // The worst peak, not the best. Each entry is already the distance
          // from one peak to its nearest dipole, so a further minimum reports
          // the best of four pairings and hides a peak 100 mm away.
          var e = recorded()
            ? Math.max.apply(null, r.reference_distance)
            : r.peak_errors[0];
          amp.push(r.amplitude_ratio[0]);
          err.push(e);
          if (e > maxErr) maxErr = e;
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
            // Clamped both ways. Only the top was clamped, so a negative
            // delivered amplitude -- which the page elsewhere calls out as
            // important -- was drawn below the amplitude panel's zero line and
            // over the error plot underneath it.
            var y = top + height -
              Math.max(0, Math.min(1, s[key][i] / maxV)) * height;
            if (i === 0) ctx.moveTo(xOf(i), y); else ctx.lineTo(xOf(i), y);
          }
          ctx.stroke();
          if (on) {
            var yc = top + height -
              Math.max(0, Math.min(1, s[key][here] / maxV)) * height;
            ctx.fillStyle = COLOURS[s.mi];
            ctx.beginPath();
            ctx.arc(xOf(here), yc, 4.5, 0, 6.2832);
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
      panel(padT + half + 16, half,
        "err", Math.max(10, Math.ceil(maxErr / 10) * 10),
        recorded() ? "to dipole (mm)" : "error (mm)");
      ctx.fillStyle = css(root, "--cp-muted");
      ctx.font = "10px system-ui, sans-serif";
      for (var i = 0; i < n; i++) {
        ctx.fillText(
          recorded() ? String(counts[i]) : P.correlations[i].toFixed(2),
          xOf(i) - 10, c.h - 8
        );
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
      var res = resultRec(cur.result);
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
        '<div><span>Amplitude retained</span><strong>' +
        (res.amplitude_ratio[0] * 100).toFixed(0) + "%</strong></div>" +
        (recorded()
          ? '<div><span>Distance to dipole fit</span><strong>' +
            res.reference_distance.map(function (v) { return v.toFixed(0); })
              .join(", ") +
            " mm</strong></div>"
          : '<div><span>Localisation error</span><strong>' +
            res.peak_errors[0].toFixed(0) + " mm</strong></div>") +
        '<div><span>Neighbour effect</span><strong>' + behaviour + "</strong></div>";
    }

    function drawEquations() {
      var cur = current();
      var res = resultRec(cur.result);
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
        "<h4>The problem " + (METHOD_LABEL[P.methods[state.method]]) + " solves</h4>" +
        '<div class="cp-eq-row"><span class="cp-eq-main">' + eq.objective +
        '</span><span class="cp-eq-sub">subject to</span><span class="cp-eq-main">' +
        eq.subject + "</span></div>" +
        '<div class="cp-eq-sol">' + eq.solution + "</div>" +
        '<p class="cp-hint">' + eq.note + "</p>" +
        (P.methods[state.method] === "abmc"
          ? '<p class="cp-note-block"><b>The template u.</b> ' +
            (TEMPLATE_NOTE[templateKey()] || "") +
            " ABMC is steered to the location whose output best matches it, at " +
            "the best lag, and only its shape matters -- the readout is " +
            "invariant to its amplitude. In a real study this is yours to " +
            "choose: an expert-annotated spike, an averaged response, or a " +
            "band-limited signal standing for the rhythm you are after. " +
            "<code>make_abmc</code> takes it as an argument for exactly that " +
            "reason.</p>"
          : "") +
        '<div class="cp-eq-live">with the controls where they are: ' + live +
        " &nbsp;&rarr;&nbsp; " + (res.amplitude_ratio[0] * 100).toFixed(0) +
        "% of the amplitude survives, peak " +
        (recorded()
          ? res.reference_distance.map(function (v) { return v.toFixed(0); })
              .join(" and ") + " mm from the dipole fit"
          : res.peak_errors[0].toFixed(0) + " mm from the source") +
        "</div>" +
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
        '<div class="cp-eq-note">' +
        (recorded()
          ? "<b>Recorded MEG.</b> MNE's <code>sample</code> dataset, the " +
            "condition and covariance window chosen above. There is no truth " +
            "here and no head-model switch: the model is whatever the " +
            "coregistration gives, which is the situation every real analysis " +
            "is in. The constraint table is still exact, because it is the " +
            "filters' response to leadfields rather than a comparison with " +
            "anything. What is gone is the localisation error, and in its " +
            "place is the distance to a dipole fit -- a different method with " +
            "different assumptions, so read it as two estimates disagreeing " +
            "rather than as one of them being wrong."
          : HEAD_NOTE[P.head_models[state.head]]) +
        "</div>" +
        "<h5>The sizes of everything in that equation</h5>" +
        dimensions(P, n, P.methods[state.method], recorded()) +
        (recorded() && m
          ? '<p class="cp-provenance">Recording: MNE\u2019s <code>' +
            m.subject + "</code> dataset, " + m.channels + " " +
            m.channel_type + ", " +
            P.real_scenes[cur.scene].available +
            " epochs available for this condition and " +
            P.real_scenes[cur.scene].trials + " averaged. Sources chosen at " +
            "full resolution from " + m.real_n_full + " vertices, scanned on " +
            m.real_n_scan + " (1 in " + P.real_decimation + " of the surface, " +
            "plus every constrained location). Covariance window " +
            Math.round(P.real_scenes[cur.scene].window[0] * 1000) + "\u2013" +
            Math.round(P.real_scenes[cur.scene].window[1] * 1000) +
            " ms, baseline noise covariance pooled over every epoch in the " +
            "session. Dipole fit goodness " +
            P.real_scenes[cur.scene].reference_gof
              .map(function (g) { return g.toFixed(0) + "%"; })
              .join(" and ") + ".</p>"
          : "") +
        (!recorded() && m
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
      applyPalette(root);
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
      var isAbmc = P.methods[state.method] === "abmc";
      root.classList.toggle("cp-showing-abmc", isAbmc);
      rebuildTemplates();
      for (var q = 0; q < templateBox.children.length; q++) {
        templateBox.children[q].setAttribute(
          "aria-pressed", q === state.template ? "true" : "false"
        );
      }
      if (HAS_REAL) {
        for (var d = 0; d < datasetBox.children.length; d++) {
          datasetBox.children[d].setAttribute(
            "aria-pressed", d === state.dataset ? "true" : "false"
          );
        }
        root.classList.toggle("cp-showing-real", recorded());
        [[condBox, "cond"], [winBox, "win"]].forEach(function (pair) {
          for (var i = 0; i < pair[0].children.length; i++) {
            pair[0].children[i].setAttribute(
              "aria-pressed", i === state[pair[1]] ? "true" : "false"
            );
          }
        });
        rebuildRealTrials();
        var picked = Math.min(state.rtrials, rtrialsBox.children.length - 1);
        for (var t = 0; t < rtrialsBox.children.length; t++) {
          rtrialsBox.children[t].setAttribute(
            "aria-pressed", t === picked ? "true" : "false"
          );
        }
        // The stored time axis differs between the two halves, so a window
        // carried over from the other one can run past the end.
        state.tSpan = Math.min(state.tSpan, timeCount());
        state.tStart = Math.min(state.tStart, timeCount() - state.tSpan);
      }
      var cur = current();
      var scene = sceneRec(cur.scene);
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
      var tp = topoTime(cur.scene);
      if (tp < state.tStart || tp >= state.tStart + state.tSpan) {
        state.tStart = Math.max(
          0, Math.min(timeCount() - state.tSpan, Math.round(tp - state.tSpan / 2))
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
    /* Redraw when the page's theme changes. Every colour on a canvas is read
     * out of a CSS custom property at draw time, so a canvas painted under one
     * palette keeps those colours until something asks for another frame.
     *
     * Watch both attributes: the theme records the user's setting in data-theme
     * and the resolved mode in data-mode, and switching between light and dark
     * while the setting stays on "auto" changes only the second. */
    new MutationObserver(function () {
      topoCache = {};
      realPeakCache = {};
      render();
    }).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme", "data-mode", "class"],
    });

    /* And when the system flips under an "auto" setting, which changes no
     * attribute at all. */
    if (window.matchMedia) {
      var dark = window.matchMedia("(prefers-color-scheme: dark)");
      var onScheme = function () {
        topoCache = {};
        realPeakCache = {};
        render();
      };
      if (dark.addEventListener) dark.addEventListener("change", onScheme);
      else if (dark.addListener) dark.addListener(onScheme);
    }

    render();
  }

  function start() {
    /* By class first. The id is a convenience for linking, but the class is
       what the stylesheet keys on, so selecting by it means a mount that is
       found is always a mount that is styled. */
    var root = document.querySelector("#" + MOUNT + ".cp-root") ||
      document.querySelector(".cp-root") || document.getElementById(MOUNT);
    if (!root) return;
    root.classList.add("cp-root");
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
