// The echoes logo in interactive 3D.
//
// The logo is an Asymptote scene (img/cover.asy in the echoes manual): a
// translucent sphere holding 100 ellipsoids and 100 superspheres. Asymptote's
// own WebGL export of it weighs 16 MB; the thirteen numbers that define each
// inclusion, recovered by _scripts/build_echoes_scene.py of the website, weigh
// 23 kB. This script rebuilds the meshes from them and draws the scene with
// plain WebGL — no library. It replaces the logo image by a canvas once the
// scene is ready: with no WebGL, or if anything fails, the image simply stays.
//
// The same file serves two sites, and is copied from the website to the echoes
// manual unchanged:
//
//   - on the website (jfbarthelemy.github.io/js/) it is loaded on demand by
//     _config/motion.html, whose shared helpers it uses. On the echoes page the
//     sphere turns slowly (one turn in 40 s) and can be turned by hand — mouse,
//     finger or arrow keys; in the Software grid it turns while hovered;
//   - anywhere else (the echoes manual) it brings the few helpers it needs and
//     starts by itself on the images matched by the `data-target` selector of
//     its <script> tag, reading the scene from `data-scene` (a path from the
//     site root).
//
// The inclusions keep their Asymptote colors in both themes. The container is
// drawn from the page's ink color, so that it reads as glass on a light page
// and on a dark one alike. A reader who asks for reduced motion gets a still
// sphere, which can still be turned by hand.
(function () {
  "use strict";

  var script = document.currentScript;
  var M = window.jfbMotion || standaloneHelpers();

  // The subset of _config/motion.html this script uses, for a page that does
  // not have it. The ink is the body's text color.
  function standaloneHelpers() {
    var H = {};
    H.calm = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    var meta = document.querySelector('meta[name="quarto:offset"]');
    H.offset = meta ? meta.getAttribute("content") : "./";
    var probe = document.createElement("canvas").getContext("2d");
    H.rgb = function () {
      probe.fillStyle = "#000";
      probe.fillStyle = getComputedStyle(document.body).color || "#000";
      var c = probe.fillStyle;
      if (c.charAt(0) === "#") return [1, 3, 5].map(function (i) { return parseInt(c.substr(i, 2), 16) / 255; });
      return c.replace(/[^\d.,]/g, "").split(",").slice(0, 3).map(function (x) { return +x / 255; });
    };
    var listeners = [];
    H.onTheme = function (fn) { listeners.push(fn); };
    new MutationObserver(function () {
      requestAnimationFrame(function () { listeners.forEach(function (fn) { fn(); }); });
    }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    H.loop = function (el, frame, fps) {
      var wanted = false, onScreen = true, raf = 0, last = 0, t = 0;
      var gap = fps ? 1000 / fps - 2 : 0;
      function tick(now) {
        raf = 0;
        if (last && now - last < gap) { schedule(); return; }
        var dt = last ? Math.min((now - last) / 1000, 0.1) : 0;
        last = now;
        t += dt;
        if (frame(dt, t) === false) { wanted = false; last = 0; return; }
        schedule();
      }
      function schedule() {
        if (wanted && onScreen && !document.hidden && !raf) raf = requestAnimationFrame(tick);
        if ((!wanted || !onScreen || document.hidden) && raf) {
          cancelAnimationFrame(raf);
          raf = 0;
          last = 0;
        }
      }
      new IntersectionObserver(function (entries) {
        onScreen = entries[entries.length - 1].isIntersecting;
        schedule();
      }).observe(el);
      document.addEventListener("visibilitychange", schedule);
      return {
        start: function () { wanted = true; schedule(); },
        stop: function () { wanted = false; schedule(); }
      };
    };
    return H;
  }

  var NU = 16, NV = 32;          // mesh resolution of one inclusion
  var TURN = 40;                 // seconds per turn on the echoes page
  var PITCH_MAX = 1.35;          // rad

  // --- Geometry ------------------------------------------------------------

  // x = c + M g_p(u) and its outward normal M^{-T} (sgn uᵢ |uᵢ|^(2 − 1/p)),
  // the gradient of the implicit form Σ |yᵢ|^(2p) = 1 with y = M⁻¹(x − c).
  function inclusionMesh(inc, out) {
    var c = inc.slice(0, 3), m = inc.slice(3, 12), p = inc[12], rgb = inc.slice(13, 16);
    var inv = invT(m);
    var e1 = 1 / p, e2 = 2 - 1 / p;
    var base = out.pos.length / 3;
    for (var i = 0; i <= NU; i++) {
      var t1 = Math.PI * i / NU, s1 = Math.sin(t1), c1 = Math.cos(t1);
      for (var j = 0; j <= NV; j++) {
        var t2 = 2 * Math.PI * j / NV;
        var u = [s1 * Math.cos(t2), s1 * Math.sin(t2), c1];
        var g = [], h = [];
        for (var k = 0; k < 3; k++) {
          var a = Math.abs(u[k]), s = u[k] < 0 ? -1 : 1;
          g.push(s * Math.pow(a, e1));
          h.push(s * Math.pow(Math.max(a, 1e-4), e2));
        }
        for (var r = 0; r < 3; r++) {
          out.pos.push(c[r] + m[3 * r] * g[0] + m[3 * r + 1] * g[1] + m[3 * r + 2] * g[2]);
        }
        var n = [
          inv[0] * h[0] + inv[1] * h[1] + inv[2] * h[2],
          inv[3] * h[0] + inv[4] * h[1] + inv[5] * h[2],
          inv[6] * h[0] + inv[7] * h[1] + inv[8] * h[2]
        ];
        var l = Math.hypot(n[0], n[1], n[2]) || 1;
        out.nor.push(n[0] / l, n[1] / l, n[2] / l);
        out.col.push(rgb[0], rgb[1], rgb[2]);
      }
    }
    for (i = 0; i < NU; i++) {
      for (j = 0; j < NV; j++) {
        var a0 = base + i * (NV + 1) + j, a1 = a0 + 1, b0 = a0 + NV + 1, b1 = b0 + 1;
        out.idx.push(a0, b0, a1, a1, b0, b1);
      }
    }
  }

  // (M⁻¹)ᵀ of a row-major 3×3, row-major.
  function invT(m) {
    var a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    var A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
    var det = a * A + b * B + c * C;
    return [
      A / det, B / det, C / det,
      (c * h - b * i) / det, (a * i - c * g) / det, (b * g - a * h) / det,
      (b * f - c * e) / det, (c * d - a * f) / det, (a * e - b * d) / det
    ];
  }

  function sphereMesh(out) {
    var nu = 32, nv = 64;
    for (var i = 0; i <= nu; i++) {
      var t1 = Math.PI * i / nu;
      for (var j = 0; j <= nv; j++) {
        var t2 = 2 * Math.PI * j / nv;
        var x = Math.sin(t1) * Math.cos(t2), y = Math.cos(t1), z = Math.sin(t1) * Math.sin(t2);
        out.pos.push(x, y, z);
        out.nor.push(x, y, z);
        out.col.push(1, 1, 1);
      }
    }
    for (i = 0; i < nu; i++) {
      for (j = 0; j < nv; j++) {
        var a0 = i * (nv + 1) + j, a1 = a0 + 1, b0 = a0 + nv + 1, b1 = b0 + 1;
        out.idx.push(a0, b0, a1, a1, b0, b1);
      }
    }
  }

  // --- WebGL ---------------------------------------------------------------

  var VS = [
    "attribute vec3 aPos; attribute vec3 aNor; attribute vec3 aCol;",
    "uniform mat3 uRot; uniform float uScale;",
    "varying vec3 vN; varying vec3 vC;",
    "void main() {",
    "  vec3 p = uRot * aPos;",
    "  vN = uRot * aNor; vC = aCol;",
    "  gl_Position = vec4(p.xy * uScale, -p.z * 0.5, 1.0);",
    "}"
  ].join("\n");

  // Lambert plus a GGX highlight, with the constants of Asymptote's material
  // (specular 0.75, shininess 0.7 hence roughness 0.3, Fresnel F₀ = 0.04).
  var FS_INCLUSION = [
    "precision mediump float;",
    "varying vec3 vN; varying vec3 vC;",
    "uniform vec3 uLight;",
    "void main() {",
    "  vec3 N = normalize(vN); if (N.z < 0.0) N = -N;",
    "  vec3 L = normalize(uLight); vec3 V = vec3(0.0, 0.0, 1.0); vec3 H = normalize(L + V);",
    "  float nl = max(dot(N, L), 0.0), nh = max(dot(N, H), 0.0), nv = max(N.z, 1e-3);",
    "  float a2 = 0.09 * 0.09;",
    "  float d = nh * nh * (a2 - 1.0) + 1.0;",
    "  float D = a2 / (3.14159 * d * d);",
    "  float F = 0.04 + 0.96 * pow(1.0 - max(dot(H, V), 0.0), 5.0);",
    "  float spec = 0.75 * D * F / (4.0 * nv) ;",
    "  vec3 col = vC * (0.2 + 0.85 * nl) + vec3(min(spec, 1.2) * nl);",
    "  gl_FragColor = vec4(col, 1.0);",
    "}"
  ].join("\n");

  // The container: a glass of the theme's ink, denser toward the rim, with the
  // same highlight. Premultiplied alpha.
  var FS_GLASS = [
    "precision mediump float;",
    "varying vec3 vN; varying vec3 vC;",
    "uniform vec3 uLight; uniform vec3 uInk; uniform float uBase; uniform float uRim;",
    "void main() {",
    "  vec3 N = normalize(vN);",
    "  float facing = abs(N.z);",
    "  float rim = pow(1.0 - facing, 2.0);",
    "  vec3 L = normalize(uLight); vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));",
    "  float nh = max(dot(N, H), 0.0);",
    "  float spec = pow(nh, 60.0) * 0.55;",
    "  float a = uBase + uRim * rim;",
    "  gl_FragColor = vec4(uInk * a + vec3(spec), min(a + spec, 1.0));",
    "}"
  ].join("\n");

  function program(gl, fs) {
    function sh(type, src) {
      var s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    var p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, "aPos");
    gl.bindAttribLocation(p, 1, "aNor");
    gl.bindAttribLocation(p, 2, "aCol");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  // Upload a mesh in batches of fewer than 65,536 vertices (WebGL1 indices are
  // 16-bit without an extension).
  function upload(gl, meshes) {
    var batches = [], cur = null;
    meshes.forEach(function (m) {
      var n = m.pos.length / 3;
      if (!cur || cur.n + n > 65535) {
        cur = { n: 0, pos: [], nor: [], col: [], idx: [] };
        batches.push(cur);
      }
      var off = cur.n;
      Array.prototype.push.apply(cur.pos, m.pos);
      Array.prototype.push.apply(cur.nor, m.nor);
      Array.prototype.push.apply(cur.col, m.col);
      for (var k = 0; k < m.idx.length; k++) cur.idx.push(m.idx[k] + off);
      cur.n += n;
    });
    return batches.map(function (b) {
      function buf(target, data) {
        var o = gl.createBuffer();
        gl.bindBuffer(target, o);
        gl.bufferData(target, data, gl.STATIC_DRAW);
        return o;
      }
      return {
        pos: buf(gl.ARRAY_BUFFER, new Float32Array(b.pos)),
        nor: buf(gl.ARRAY_BUFFER, new Float32Array(b.nor)),
        col: buf(gl.ARRAY_BUFFER, new Float32Array(b.col)),
        idx: buf(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(b.idx)),
        count: b.idx.length
      };
    });
  }

  function draw(gl, batches) {
    batches.forEach(function (b) {
      [["pos", 0], ["nor", 1], ["col", 2]].forEach(function (a) {
        gl.bindBuffer(gl.ARRAY_BUFFER, b[a[0]]);
        gl.enableVertexAttribArray(a[1]);
        gl.vertexAttribPointer(a[1], 3, gl.FLOAT, false, 0, 0);
      });
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.idx);
      gl.drawElements(gl.TRIANGLES, b.count, gl.UNSIGNED_SHORT, 0);
    });
  }

  // Rotation Rx(pitch) · Ry(yaw), column-major for uniformMatrix3fv.
  function rotation(yaw, pitch) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    // rows of R = Rx · Ry
    var r = [
      cy, 0, sy,
      sp * sy, cp, -sp * cy,
      -cp * sy, sp, cp * cy
    ];
    return new Float32Array([r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]);
  }

  // --- Viewer --------------------------------------------------------------

  function Viewer(img, scene, mode) {
    var canvas = document.createElement("canvas");
    var attrs = { antialias: true, premultipliedAlpha: true, alpha: true };
    var gl = canvas.getContext("webgl", attrs) || canvas.getContext("experimental-webgl", attrs);
    if (!gl) return null;

    var inclusions = [];
    scene.inclusions.forEach(function (inc) {
      var m = { pos: [], nor: [], col: [], idx: [] };
      inclusionMesh(inc, m);
      inclusions.push(m);
    });
    var glass = { pos: [], nor: [], col: [], idx: [] };
    sphereMesh(glass);

    var pIncl = program(gl, FS_INCLUSION), pGlass = program(gl, FS_GLASS);
    var bIncl = upload(gl, inclusions), bGlass = upload(gl, [glass]);
    var light = new Float32Array(scene.light);

    canvas.className = img.className + " echoes-viewer";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "The echoes logo, a sphere holding two hundred inclusions" +
      (mode === "page" ? " — drag, or use the arrow keys, to turn it" : ""));
    if (mode === "page") canvas.tabIndex = 0;

    var yaw = 0, pitch = 0, ink = [0, 0, 0], dark = false;
    var spinning = mode === "page" && !M.calm;

    function readTheme() {
      ink = M.rgb("ink");
      dark = document.body.classList.contains("quarto-dark");
    }

    function render() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
      if (!w || !h) return;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 0);
      // clear() honors the depth mask, which the glass pass leaves off.
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      var rot = rotation(yaw, pitch), scale = 0.96;

      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.disable(gl.CULL_FACE);
      gl.useProgram(pIncl);
      gl.uniformMatrix3fv(gl.getUniformLocation(pIncl, "uRot"), false, rot);
      gl.uniform1f(gl.getUniformLocation(pIncl, "uScale"), scale);
      gl.uniform3fv(gl.getUniformLocation(pIncl, "uLight"), light);
      draw(gl, bIncl);

      // Glass: back half first, then the front half over everything.
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.enable(gl.CULL_FACE);
      gl.useProgram(pGlass);
      gl.uniformMatrix3fv(gl.getUniformLocation(pGlass, "uRot"), false, rot);
      gl.uniform1f(gl.getUniformLocation(pGlass, "uScale"), scale);
      gl.uniform3fv(gl.getUniformLocation(pGlass, "uLight"), light);
      gl.uniform3fv(gl.getUniformLocation(pGlass, "uInk"), new Float32Array(ink));
      gl.uniform1f(gl.getUniformLocation(pGlass, "uBase"), dark ? 0.06 : 0.035);
      gl.uniform1f(gl.getUniformLocation(pGlass, "uRim"), dark ? 0.26 : 0.18);
      [gl.FRONT, gl.BACK].forEach(function (face) {
        gl.cullFace(face);
        draw(gl, bGlass);
      });
    }

    readTheme();
    M.onTheme(function () { readTheme(); render(); });

    var loop = M.loop(canvas, function (dt) {
      if (spinning) yaw += 2 * Math.PI * dt / TURN;
      render();
      return spinning;
    }, 30);

    // Turning by hand.
    var drag = null;
    canvas.addEventListener("pointerdown", function (e) {
      drag = { x: e.clientX, y: e.clientY, spinning: spinning };
      spinning = false;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var s = 3 / Math.max(canvas.clientWidth, 1);
      yaw += (e.clientX - drag.x) * s;
      pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitch + (e.clientY - drag.y) * s));
      drag.x = e.clientX;
      drag.y = e.clientY;
      render();
    });
    function release() {
      if (!drag) return;
      spinning = drag.spinning;
      drag = null;
      if (spinning) loop.start();
    }
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("keydown", function (e) {
      var step = 0.12, used = true;
      if (e.key === "ArrowLeft") yaw -= step;
      else if (e.key === "ArrowRight") yaw += step;
      else if (e.key === "ArrowUp") pitch = Math.max(-PITCH_MAX, pitch - step);
      else if (e.key === "ArrowDown") pitch = Math.min(PITCH_MAX, pitch + step);
      else used = false;
      if (used) { e.preventDefault(); render(); }
    });

    if (mode === "card") {
      var card = img.closest(".quarto-grid-item") || canvas;
      card.addEventListener("mouseenter", function () { if (!M.calm) { spinning = true; loop.start(); } });
      card.addEventListener("mouseleave", function () { spinning = false; });
    }

    img.replaceWith(canvas);
    render();
    if (spinning) loop.start();
    new ResizeObserver(function () { render(); }).observe(canvas);

    // Deterministic frames for the GIF/MP4 exporter: turn fraction in [0, 1).
    return {
      canvas: canvas,
      frame: function (f) { spinning = false; yaw = 2 * Math.PI * f; pitch = 0; render(); }
    };
  }

  M.echoesViewer = function (imgs, scenePath) {
    fetch(M.offset + (scenePath || "images/echoes_scene.json"))
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (scene) {
        var viewers = [];
        imgs.forEach(function (img) {
          try {
            var v = Viewer(img, scene, img.closest(".quarto-grid-item") ? "card" : "page");
            if (v) viewers.push(v);
          } catch (err) {
            if (window.console) console.warn("echoes viewer:", err);
          }
        });
        window.jfbEchoes = viewers;
      })
      .catch(function (err) { if (window.console) console.warn("echoes viewer:", err); });
  };

  // Standalone: start on the images the <script> tag names.
  if (!window.jfbMotion && script && script.getAttribute("data-target") &&
      window.WebGLRenderingContext && "ResizeObserver" in window && "IntersectionObserver" in window) {
    var start = function () {
      var imgs = Array.prototype.slice.call(document.querySelectorAll(script.getAttribute("data-target")));
      if (imgs.length) M.echoesViewer(imgs, script.getAttribute("data-scene"));
    };
    if (document.readyState !== "loading") start();
    else document.addEventListener("DOMContentLoaded", start);
  }
})();
