/**
 * scan.js — 3D Body Scan
 *
 * Pipeline:
 *  1. Camera preview + pre-load depth AI model (Transformers.js + DepthAnythingV2)
 *  2. Guided 8-angle capture: countdown → grab frame → depth estimation → segmentation
 *  3. Build one mesh "slab" per capture, rotate each to its angle, merge all
 *  4. Three.js viewer (OrbitControls, auto-rotate) → GLTFExporter → .glb download
 *
 * THREE, THREE.OrbitControls, THREE.GLTFExporter, THREE.BufferGeometryUtils are
 * all loaded as UMD globals from the script tags in scan.html.
 * Transformers.js is loaded via dynamic import() inside loadDepthModel().
 */

'use strict';

// ── Constants ──────────────────────────────────────────────────────────────
const CAPTURE_W = 480;    // width fed to depth model (keep low for perf)
const CAPTURE_H = 360;    // height fed to depth model
const DEPTH_Z_SCALE = 90; // world-unit scale for the depth axis
// Depth Anything: larger value = closer. Keep pixels where d >= threshold.
const FG_DEPTH_MIN_FRAC = 0.40; // keep closest 60% of depth range as foreground
const NUM_ANGLES = 8;

// Body-rotation angle (degrees) per step, plus human-readable labels + instructions
const ANGLE_STEPS = [
  { rotation: 0,   label: 'Front',        instruction: 'Face the camera directly — stand tall and still.' },
  { rotation: 45,  label: '45° Left',     instruction: 'Turn your whole body 45° to your left.' },
  { rotation: 90,  label: 'Left Side',    instruction: 'Show your left profile — feet planted, body rotated.' },
  { rotation: 135, label: 'Back-Left',    instruction: 'Continue turning — you\'re almost facing away.' },
  { rotation: 180, label: 'Back',         instruction: 'Face fully away from the camera.' },
  { rotation: 225, label: 'Back-Right',   instruction: 'Continue rotating to your right.' },
  { rotation: 270, label: 'Right Side',   instruction: 'Show your right profile.' },
  { rotation: 315, label: '45° Right',    instruction: 'Almost back to the start — one more!' },
];

// ── Platform detection ─────────────────────────────────────────────────────
const isIOS     = /iP(hone|ad|od)/.test(navigator.userAgent);
const isMobile  = isIOS || /Mobi|Android/i.test(navigator.userAgent);
// Fewer vertices on mobile to keep build time and memory manageable
const VERT_STRIDE = isMobile ? 6 : 4; // sample every N-th pixel in each axis

// ── DOM refs ───────────────────────────────────────────────────────────────
const setupPhase       = document.getElementById('setupPhase');
const capturePhase     = document.getElementById('capturePhase');
const viewerPhase      = document.getElementById('viewerPhase');
const loadingOverlay   = document.getElementById('loadingOverlay');
const loadingMsg       = document.getElementById('loadingMsg');

const setupVideo       = document.getElementById('setupVideo');
const startScanBtn     = document.getElementById('startScanBtn');
const modelStatus      = document.getElementById('modelStatus');

const captureVideo     = document.getElementById('captureVideo');
const captureFlash     = document.getElementById('captureFlash');
const progressDots     = document.getElementById('progressDots');
const angleLabel       = document.getElementById('angleLabel');
const angleInstruction = document.getElementById('angleInstruction');
const captureCountdown = document.getElementById('captureCountdown');
const captureStatus    = document.getElementById('captureStatus');
const silhouetteWrap   = document.getElementById('silhouetteWrap');

const viewer3d         = document.getElementById('viewer3d');
const downloadBtn      = document.getElementById('downloadBtn');
const rescanBtn        = document.getElementById('rescanBtn');
const meshStats        = document.getElementById('meshStats');

// ── Mutable state ──────────────────────────────────────────────────────────
let depthPipeline  = null;   // Transformers.js pipeline instance
let segmenter      = null;   // MediaPipe SelfieSegmentation instance
let currentStream  = null;   // active MediaStream

let captures       = [];     // array of per-angle capture data
let finalMesh      = null;   // THREE.Mesh of the assembled scan
let threeRenderer  = null;   // THREE.WebGLRenderer (so we can dispose on rescan)

// Scratch canvas for grabbing video frames
let grabCanvas, grabCtx;

// ── Utility helpers ────────────────────────────────────────────────────────
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function showPhase(phaseEl) {
  [setupPhase, capturePhase, viewerPhase].forEach(p => {
    p.style.display = (p === phaseEl) ? 'flex' : 'none';
  });
}

function showLoading(msg) {
  loadingMsg.textContent = msg;
  loadingOverlay.style.display = 'flex';
}
function hideLoading() {
  loadingOverlay.style.display = 'none';
}

// ── Camera helpers ─────────────────────────────────────────────────────────
async function startCamera(videoEl) {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  const constraints = {
    video: {
      facingMode: 'user',
      width:  { ideal: 640 },
      height: { ideal: 480 },
    },
    audio: false,
  };
  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = currentStream;
    await videoEl.play();
  } catch (err) {
    modelStatus.textContent = '⚠ Camera access denied — please allow camera and refresh.';
    throw err;
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
}

// ── Depth model loading (Transformers.js) ──────────────────────────────────
async function loadDepthModel() {
  modelStatus.textContent = 'Downloading depth AI model (~25 MB) — one-time, cached after first use…';

  try {
    // Dynamic import keeps the module loading deferred (no upfront bundle)
    const { pipeline, env } = await import(
      'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js'
    );

    // Allow local model caching (uses browser Cache API / IndexedDB)
    env.allowLocalModels = true;
    env.useBrowserCache  = true;

    const device = (typeof navigator.gpu !== 'undefined') ? 'webgpu' : 'wasm';

    depthPipeline = await pipeline(
      'depth-estimation',
      'Xenova/depth-anything-v2-small',
      {
        device,
        progress_callback: ({ status, loaded, total }) => {
          if (status === 'progress' && total > 0) {
            const pct = Math.round((loaded / total) * 100);
            modelStatus.textContent = `Downloading depth model: ${pct}% — cached after first use`;
          }
        },
      }
    );

    // Warmup: one tiny inference so WASM/WebGPU shaders compile before first
    // real capture (avoids a multi-second freeze on the first actual press).
    modelStatus.textContent = 'Warming up model…';
    const warmup = document.createElement('canvas');
    warmup.width = 64; warmup.height = 64;
    await depthPipeline(warmup);

    modelStatus.textContent = '✓ Model ready — click Start Scan when you\'re set!';
    startScanBtn.disabled = false;
  } catch (err) {
    modelStatus.textContent = '⚠ Could not load depth model. Check your internet connection and refresh.';
    console.error('[scan] Depth model load error:', err);
  }
}

// ── Segmentation (MediaPipe SelfieSegmentation) ────────────────────────────
function initSegmenter() {
  return new Promise((resolve, reject) => {
    if (typeof SelfieSegmentation === 'undefined') {
      resolve(null); // script didn't load — fall back to depth-based mask
      return;
    }
    const seg = new SelfieSegmentation({
      locateFile: (f) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}`,
    });
    seg.setOptions({ modelSelection: 1, selfieMode: true });
    seg.onResults(() => {}); // placeholder; overridden per-call
    seg.initialize()
      .then(() => resolve(seg))
      .catch((err) => {
        console.warn('[scan] SelfieSegmentation init failed, using depth mask:', err);
        resolve(null);
      });
  });
}

/**
 * Run selfie segmentation on the current grab canvas.
 * Returns a Uint8Array mask (255 = person, 0 = background) at CAPTURE_W×CAPTURE_H,
 * or null on failure (caller falls back to depth mask).
 */
function runSegmentation(canvas) {
  if (!segmenter) return Promise.resolve(null);

  return new Promise((resolve) => {
    // One-shot results handler
    segmenter.onResults(({ segmentationMask }) => {
      try {
        // segmentationMask is a canvas; read its alpha/red channel
        const tmpC = document.createElement('canvas');
        tmpC.width  = CAPTURE_W;
        tmpC.height = CAPTURE_H;
        const tmpX = tmpC.getContext('2d');
        tmpX.drawImage(segmentationMask, 0, 0, CAPTURE_W, CAPTURE_H);
        const imgData = tmpX.getImageData(0, 0, CAPTURE_W, CAPTURE_H);
        const mask = new Uint8Array(CAPTURE_W * CAPTURE_H);
        for (let i = 0; i < mask.length; i++) {
          // Red channel: 255 = person, 0 = background
          mask[i] = imgData.data[i * 4] > 127 ? 255 : 0;
        }
        resolve(mask);
      } catch (e) {
        resolve(null);
      }
    });
    segmenter.send({ image: canvas }).catch(() => resolve(null));
  });
}

/**
 * Depth-based foreground mask.
 * Depth Anything: larger value ≈ closer to camera.
 * We keep pixels in the closest (1 - FG_DEPTH_MIN_FRAC) fraction.
 */
function depthMask(depthData, w, h) {
  let minD = Infinity, maxD = -Infinity;
  for (let i = 0; i < depthData.length; i++) {
    if (depthData[i] < minD) minD = depthData[i];
    if (depthData[i] > maxD) maxD = depthData[i];
  }
  const range = maxD - minD;
  const threshold = maxD - range * FG_DEPTH_MIN_FRAC; // keep top 60% closest
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < depthData.length; i++) {
    mask[i] = depthData[i] >= threshold ? 255 : 0;
  }
  return mask;
}

// ── Capture flow ───────────────────────────────────────────────────────────
/** Build progress dots in the HUD (called once at scan start). */
function initProgressDots() {
  progressDots.innerHTML = '';
  for (let i = 0; i < NUM_ANGLES; i++) {
    const dot = document.createElement('div');
    dot.className = 'progress-dot';
    dot.id = `dot${i}`;
    progressDots.appendChild(dot);
  }
}

/** Rotate the SVG silhouette to suggest the target body angle (Y-axis tilt). */
function updateSilhouette(rotationDeg) {
  // Map 0-360 body rotation to a CSS perspective skew hint
  // 0° = front-on (no skew), 90° = side (max skew)
  const side = Math.abs(Math.sin((rotationDeg * Math.PI) / 180));
  silhouetteWrap.style.transform = `scaleX(${1 - side * 0.62})`;
}

/**
 * Capture one angle:
 *  1. Show label + countdown
 *  2. Grab frame from captureVideo
 *  3. Run depth estimation + segmentation in parallel
 *  4. Store compacted capture data
 */
async function captureAngle(idx) {
  const step = ANGLE_STEPS[idx];

  // Highlight active dot
  const dot = document.getElementById(`dot${idx}`);
  dot.classList.add('active');

  // Update HUD
  angleLabel.textContent = step.label;
  angleInstruction.textContent = step.instruction;
  captureCountdown.textContent = '';
  captureStatus.textContent = 'Get into position…';
  updateSilhouette(step.rotation);

  // Give user 1.8 s to get into position before countdown starts
  await sleep(1800);

  // 3-second countdown
  for (let c = 3; c >= 1; c--) {
    captureCountdown.textContent = c;
    captureCountdown.classList.remove('tick');
    // Force reflow so the class removal + re-add triggers the animation
    void captureCountdown.offsetWidth;
    captureCountdown.classList.add('tick');
    await sleep(900);
  }
  captureCountdown.textContent = '';

  // Flash
  captureFlash.style.opacity = '0.9';
  setTimeout(() => { captureFlash.style.opacity = '0'; }, 120);

  // Grab frame
  captureStatus.textContent = '📸 Captured — running depth AI…';
  grabCtx.drawImage(captureVideo, 0, 0, CAPTURE_W, CAPTURE_H);
  const colorRGBA = grabCtx.getImageData(0, 0, CAPTURE_W, CAPTURE_H).data;

  // Depth estimation
  let depthVals;
  let depthW = CAPTURE_W, depthH = CAPTURE_H;
  try {
    const result = await depthPipeline(grabCanvas);
    const depthImg = result.depth;             // RawImage (Uint8ClampedArray, 1-ch grayscale)
    depthW = depthImg.width;
    depthH = depthImg.height;
    const ch = depthImg.channels || 1;
    depthVals = new Float32Array(depthW * depthH);
    for (let i = 0; i < depthVals.length; i++) {
      depthVals[i] = depthImg.data[i * ch] / 255; // normalise → [0, 1]
    }
  } catch (err) {
    console.warn('[scan] Depth estimation failed for angle', idx, err);
    depthVals = new Float32Array(CAPTURE_W * CAPTURE_H).fill(0.5);
    depthW = CAPTURE_W;
    depthH = CAPTURE_H;
  }

  // Segmentation (parallel with depth is not possible without workers, so serial)
  captureStatus.textContent = 'Segmenting person…';
  let mask = await runSegmentation(grabCanvas);
  if (!mask) {
    mask = depthMask(depthVals, depthW, depthH);
  }

  // Store — use Float32 for depth, Uint8 for color (keep compact)
  const cap = {
    color:    colorRGBA,  // Uint8ClampedArray RGBA at CAPTURE_W × CAPTURE_H
    depth:    depthVals,  // Float32Array at depthW × depthH
    mask,                 // Uint8Array at depthW × depthH (or CAPTURE_W × CAPTURE_H)
    colorW:   CAPTURE_W,
    colorH:   CAPTURE_H,
    depthW,
    depthH,
    angleRad: (step.rotation * Math.PI) / 180,
  };
  captures.push(cap);

  // Mark dot done
  dot.classList.remove('active');
  dot.classList.add('done');
  captureStatus.textContent = '✓ Done';
}

// ── Mesh construction ──────────────────────────────────────────────────────
/**
 * Convert one capture's depth map + color + mask into a THREE.BufferGeometry.
 *
 * Each pixel (px, py) with depth d becomes a 3D vertex using a pinhole model:
 *   z = d * DEPTH_Z_SCALE           (closer = higher d = higher z)
 *   x = (px - cx) / fx * z
 *   y = -(py - cy) / fy * z         (flip Y: image top → world up)
 *
 * Adjacent valid pixels form two triangles (grid topology).
 * The whole slab is then rotated around the Y-axis by capture angle.
 */
function buildSlabGeometry(cap) {
  const { color, depth, mask, colorW, colorH, depthW, depthH, angleRad } = cap;
  const stride = VERT_STRIDE;
  const gW = Math.floor(depthW / stride);
  const gH = Math.floor(depthH / stride);

  // Approximate pinhole focal lengths (reasonable for typical webcam FOV ~65°)
  const fx = depthW * 0.85;
  const fy = depthH * 0.85;
  const cx = depthW / 2;
  const cy = depthH / 2;

  // Pre-compute rotation (Y-axis)
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);

  // Pass 1: emit valid vertices
  const positions = [];
  const colors    = [];
  const validIdx  = new Int32Array(gW * gH).fill(-1);
  let vi = 0;

  for (let gy = 0; gy < gH; gy++) {
    for (let gx = 0; gx < gW; gx++) {
      const dpx = gx * stride;
      const dpy = gy * stride;
      const di  = dpy * depthW + dpx;

      if (mask[di] < 128) continue; // background

      const d = depth[di]; // 0-1, larger = closer
      if (d < 0.05) continue; // effectively background

      // Unproject
      const z  = d * DEPTH_Z_SCALE;
      const lx = (dpx - cx) / fx * z;
      const ly = -(dpy - cy) / fy * z; // flip Y

      // Rotate around Y axis
      const rx = lx * cosA + z * sinA;
      const rz = -lx * sinA + z * cosA;

      // Sample colour from the full-res colour image (nearest neighbour)
      // Map depth-grid pixel back to colour-image pixel
      const cpx = Math.min(Math.round(dpx * colorW / depthW), colorW - 1);
      const cpy = Math.min(Math.round(dpy * colorH / depthH), colorH - 1);
      const ci  = (cpy * colorW + cpx) * 4;

      positions.push(rx, ly, rz);
      colors.push(color[ci] / 255, color[ci + 1] / 255, color[ci + 2] / 255);
      validIdx[gy * gW + gx] = vi++;
    }
  }

  if (vi === 0) return null;

  // Pass 2: build triangle indices from grid adjacency
  const indices = [];
  for (let gy = 0; gy < gH - 1; gy++) {
    for (let gx = 0; gx < gW - 1; gx++) {
      const tl = validIdx[gy * gW + gx];
      const tr = validIdx[gy * gW + gx + 1];
      const bl = validIdx[(gy + 1) * gW + gx];
      const br = validIdx[(gy + 1) * gW + gx + 1];
      if (tl >= 0 && tr >= 0 && bl >= 0) indices.push(tl, bl, tr);
      if (tr >= 0 && bl >= 0 && br >= 0) indices.push(tr, bl, br);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(colors),    3));
  if (indices.length > 0) {
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
  }
  return geo;
}

/** Merge all slab geometries into one mesh centred at the origin. */
function assembleModel() {
  const geos = [];
  for (const cap of captures) {
    const g = buildSlabGeometry(cap);
    if (g) geos.push(g);
  }
  if (geos.length === 0) return null;

  const merged = THREE.BufferGeometryUtils.mergeGeometries(geos, false);
  geos.forEach(g => g.dispose());

  // Centre the model
  merged.computeBoundingBox();
  const centre = new THREE.Vector3();
  merged.boundingBox.getCenter(centre);
  merged.translate(-centre.x, -centre.y, -centre.z);

  const mat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
  });
  return new THREE.Mesh(merged, mat);
}

// ── Three.js viewer ────────────────────────────────────────────────────────
let threeScene, threeCamera, threeControls, threeRAF;

function initViewer(mesh) {
  // Clean up any previous renderer
  if (threeRenderer) {
    threeRenderer.dispose();
    viewer3d.innerHTML = '';
  }

  const w = viewer3d.clientWidth  || window.innerWidth;
  const h = viewer3d.clientHeight || (window.innerHeight - 80);

  threeScene = new THREE.Scene();
  threeScene.background = new THREE.Color(0x0a0a0f);

  threeCamera = new THREE.PerspectiveCamera(50, w / h, 0.1, 5000);

  threeRenderer = new THREE.WebGLRenderer({ antialias: !isMobile });
  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  threeRenderer.setSize(w, h);
  viewer3d.appendChild(threeRenderer.domElement);

  threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
  threeControls.enableDamping  = true;
  threeControls.dampingFactor  = 0.07;
  threeControls.autoRotate     = true;
  threeControls.autoRotateSpeed = 1.8;
  threeControls.addEventListener('start', () => {
    threeControls.autoRotate = false;
  }, { once: true });

  // Add mesh and fit camera
  threeScene.add(mesh);
  const box    = new THREE.Box3().setFromObject(mesh);
  const centre = box.getCenter(new THREE.Vector3());
  const size   = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  threeCamera.position.set(centre.x, centre.y + size.y * 0.15, centre.z + maxDim * 2.0);
  threeControls.target.copy(centre);
  threeControls.update();

  // Responsive resize
  const ro = new ResizeObserver(() => {
    const nw = viewer3d.clientWidth;
    const nh = viewer3d.clientHeight;
    if (nw > 0 && nh > 0) {
      threeCamera.aspect = nw / nh;
      threeCamera.updateProjectionMatrix();
      threeRenderer.setSize(nw, nh);
    }
  });
  ro.observe(viewer3d);

  // Render loop
  if (threeRAF) cancelAnimationFrame(threeRAF);
  function animate() {
    threeRAF = requestAnimationFrame(animate);
    threeControls.update();
    threeRenderer.render(threeScene, threeCamera);
  }
  animate();
}

// ── Export ─────────────────────────────────────────────────────────────────
async function exportGLB() {
  if (!finalMesh) return;
  downloadBtn.disabled = true;
  downloadBtn.textContent = 'Preparing…';

  try {
    const exporter = new THREE.GLTFExporter();
    const glb = await new Promise((resolve, reject) => {
      exporter.parse(threeScene, resolve, reject, { binary: true });
    });

    const blob     = new Blob([glb], { type: 'model/gltf-binary' });
    const filename = `body-scan-${Date.now()}.glb`;

    // iOS: prefer Web Share API so the file lands in Files / AR Quick Look
    if (isIOS && navigator.canShare) {
      const file = new File([blob], filename, { type: 'model/gltf-binary' });
      if (navigator.canShare({ files: [file] })) {
        try {
          await navigator.share({ files: [file], title: 'HandTrack 3D Body Scan' });
          return;
        } catch (e) {
          if (e.name !== 'AbortError') console.warn('[scan] Share failed:', e);
        }
      }
    }

    // Desktop / Android fallback: anchor download
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href     = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('[scan] GLB export failed:', err);
    alert('Export failed. Please try again.');
  } finally {
    downloadBtn.disabled = false;
    downloadBtn.innerHTML = `
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 3v12"/>
      </svg>
      Download .glb`;
  }
}

// ── Main scan flow ─────────────────────────────────────────────────────────
async function runScanFlow() {
  // Re-init scratch canvas
  grabCanvas = document.createElement('canvas');
  grabCanvas.width  = CAPTURE_W;
  grabCanvas.height = CAPTURE_H;
  grabCtx = grabCanvas.getContext('2d', { willReadFrequently: true });

  // Switch camera to capture video element
  await startCamera(captureVideo);
  showPhase(capturePhase);

  // Reset captures
  captures = [];

  // Init segmenter (non-blocking, failure is graceful)
  segmenter = await initSegmenter();

  // Build progress dots
  initProgressDots();

  // Capture all 8 angles sequentially
  for (let i = 0; i < NUM_ANGLES; i++) {
    await captureAngle(i);
  }

  // Build the 3D model
  captureStatus.textContent = '';
  showLoading('Building 3D mesh — this may take a moment…');
  await sleep(60); // yield to DOM paint

  finalMesh = assembleModel();

  // Free capture memory now that mesh is built
  captures = [];

  hideLoading();

  if (!finalMesh) {
    alert('Could not build 3D model — no valid depth data. Please try again with better lighting.');
    showPhase(setupPhase);
    await startCamera(setupVideo);
    return;
  }

  // Show viewer
  showPhase(viewerPhase);
  await sleep(30); // ensure viewer3d has layout dimensions
  initViewer(finalMesh);

  // Stats
  const vertCount = finalMesh.geometry.attributes.position.count;
  const triCount  = finalMesh.geometry.index
    ? finalMesh.geometry.index.count / 3
    : vertCount / 3;
  meshStats.textContent =
    `${vertCount.toLocaleString()} vertices · ${Math.round(triCount).toLocaleString()} triangles`;
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  // Start the setup preview camera
  try {
    await startCamera(setupVideo);
  } catch {
    // error message already set in startCamera
  }

  // Load depth model in the background while user reads instructions
  loadDepthModel(); // intentionally not awaited — progress shown via modelStatus

  // ── Event listeners ──
  startScanBtn.addEventListener('click', async () => {
    if (!depthPipeline || startScanBtn.disabled) return;
    startScanBtn.disabled = true;
    startScanBtn.textContent = 'Starting…';
    stopCamera(); // release setup camera before opening capture camera
    try {
      await runScanFlow();
    } catch (err) {
      console.error('[scan] Scan flow error:', err);
      hideLoading();
      showPhase(setupPhase);
      await startCamera(setupVideo);
      startScanBtn.disabled  = false;
      startScanBtn.textContent = 'Start Scan';
    }
  });

  downloadBtn.addEventListener('click', exportGLB);

  rescanBtn.addEventListener('click', async () => {
    // Clean up Three.js
    if (threeRAF) cancelAnimationFrame(threeRAF);
    if (finalMesh) {
      finalMesh.geometry.dispose();
      finalMesh.material.dispose();
      finalMesh = null;
    }
    if (threeRenderer) {
      threeRenderer.dispose();
      viewer3d.innerHTML = '';
      threeRenderer = null;
    }
    meshStats.textContent = '';
    captures = [];

    showPhase(setupPhase);
    try { await startCamera(setupVideo); } catch { /* handled */ }
    startScanBtn.disabled  = !depthPipeline;
    startScanBtn.textContent = 'Start Scan';
  });
}

init().catch(console.error);
