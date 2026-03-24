/* ══════════════════════════════════════════════════════════
   HandTrack — app.js
   MediaPipe Holistic · hand gestures · face expressions · canvas overlay
══════════════════════════════════════════════════════════ */

'use strict';

// ── DOM refs — shared ─────────────────────────────────────
const videoEl        = document.getElementById('videoEl');
const canvas         = document.getElementById('overlayCanvas');
const ctx            = canvas.getContext('2d');
const cameraSelect   = document.getElementById('cameraSelect');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText    = document.getElementById('loadingText');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const noCameraMsg    = document.getElementById('noCameraMsg');

// ── Resolution constants ──────────────────────────────────
const TRACK_W  = 640,  TRACK_H  = 360;   // MediaPipe processing canvas
const EXPORT_W = 1280, EXPORT_H = 720;   // Screenshot output

// ── DOM refs — hands ──────────────────────────────────────
const infoPanel      = document.getElementById('infoPanel');
const handCountPanel = document.getElementById('handCountPanel');
const secondHandDiv  = document.getElementById('secondHand');
const handCountEl    = document.getElementById('handCount');
const faceDetectedEl = document.getElementById('faceDetected');
const fpsEl          = document.getElementById('fpsCounter');

const gestureLabel  = document.getElementById('gestureLabel');
const coordX        = document.getElementById('coordX');
const coordY        = document.getElementById('coordY');
const pixelX        = document.getElementById('pixelX');
const pixelY        = document.getElementById('pixelY');

const gestureLabel2 = document.getElementById('gestureLabel2');
const coordX2       = document.getElementById('coordX2');
const coordY2       = document.getElementById('coordY2');
const pixelX2       = document.getElementById('pixelX2');
const pixelY2       = document.getElementById('pixelY2');

// ── DOM refs — face ───────────────────────────────────────
const facePanel       = document.getElementById('facePanel');
const expressionLabel = document.getElementById('expressionLabel');
const mouthStateEl    = document.getElementById('mouthState');
const leftEyeStateEl  = document.getElementById('leftEyeState');
const rightEyeStateEl = document.getElementById('rightEyeState');
const browsStateEl    = document.getElementById('browsState');

// ── DOM refs — toggles + drawing ─────────────────────────
const toggleHands   = document.getElementById('toggleHands');
const toggleFace    = document.getElementById('toggleFace');
const toggleDrawBtn = document.getElementById('toggleDraw');
const screenshotBtn = document.getElementById('screenshotBtn');
const drawToolbar   = document.getElementById('drawToolbar');
const clearDrawBtn  = document.getElementById('clearDraw');
const colorSwatches = document.querySelectorAll('.color-swatch');
const flipCameraBtn = document.getElementById('flipCameraBtn');
const fullscreenBtn = document.getElementById('fullscreenBtn');

// ── Drawing canvas ────────────────────────────────────────
const drawingCanvas = document.getElementById('drawingCanvas');
const drawCtx       = drawingCanvas.getContext('2d');

// ── Platform Detection ────────────────────────────────────
const UA        = navigator.userAgent || '';
const isIOS     = /iPhone|iPad|iPod/.test(UA) && !window.MSStream;
const isAndroid = /Android/i.test(UA);
const isMobile  = isIOS || isAndroid || window.matchMedia('(pointer: coarse)').matches;
document.body.dataset.platform = isIOS ? 'ios' : isAndroid ? 'android' : 'desktop';

// Show flip button on mobile
if (isMobile) flipCameraBtn.style.display = 'inline-flex';

// ── State ─────────────────────────────────────────────────
let showHands      = true;
let showFace       = true;
let drawMode       = false;
let drawColor      = '#ff6584';
let lastDrawPoint  = null;
let currentStream  = null;
let holisticModel  = null;
let modelReady     = false;
let currentFacing  = 'user';   // 'user' | 'environment'
let frameLoopToken = 0;        // incremented on each startCamera; old loops self-cancel

// ── FPS tracking ──────────────────────────────────────────
let fpsFrameCount = 0;
let fpsLastTime   = performance.now();

// ── Drawing state ─────────────────────────────────────────
let prevDrawPoint  = null; // second-to-last point for bezier smoothing

// ══════════════════════════════════════════════════════════
//  HAND GESTURE CLASSIFICATION
// ══════════════════════════════════════════════════════════
const LM = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

// Pre-allocated constant — avoids a new array every frame in drawHandSkeleton
const FINGER_TIPS = [LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP];

function isFingerExtended(lm, tipIdx, pipIdx) {
  return lm[tipIdx].y < lm[pipIdx].y;
}

function isThumbExtended(lm) {
  const tip = lm[LM.THUMB_TIP];
  const mcp = lm[LM.THUMB_MCP];
  const ip  = lm[LM.THUMB_IP];
  return Math.abs(tip.x - mcp.x) > 0.06 || tip.y < ip.y - 0.02;
}

function classifyGesture(lm) {
  const thumb  = isThumbExtended(lm);
  const index  = isFingerExtended(lm, LM.INDEX_TIP,  LM.INDEX_PIP);
  const middle = isFingerExtended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP);
  const ring   = isFingerExtended(lm, LM.RING_TIP,   LM.RING_PIP);
  const pinky  = isFingerExtended(lm, LM.PINKY_TIP,  LM.PINKY_PIP);
  const extCount = (index ? 1 : 0) + (middle ? 1 : 0) + (ring ? 1 : 0) + (pinky ? 1 : 0);

  if (thumb && !index && !middle && !ring && !pinky) return '👍 Thumbs Up';
  if (extCount >= 4)                                  return '✋ Open Hand';
  if (index && middle && !ring && !pinky)             return '✌️ Peace';
  if (index && !middle && !ring && !pinky)            return '☝️ Pointing';
  if (extCount === 0 && !thumb)                       return '✊ Fist';
  if (thumb && !index && !middle && !ring && pinky)   return '🤙 Call Me';
  if (index && !middle && !ring && pinky)             return '🤘 Rock';
  if (index && middle && ring && !pinky)              return '3️⃣ Three';
  return '🖐 Hand';
}

/** True only when index is extended AND middle/ring/pinky are all curled. */
function isPointing(lm) {
  const index  = isFingerExtended(lm, LM.INDEX_TIP,  LM.INDEX_PIP);
  const middle = isFingerExtended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP);
  const ring   = isFingerExtended(lm, LM.RING_TIP,   LM.RING_PIP);
  const pinky  = isFingerExtended(lm, LM.PINKY_TIP,  LM.PINKY_PIP);
  return index && !middle && !ring && !pinky;
}

// ── Top-level rendering helpers ───────────────────────────
// Defined here (not inside onResults) so no new function objects are
// allocated on every frame.

function drawHandSkeleton(landmarks, w, h) {
  drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#6c63ff', lineWidth: 2 });
  drawLandmarks(ctx, landmarks, { color: '#00d4ff', fillColor: '#00d4ff', radius: 4, lineWidth: 1 });
  ctx.fillStyle = '#ff6584';
  for (let i = 0; i < FINGER_TIPS.length; i++) {
    const lm = landmarks[FINGER_TIPS[i]];
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 6, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function updateHandPanel(landmarks, w, h, label, xEl, yEl, pxEl, pyEl) {
  label.textContent = classifyGesture(landmarks);
  const wrist = landmarks[LM.WRIST];
  xEl.textContent  = wrist.x.toFixed(3);
  yEl.textContent  = wrist.y.toFixed(3);
  pxEl.textContent = Math.round(wrist.x * w) + 'px';
  pyEl.textContent = Math.round(wrist.y * h) + 'px';
}

// ══════════════════════════════════════════════════════════
//  FACE EXPRESSION CLASSIFICATION
// ══════════════════════════════════════════════════════════
// Key MediaPipe Face Mesh landmark indices
const FLM = {
  MOUTH_LEFT:  61,
  MOUTH_RIGHT: 291,
  UPPER_LIP:   13,
  LOWER_LIP:   14,
  // 6-point eye landmarks for Eye Aspect Ratio (EAR)
  L_EYE: [33, 160, 158, 133, 153, 144],   // outer, top1, top2, inner, bot1, bot2
  R_EYE: [362, 385, 387, 263, 373, 380],
  // Inner eyebrow corners
  L_BROW_INNER: 70,
  R_BROW_INNER: 300,
  // Inner eye corners (reference for brow height)
  L_EYE_INNER:  133,
  R_EYE_INNER:  362,
};

/** Euclidean distance between two {x, y} normalized landmarks */
function lmDist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Eye Aspect Ratio — < 0.20 = closed
 * EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
 * indices: [outer, top1, top2, inner, bot1, bot2]
 */
function eyeAspectRatio(lm, idx) {
  // Direct index access — avoids allocating a new array via .map() every frame
  const p1 = lm[idx[0]], p2 = lm[idx[1]], p3 = lm[idx[2]];
  const p4 = lm[idx[3]], p5 = lm[idx[4]], p6 = lm[idx[5]];
  return (lmDist(p2, p6) + lmDist(p3, p5)) / (2 * lmDist(p1, p4));
}

/**
 * Classify face expression from 468 face mesh landmarks.
 * Returns { label, mouth, leftEye, rightEye, brows }
 */
function classifyExpression(lm) {
  // Mouth open: gap between upper and lower lip
  const mouthOpen = lmDist(lm[FLM.UPPER_LIP], lm[FLM.LOWER_LIP]) > 0.04;

  // Smile / frown: compare mouth corners Y to lip centre Y
  const lipCentreY = (lm[FLM.UPPER_LIP].y + lm[FLM.LOWER_LIP].y) / 2;
  const cornerY    = (lm[FLM.MOUTH_LEFT].y + lm[FLM.MOUTH_RIGHT].y) / 2;
  const smiling    = cornerY < lipCentreY - 0.008;
  const frowning   = cornerY > lipCentreY + 0.008;

  // Eyes closed via EAR
  const leftEAR  = eyeAspectRatio(lm, FLM.L_EYE);
  const rightEAR = eyeAspectRatio(lm, FLM.R_EYE);
  const leftClosed  = leftEAR  < 0.20;
  const rightClosed = rightEAR < 0.20;
  const bothClosed  = leftClosed && rightClosed;

  // Eyebrows raised: brow inner Y is well above eye inner corner Y
  const lBrowDist = lm[FLM.L_EYE_INNER].y - lm[FLM.L_BROW_INNER].y;
  const rBrowDist = lm[FLM.R_EYE_INNER].y - lm[FLM.R_BROW_INNER].y;
  const browsRaised = lBrowDist > 0.065 && rBrowDist > 0.065;

  // Compose label (priority order)
  let label;
  if (bothClosed)        label = '😌 Eyes Closed';
  else if (mouthOpen && browsRaised) label = '😮 Surprised';
  else if (mouthOpen)    label = '😮 Mouth Open';
  else if (smiling)      label = '😊 Smiling';
  else if (frowning)     label = '😞 Frowning';
  else if (browsRaised)  label = '🤨 Brows Raised';
  else                   label = '😐 Neutral';

  return {
    label,
    mouth:    mouthOpen  ? 'Open'   : 'Closed',
    leftEye:  leftClosed ? 'Closed' : 'Open',
    rightEye: rightClosed? 'Closed' : 'Open',
    brows:    browsRaised? 'Raised' : 'Normal',
  };
}

// ══════════════════════════════════════════════════════════
//  CANVAS
// ══════════════════════════════════════════════════════════
function resizeCanvas(w, h) {
  canvas.width         = w;
  canvas.height        = h;
  drawingCanvas.width  = w;
  drawingCanvas.height = h;
}

// ══════════════════════════════════════════════════════════
//  onResults — called every frame by MediaPipe Holistic
// ══════════════════════════════════════════════════════════
function onResults(results) {
  const w = results.image.width;
  const h = results.image.height;

  if (canvas.width !== w || canvas.height !== h) resizeCanvas(w, h);

  // ── FPS ──────────────────────────────────────────────
  fpsFrameCount++;
  const nowMs = performance.now();
  if (nowMs - fpsLastTime >= 500) {
    const fps = Math.round(fpsFrameCount * 1000 / (nowMs - fpsLastTime));
    fpsFrameCount = 0;
    fpsLastTime   = nowMs;
    if (fpsEl) fpsEl.textContent = fps + ' fps';
  }

  // Mirror only when using the front (selfie) camera.
  const mirrored = currentFacing === 'user';
  ctx.clearRect(0, 0, w, h);
  if (mirrored) ctx.setTransform(-1, 0, 0, 1, w, 0);

  // ── Face ─────────────────────────────────────────────
  const hasFace = !!(results.faceLandmarks && results.faceLandmarks.length);
  if (showFace && hasFace) {
    drawConnectors(ctx, results.faceLandmarks, FACEMESH_CONTOURS, { color: '#6c63ff', lineWidth: 2 });
    drawLandmarks(ctx, results.faceLandmarks, { color: '#00d4ff', fillColor: '#00d4ff', radius: 1.5, lineWidth: 0 });
    const expr = classifyExpression(results.faceLandmarks);
    facePanel.style.display     = 'block';
    expressionLabel.textContent = expr.label;
    mouthStateEl.textContent    = expr.mouth;
    leftEyeStateEl.textContent  = expr.leftEye;
    rightEyeStateEl.textContent = expr.rightEye;
    browsStateEl.textContent    = expr.brows;
  } else {
    facePanel.style.display = 'none';
  }

  // ── Hands ────────────────────────────────────────────
  const leftLM  = results.leftHandLandmarks;
  const rightLM = results.rightHandLandmarks;
  const handCount = (leftLM ? 1 : 0) + (rightLM ? 1 : 0);

  handCountPanel.style.display = 'block';
  handCountEl.textContent      = handCount;
  faceDetectedEl.textContent   = hasFace ? 'Yes' : 'No';

  if (showHands && handCount > 0) {
    infoPanel.style.display = 'block';
    if (leftLM) {
      drawHandSkeleton(leftLM, w, h);
      updateHandPanel(leftLM, w, h, gestureLabel, coordX, coordY, pixelX, pixelY);
    } else {
      gestureLabel.textContent = '—';
      coordX.textContent = coordY.textContent = pixelX.textContent = pixelY.textContent = '—';
    }
    if (rightLM) {
      secondHandDiv.classList.add('visible');
      drawHandSkeleton(rightLM, w, h);
      updateHandPanel(rightLM, w, h, gestureLabel2, coordX2, coordY2, pixelX2, pixelY2);
    } else {
      secondHandDiv.classList.remove('visible');
    }
  } else {
    infoPanel.style.display = 'none';
  }

  // ── Air Drawing ───────────────────────────────────────
  if (drawMode) {
    // Set stroke properties once per frame, outside the per-point logic
    drawCtx.strokeStyle = drawColor;
    drawCtx.lineWidth   = 5;
    drawCtx.lineCap     = 'round';
    drawCtx.lineJoin    = 'round';

    const penHand = leftLM || rightLM;
    if (penHand && isPointing(penHand)) {
      const tip   = penHand[LM.INDEX_TIP];
      const rawX  = tip.x * w;                        // ctx coords
      const rawY  = tip.y * h;
      const drawX = mirrored ? (w - rawX) : rawX;    // drawCtx coords (no transform)

      if (lastDrawPoint) {
        drawCtx.beginPath();
        if (prevDrawPoint) {
          // Midpoint bezier — smooth curves through consecutive points
          drawCtx.moveTo(
            (prevDrawPoint.x + lastDrawPoint.x) / 2,
            (prevDrawPoint.y + lastDrawPoint.y) / 2
          );
          drawCtx.quadraticCurveTo(
            lastDrawPoint.x, lastDrawPoint.y,
            (lastDrawPoint.x + drawX) / 2,
            (lastDrawPoint.y + rawY) / 2
          );
        } else {
          drawCtx.moveTo(lastDrawPoint.x, lastDrawPoint.y);
          drawCtx.lineTo(drawX, rawY);
        }
        drawCtx.stroke();
      }

      prevDrawPoint = lastDrawPoint;
      lastDrawPoint = { x: drawX, y: rawY };

      // Active cursor on overlay canvas (mirror transform already applied)
      ctx.beginPath();
      ctx.arc(rawX, rawY, 10, 0, 2 * Math.PI);
      ctx.strokeStyle = drawColor;
      ctx.lineWidth   = 3;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(rawX, rawY, 3, 0, 2 * Math.PI);
      ctx.fillStyle = drawColor;
      ctx.fill();
    } else {
      prevDrawPoint = null;
      lastDrawPoint = null;
      // Ghost cursor when hand visible but not pointing
      if (penHand) {
        const tip = penHand[LM.INDEX_TIP];
        ctx.beginPath();
        ctx.arc(tip.x * w, tip.y * h, 7, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth   = 2;
        ctx.stroke();
      }
    }
  }

  // Reset canvas transform to identity for next frame (only needed when mirror was applied)
  if (mirrored) ctx.setTransform(1, 0, 0, 1, 0, 0);
}

// ══════════════════════════════════════════════════════════
//  MODEL INIT
// ══════════════════════════════════════════════════════════
function initModel() {
  holisticModel = new Holistic({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });

  holisticModel.setOptions({
    modelComplexity: 0,          // 0 = lite — fastest, works on mobile
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });

  holisticModel.onResults(onResults);
}

// ══════════════════════════════════════════════════════════
//  CAMERA
// ══════════════════════════════════════════════════════════
async function startCamera(deviceId) {
  // Give this invocation a unique token; any older rAF loop that is still
  // awaiting holisticModel.send() will see a mismatched token and stop.
  const myToken = ++frameLoopToken;

  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }

  setStatus('loading', 'Starting camera…');
  loadingText.textContent = 'Starting camera…';

  const constraints = {
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width:    { ideal: EXPORT_W },
      height:   { ideal: EXPORT_H },
      // facingMode only when no specific deviceId (used by flip button)
      facingMode: deviceId ? undefined : { ideal: currentFacing },
    },
    audio: false,
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = currentStream;
    await videoEl.play();
  } catch (err) {
    console.error('Camera error:', err);
    setStatus('error', 'Camera access denied');
    noCameraMsg.style.display = 'flex';
    loadingOverlay.style.display = 'none';
    return;
  }

  // Mirror CSS only for the front/selfie camera
  videoEl.classList.toggle('mirrored', currentFacing === 'user');

  // Downsample each video frame into a small canvas before sending to MediaPipe.
  // iOS Safari cannot feed HTMLVideoElement pixels directly into the WebGL pipeline;
  // an intermediate 2D canvas is required for cross-platform compatibility.
  const trackCanvas = document.createElement('canvas');
  trackCanvas.width  = TRACK_W;
  trackCanvas.height = TRACK_H;
  const trackCtx = trackCanvas.getContext('2d');
  let shownActive = false;

  loadingText.textContent = 'Loading tracking model…';

  const loop = async () => {
    // Stop if a newer startCamera call has taken over
    if (frameLoopToken !== myToken) return;

    if (holisticModel && videoEl.readyState >= 2) {
      try {
        trackCtx.drawImage(videoEl, 0, 0, TRACK_W, TRACK_H);
        await holisticModel.send({ image: trackCanvas });
        if (!shownActive) {
          shownActive = true;
          loadingOverlay.style.display = 'none';
          setStatus('active', 'Tracking active');
          modelReady = true;
        }
      } catch (err) {
        console.warn('MediaPipe frame error (will retry):', err);
      }
    }

    // Re-check token after the await before scheduling next frame
    if (frameLoopToken === myToken) requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

async function populateCameras() {
  try {
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tempStream.getTracks().forEach(t => t.stop());
  } catch { /* permission denied, labels will be empty */ }

  let devices;
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (err) {
    console.error('enumerateDevices error:', err);
    return;
  }

  const videoDevices = devices.filter(d => d.kind === 'videoinput');
  cameraSelect.innerHTML = '';

  if (videoDevices.length === 0) {
    noCameraMsg.style.display = 'flex';
    loadingOverlay.style.display = 'none';
    return;
  }

  videoDevices.forEach((device, i) => {
    const opt = document.createElement('option');
    opt.value = device.deviceId;
    opt.textContent = device.label || `Camera ${i + 1}`;
    cameraSelect.appendChild(opt);
  });
}

// ══════════════════════════════════════════════════════════
//  HELPERS & EVENT LISTENERS
// ══════════════════════════════════════════════════════════
function setStatus(state, text) {
  statusDot.className = 'dot';
  if (state === 'active')  statusDot.classList.add('active');
  if (state === 'loading') statusDot.classList.add('loading');
  statusText.textContent = text;
}

cameraSelect.addEventListener('change', () => {
  modelReady = false;
  loadingOverlay.style.display = 'flex';
  infoPanel.style.display = 'none';
  facePanel.style.display = 'none';
  handCountPanel.style.display = 'none';
  startCamera(cameraSelect.value);
});

toggleHands.addEventListener('click', () => {
  showHands = !showHands;
  toggleHands.classList.toggle('active', showHands);
  if (!showHands) infoPanel.style.display = 'none';
});

toggleFace.addEventListener('click', () => {
  showFace = !showFace;
  toggleFace.classList.toggle('active', showFace);
  if (!showFace) facePanel.style.display = 'none';
});

toggleDrawBtn.addEventListener('click', () => {
  drawMode = !drawMode;
  toggleDrawBtn.classList.toggle('active', drawMode);
  drawToolbar.style.display = drawMode ? 'flex' : 'none';
  if (!drawMode) { lastDrawPoint = null; prevDrawPoint = null; }
});

clearDrawBtn.addEventListener('click', () => {
  drawCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
});

colorSwatches.forEach(swatch => {
  swatch.addEventListener('click', () => {
    colorSwatches.forEach(s => s.classList.remove('active'));
    swatch.classList.add('active');
    drawColor = swatch.dataset.color;
  });
});

screenshotBtn.addEventListener('click', () => {
  const tmp  = document.createElement('canvas');
  tmp.width  = EXPORT_W;
  tmp.height = EXPORT_H;
  const tCtx = tmp.getContext('2d');
  tCtx.imageSmoothingEnabled = true;
  tCtx.imageSmoothingQuality = 'high';

  // 1. Full-res video — mirror only for front camera (CSS transform doesn't carry into drawImage)
  tCtx.save();
  if (currentFacing === 'user') {
    tCtx.translate(EXPORT_W, 0);
    tCtx.scale(-1, 1);
  }
  tCtx.drawImage(videoEl, 0, 0, EXPORT_W, EXPORT_H);
  tCtx.restore();
  // 2. Skeleton overlay (already mirrored in its pixel data, scale 2×)
  tCtx.drawImage(canvas, 0, 0, EXPORT_W, EXPORT_H);
  // 3. Drawing strokes (pre-mirrored coords, scale 2×)
  tCtx.drawImage(drawingCanvas, 0, 0, EXPORT_W, EXPORT_H);

  const filename = `handtrack-${Date.now()}.png`;

  tmp.toBlob(async (blob) => {
    // iOS: use Web Share API — triggers the native share sheet which
    // includes "Save to Photos" as an option (requires iOS 15+ / Safari 15+)
    if (isIOS && navigator.canShare) {
      const file = new File([blob], filename, { type: 'image/png' });
      if (navigator.canShare({ files: [file] })) {
        try {
          await navigator.share({ files: [file], title: 'HandTrack snapshot' });
        } catch (err) {
          if (err.name !== 'AbortError') console.error('Share failed:', err);
        }
        return;
      }
    }

    // Desktop / Android: standard blob URL download
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = filename;
    link.href     = url;
    link.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
});

// ── Camera flip (mobile only) ──────────────────────────
flipCameraBtn.addEventListener('click', () => {
  currentFacing = currentFacing === 'user' ? 'environment' : 'user';
  modelReady = false;
  loadingOverlay.style.display = 'flex';
  startCamera(''); // restart with new facing
});

// ── Fullscreen ─────────────────────────────────────────
function setFullscreenLabel(active) {
  fullscreenBtn.textContent = active ? '✕' : '⛶';
  fullscreenBtn.title       = active ? 'Exit fullscreen' : 'Fullscreen';
  fullscreenBtn.classList.toggle('active', active);
}

fullscreenBtn.addEventListener('click', async () => {
  const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
  try {
    if (!fsEl) {
      const el = document.documentElement;
      if (el.requestFullscreen)            await el.requestFullscreen();
      else if (el.webkitRequestFullscreen) await el.webkitRequestFullscreen();
    } else {
      if (document.exitFullscreen)            await document.exitFullscreen();
      else if (document.webkitExitFullscreen) await document.webkitExitFullscreen();
    }
  } catch (err) {
    console.log('Fullscreen unavailable:', err.message);
  }
});

document.addEventListener('fullscreenchange',        () => setFullscreenLabel(!!document.fullscreenElement));
document.addEventListener('webkitfullscreenchange',  () => setFullscreenLabel(!!document.webkitFullscreenElement));

// ══════════════════════════════════════════════════════════
//  MAIN INIT
// ══════════════════════════════════════════════════════════
(async function init() {
  setStatus('loading', 'Initializing…');
  initModel();
  await populateCameras();

  const firstDevice = cameraSelect.options[0]?.value || '';
  if (cameraSelect.options.length > 0) {
    await startCamera(firstDevice);
  }
})();
