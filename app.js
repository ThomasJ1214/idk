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

// ── DOM refs — hands ──────────────────────────────────────
const infoPanel      = document.getElementById('infoPanel');
const handCountPanel = document.getElementById('handCountPanel');
const secondHandDiv  = document.getElementById('secondHand');
const handCountEl    = document.getElementById('handCount');
const faceDetectedEl = document.getElementById('faceDetected');

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

// ── Drawing canvas ────────────────────────────────────────
const drawingCanvas = document.getElementById('drawingCanvas');
const drawCtx       = drawingCanvas.getContext('2d');

// ── State ─────────────────────────────────────────────────
let showHands       = true;
let showFace        = true;
let drawMode        = false;
let drawColor       = '#ff6584';
let lastDrawPoint   = null;
let currentStream   = null;
let mediapipeCamera = null;
let holisticModel   = null;
let modelReady      = false;

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
  const extCount = [index, middle, ring, pinky].filter(Boolean).length;

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
function eyeAspectRatio(lm, indices) {
  const [p1, p2, p3, p4, p5, p6] = indices.map(i => lm[i]);
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

  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(results.image, 0, 0, w, h);

  // ── Face overlay ─────────────────────────────────────
  const hasFace = !!(results.faceLandmarks && results.faceLandmarks.length > 0);

  if (showFace && hasFace) {
    // Draw face contours — same colour/weight as hand skeleton
    drawConnectors(ctx, results.faceLandmarks, FACEMESH_CONTOURS, {
      color: '#6c63ff',
      lineWidth: 2,
    });
    drawLandmarks(ctx, results.faceLandmarks, {
      color: '#00d4ff',
      fillColor: '#00d4ff',
      radius: 1.5,
      lineWidth: 0,
    });

    // Classify and display expression
    const expr = classifyExpression(results.faceLandmarks);
    facePanel.style.display       = 'block';
    expressionLabel.textContent   = expr.label;
    mouthStateEl.textContent      = expr.mouth;
    leftEyeStateEl.textContent    = expr.leftEye;
    rightEyeStateEl.textContent   = expr.rightEye;
    browsStateEl.textContent      = expr.brows;
  } else {
    facePanel.style.display = 'none';
  }

  // ── Hand overlays ─────────────────────────────────────
  const leftLM  = results.leftHandLandmarks;   // person's left hand
  const rightLM = results.rightHandLandmarks;  // person's right hand
  const handDetectedCount = (leftLM ? 1 : 0) + (rightLM ? 1 : 0);

  handCountPanel.style.display = 'block';
  handCountEl.textContent      = handDetectedCount;
  faceDetectedEl.textContent   = hasFace ? 'Yes' : 'No';

  if (showHands && handDetectedCount > 0) {
    infoPanel.style.display = 'block';

    const drawHand = (landmarks) => {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
        color: '#6c63ff',
        lineWidth: 2,
      });
      drawLandmarks(ctx, landmarks, {
        color: '#00d4ff',
        fillColor: '#00d4ff',
        radius: 4,
        lineWidth: 1,
      });
      // Highlight fingertips
      const tips = [LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP];
      tips.forEach(idx => {
        const lm = landmarks[idx];
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#ff6584';
        ctx.fill();
      });
    };

    const updateHandUI = (landmarks, label, coordXEl, coordYEl, pixelXEl, pixelYEl) => {
      const gesture = classifyGesture(landmarks);
      const wrist   = landmarks[LM.WRIST];
      const px = Math.round(wrist.x * w);
      const py = Math.round(wrist.y * h);
      label.textContent    = gesture;
      coordXEl.textContent = wrist.x.toFixed(3);
      coordYEl.textContent = wrist.y.toFixed(3);
      pixelXEl.textContent = px + 'px';
      pixelYEl.textContent = py + 'px';
    };

    if (leftLM) {
      drawHand(leftLM);
      updateHandUI(leftLM, gestureLabel, coordX, coordY, pixelX, pixelY);
    } else {
      gestureLabel.textContent = '—';
      coordX.textContent = coordY.textContent = pixelX.textContent = pixelY.textContent = '—';
    }

    if (rightLM) {
      secondHandDiv.classList.add('visible');
      drawHand(rightLM);
      updateHandUI(rightLM, gestureLabel2, coordX2, coordY2, pixelX2, pixelY2);
    } else {
      secondHandDiv.classList.remove('visible');
    }

  } else {
    infoPanel.style.display = 'none';
  }

  // ── Air Drawing ───────────────────────────────────────
  if (drawMode) {
    const penHand = leftLM || rightLM;
    if (penHand) {
      const indexUp = isFingerExtended(penHand, LM.INDEX_TIP, LM.INDEX_PIP);
      const tip = penHand[LM.INDEX_TIP];
      const px = tip.x * w;
      const py = tip.y * h;

      if (indexUp) {
        // Draw stroke on persistent drawing canvas
        if (lastDrawPoint) {
          drawCtx.beginPath();
          drawCtx.moveTo(lastDrawPoint.x, lastDrawPoint.y);
          drawCtx.lineTo(px, py);
          drawCtx.strokeStyle = drawColor;
          drawCtx.lineWidth   = 5;
          drawCtx.lineCap     = 'round';
          drawCtx.lineJoin    = 'round';
          drawCtx.stroke();
        }
        lastDrawPoint = { x: px, y: py };

        // Cursor ring while drawing
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, 2 * Math.PI);
        ctx.strokeStyle = drawColor;
        ctx.lineWidth   = 3;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, 2 * Math.PI);
        ctx.fillStyle = drawColor;
        ctx.fill();
      } else {
        lastDrawPoint = null;
        // Dim cursor when pen is "up" (other fingers down)
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth   = 2;
        ctx.stroke();
      }
    } else {
      lastDrawPoint = null; // no hand visible
    }
  }

  ctx.restore();
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
    modelComplexity: 0,          // 0 = lite, fastest
    smoothLandmarks: true,
    enableSegmentation: false,   // skip segmentation mask for performance
    smoothSegmentation: false,
    refineFaceLandmarks: false,  // keep face lite for performance
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });

  holisticModel.onResults(onResults);
}

// ══════════════════════════════════════════════════════════
//  CAMERA
// ══════════════════════════════════════════════════════════
async function startCamera(deviceId) {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  if (mediapipeCamera) {
    mediapipeCamera.stop();
    mediapipeCamera = null;
  }

  setStatus('loading', 'Starting camera…');

  const constraints = {
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width:  { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: deviceId ? undefined : 'user',
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

  mediapipeCamera = new Camera(videoEl, {
    onFrame: async () => {
      if (holisticModel) await holisticModel.send({ image: videoEl });
    },
    width: 1280,
    height: 720,
  });

  loadingText.textContent = 'Loading tracking model…';

  mediapipeCamera.start().then(() => {
    loadingOverlay.style.display = 'none';
    setStatus('active', 'Tracking active');
    modelReady = true;
  }).catch(err => {
    console.error('MediaPipe camera error:', err);
    setStatus('error', 'Tracking failed to start');
    loadingOverlay.style.display = 'none';
  });
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
  if (!drawMode) lastDrawPoint = null;
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
  const tmp    = document.createElement('canvas');
  tmp.width    = canvas.width;
  tmp.height   = canvas.height;
  const tCtx   = tmp.getContext('2d');
  tCtx.drawImage(canvas, 0, 0);
  tCtx.drawImage(drawingCanvas, 0, 0);
  const link   = document.createElement('a');
  link.download = `handtrack-${Date.now()}.png`;
  link.href     = tmp.toDataURL('image/png');
  link.click();
});

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
