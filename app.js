/* ══════════════════════════════════════════════════════════
   HandTrack — app.js
   MediaPipe Hands · gesture classification · canvas overlay
══════════════════════════════════════════════════════════ */

'use strict';

// ── DOM refs ─────────────────────────────────────────────
const videoEl        = document.getElementById('videoEl');
const canvas         = document.getElementById('overlayCanvas');
const ctx            = canvas.getContext('2d');
const cameraSelect   = document.getElementById('cameraSelect');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText    = document.getElementById('loadingText');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const infoPanel      = document.getElementById('infoPanel');
const handCountPanel = document.getElementById('handCountPanel');
const noCameraMsg    = document.getElementById('noCameraMsg');
const secondHandDiv  = document.getElementById('secondHand');
const viewport       = document.getElementById('viewport');

// Hand 1 UI
const gestureLabel = document.getElementById('gestureLabel');
const coordX       = document.getElementById('coordX');
const coordY       = document.getElementById('coordY');
const pixelX       = document.getElementById('pixelX');
const pixelY       = document.getElementById('pixelY');

// Hand 2 UI
const gestureLabel2 = document.getElementById('gestureLabel2');
const coordX2       = document.getElementById('coordX2');
const coordY2       = document.getElementById('coordY2');
const pixelX2       = document.getElementById('pixelX2');
const pixelY2       = document.getElementById('pixelY2');

const handCount = document.getElementById('handCount');

// ── State ─────────────────────────────────────────────────
let currentStream = null;
let mediapipeCamera = null;
let handsModel = null;
let modelReady = false;

// ── Landmark indices ───────────────────────────────────────
const LM = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

// ── Gesture classification ─────────────────────────────────
/**
 * Returns true if the given finger tip is extended above its PIP joint.
 * MediaPipe Y coords: 0 = top of image, 1 = bottom.
 * So "extended" means tip.y < pip.y (closer to top).
 */
function isFingerExtended(lm, tipIdx, pipIdx) {
  return lm[tipIdx].y < lm[pipIdx].y;
}

/**
 * Thumb check: compare tip X to MCP X.
 * For right hand (mirrored): tip should be further left (smaller x).
 * We use a threshold-based approach that works for both hands.
 */
function isThumbExtended(lm) {
  const tip = lm[LM.THUMB_TIP];
  const mcp = lm[LM.THUMB_MCP];
  const ip  = lm[LM.THUMB_IP];
  // Extended if tip is clearly to the side of the MCP
  return Math.abs(tip.x - mcp.x) > 0.06 || tip.y < ip.y - 0.02;
}

function classifyGesture(lm) {
  const thumb  = isThumbExtended(lm);
  const index  = isFingerExtended(lm, LM.INDEX_TIP,  LM.INDEX_PIP);
  const middle = isFingerExtended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP);
  const ring   = isFingerExtended(lm, LM.RING_TIP,   LM.RING_PIP);
  const pinky  = isFingerExtended(lm, LM.PINKY_TIP,  LM.PINKY_PIP);

  const extCount = [index, middle, ring, pinky].filter(Boolean).length;

  // Thumbs Up: thumb extended, all other fingers curled
  if (thumb && !index && !middle && !ring && !pinky) return '👍 Thumbs Up';

  // Open Hand: most or all fingers extended
  if (extCount >= 4) return '✋ Open Hand';

  // Peace: index + middle only
  if (index && middle && !ring && !pinky) return '✌️ Peace';

  // Pointing: only index
  if (index && !middle && !ring && !pinky) return '☝️ Pointing';

  // Fist: no fingers extended
  if (extCount === 0 && !thumb) return '✊ Fist';

  // Call Me: thumb + pinky
  if (thumb && !index && !middle && !ring && pinky) return '🤙 Call Me';

  // Rock: index + pinky (horns)
  if (index && !middle && !ring && pinky) return '🤘 Rock';

  // Three fingers
  if (index && middle && ring && !pinky) return '3️⃣ Three';

  // Default
  return '🖐 Hand';
}

// ── Resize canvas to match video dimensions ───────────────
function resizeCanvas(w, h) {
  canvas.width  = w;
  canvas.height = h;
}

// ── Draw results from MediaPipe ───────────────────────────
function onResults(results) {
  const w = results.image.width;
  const h = results.image.height;

  if (canvas.width !== w || canvas.height !== h) resizeCanvas(w, h);

  ctx.save();
  ctx.clearRect(0, 0, w, h);

  // Draw the camera frame
  ctx.drawImage(results.image, 0, 0, w, h);

  const detected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

  if (detected) {
    infoPanel.style.display = 'block';
    handCountPanel.style.display = 'block';
    handCount.textContent = results.multiHandLandmarks.length;

    results.multiHandLandmarks.forEach((landmarks, i) => {
      // Draw skeleton connections
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
        color: '#6c63ff',
        lineWidth: 2,
      });

      // Draw landmark dots
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

      // Classify gesture
      const gesture = classifyGesture(landmarks);
      const wrist   = landmarks[LM.WRIST];
      const px = Math.round(wrist.x * w);
      const py = Math.round(wrist.y * h);

      if (i === 0) {
        gestureLabel.textContent = gesture;
        coordX.textContent  = wrist.x.toFixed(3);
        coordY.textContent  = wrist.y.toFixed(3);
        pixelX.textContent  = px + 'px';
        pixelY.textContent  = py + 'px';
      } else if (i === 1) {
        secondHandDiv.classList.add('visible');
        gestureLabel2.textContent = gesture;
        coordX2.textContent  = wrist.x.toFixed(3);
        coordY2.textContent  = wrist.y.toFixed(3);
        pixelX2.textContent  = px + 'px';
        pixelY2.textContent  = py + 'px';
      }
    });

    // Hide second hand panel if only one hand
    if (results.multiHandLandmarks.length < 2) {
      secondHandDiv.classList.remove('visible');
    }

  } else {
    // No hands detected
    infoPanel.style.display = 'none';
    handCountPanel.style.display = 'block';
    handCount.textContent = '0';
  }

  ctx.restore();
}

// ── Init MediaPipe Hands ──────────────────────────────────
function initModel() {
  handsModel = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  handsModel.setOptions({
    maxNumHands: 2,
    modelComplexity: 0,   // 0 = lite (faster), 1 = full
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });

  handsModel.onResults(onResults);

  return handsModel;
}

// ── Start camera stream ───────────────────────────────────
async function startCamera(deviceId) {
  // Stop existing stream
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
      width: { ideal: 1280 },
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

  // Use MediaPipe Camera utility to feed frames into the model
  mediapipeCamera = new Camera(videoEl, {
    onFrame: async () => {
      if (handsModel) await handsModel.send({ image: videoEl });
    },
    width: 1280,
    height: 720,
  });

  loadingText.textContent = 'Loading hand tracking model…';

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

// ── Enumerate cameras ─────────────────────────────────────
async function populateCameras() {
  try {
    // Request permission first so labels are returned
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tempStream.getTracks().forEach(t => t.stop());
  } catch {
    // Permission denied — carry on; labels will be empty
  }

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

// ── Status helper ─────────────────────────────────────────
function setStatus(state, text) {
  statusDot.className = 'dot';
  if (state === 'active')  statusDot.classList.add('active');
  if (state === 'loading') statusDot.classList.add('loading');
  statusText.textContent = text;
}

// ── Camera select change ──────────────────────────────────
cameraSelect.addEventListener('change', () => {
  modelReady = false;
  loadingOverlay.style.display = 'flex';
  infoPanel.style.display = 'none';
  handCountPanel.style.display = 'none';
  startCamera(cameraSelect.value);
});

// ── Canvas sizing on window resize ───────────────────────
window.addEventListener('resize', () => {
  // Canvas auto-sizes via CSS; just trigger a redraw next frame
});

// ── Main init ─────────────────────────────────────────────
(async function init() {
  setStatus('loading', 'Initializing…');
  initModel();
  await populateCameras();

  const firstDevice = cameraSelect.options[0]?.value || '';
  if (firstDevice || cameraSelect.options.length > 0) {
    await startCamera(firstDevice);
  }
})();
