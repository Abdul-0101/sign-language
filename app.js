let session;
const outputDiv = document.getElementById("output");

// A-Z label map (index → letter)
const labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split('');

// Load ONNX model
async function loadONNX() {
  session = await ort.InferenceSession.create("model.onnx");
  console.log("✅ ONNX model loaded");
}
loadONNX();

// Setup MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.8,
  minTrackingConfidence: 0.7
});
hands.onResults(onResults);

// Camera setup
const video = document.getElementById("video");
const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({ image: video });
  },
  width: 640,
  height: 480,
});
camera.start();

// Handle results from MediaPipe
async function onResults(results) {
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0];

    let xs = lm.map(p => p.x);
    let ys = lm.map(p => p.y);

    let feat = [];
    for (let i = 0; i < 21; i++) {
      feat.push(lm[i].x - Math.min(...xs));
      feat.push(lm[i].y - Math.min(...ys));
    }

    if (feat.length !== 42) {
      outputDiv.innerText = "⚠ Invalid feature vector";
      return;
    }

    const inputTensor = new ort.Tensor("float32", new Float32Array(feat), [1, 42]);
    const prediction = await session.run({ input: inputTensor });

    // Get output name dynamically
    const outputName = session.outputNames[0];
    const predictionData = prediction[outputName].data;

    console.log("Model output:", predictionData);

    const letterIndex = Math.round(predictionData[0]); // assuming it outputs [index]
    const letter = labels[letterIndex] || "?";
    outputDiv.innerText = `Letter: ${letter}`;
  } else {
    outputDiv.innerText = "Letter: -";
  }
}
