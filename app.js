let session;
const outputDiv = document.getElementById("output");

async function loadONNX() {
  session = await ort.InferenceSession.create("model.onnx");
  console.log("✅ ONNX model loaded");
}
loadONNX();

// MediaPipe Hands setup
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.8,
  minTrackingConfidence: 0.7
});

hands.onResults(async (results) => {
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0];
    let xs = lm.map(p => p.x);
    let ys = lm.map(p => p.y);

    let feat = [];
    for (let i = 0; i < 21; i++) {
      feat.push(lm[i].x - Math.min(...xs));
      feat.push(lm[i].y - Math.min(...ys));
    }

    const inputTensor = new ort.Tensor("float32", new Float32Array(feat), [1, 42]);
    const prediction = await session.run({ input: inputTensor });

    const letter = prediction.output.data[0];
    outputDiv.innerText = `Letter: ${letter}`;
  } else {
    outputDiv.innerText = "Letter: -";
  }
});

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
