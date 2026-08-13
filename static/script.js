let selectedFile = null;
let cameraStream = null;

const deviceInput = document.getElementById("deviceInput");
const uploadBtn = document.getElementById("uploadBtn");
const cameraBtn = document.getElementById("cameraBtn");
const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("resultBox");
const diseaseName = document.getElementById("diseaseName");
const confidence = document.getElementById("confidence");
const resetBtn = document.getElementById("resetBtn");
const card = document.querySelector(".card");
const cameraModal = document.getElementById("cameraModal");
const cameraPreview = document.getElementById("cameraPreview");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraStatus = document.getElementById("cameraStatus");
const takePhotoBtn = document.getElementById("takePhotoBtn");
const cancelCameraBtn = document.getElementById("cancelCameraBtn");

function handleFileSelect(event) {
  const file = event.target.files[0];

  if (!file) return;

  selectedFile = file;

  previewImage.src = URL.createObjectURL(file);
  previewBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
  card.classList.remove("has-result");
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraPreview.srcObject = null;
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Camera access is not available in this browser. Please upload a photo from your device.");
    return;
  }

  cameraModal.classList.remove("hidden");
  cameraStatus.textContent = "Requesting camera permission...";
  takePhotoBtn.disabled = true;

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    cameraStatus.textContent = "Position the affected area in the frame, then take the photo.";
    takePhotoBtn.disabled = false;
  } catch (error) {
    stopCamera();
    cameraModal.classList.add("hidden");
    const message = error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access for this site and try again."
      : "Unable to open the camera. Check that it is connected and not being used by another app.";
    alert(message);
    console.error("Camera error:", error);
  }
}

function closeCamera() {
  stopCamera();
  cameraModal.classList.add("hidden");
  cameraStatus.textContent = "";
}

function takePhoto() {
  const width = cameraPreview.videoWidth;
  const height = cameraPreview.videoHeight;

  if (!width || !height) {
    cameraStatus.textContent = "The camera is still starting. Please try again in a moment.";
    return;
  }

  cameraCanvas.width = width;
  cameraCanvas.height = height;
  cameraCanvas.getContext("2d").drawImage(cameraPreview, 0, 0, width, height);
  cameraCanvas.toBlob((blob) => {
    if (!blob) return;
    selectedFile = new File([blob], "camera-photo.jpg", { type: "image/jpeg" });
    previewImage.src = URL.createObjectURL(selectedFile);
    previewBox.classList.remove("hidden");
    resultBox.classList.add("hidden");
    card.classList.remove("has-result");
    closeCamera();
  }, "image/jpeg", 0.92);
}

function resetPrediction() {
  selectedFile = null;
  deviceInput.value = "";
  previewImage.removeAttribute("src");
  previewBox.classList.add("hidden");
  resultBox.classList.add("hidden");
  card.classList.remove("has-result");
  diseaseName.textContent = "";
  confidence.textContent = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

uploadBtn.addEventListener("click", () => deviceInput.click());
cameraBtn.addEventListener("click", openCamera);
deviceInput.addEventListener("change", handleFileSelect);
takePhotoBtn.addEventListener("click", takePhoto);
cancelCameraBtn.addEventListener("click", closeCamera);
resetBtn.addEventListener("click", resetPrediction);
window.addEventListener("beforeunload", stopCamera);

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    alert("Please upload or capture an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

  loader.classList.remove("hidden");
  resultBox.classList.add("hidden");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    diseaseName.textContent = data.disease;
    confidence.textContent = data.confidence;

    resultBox.classList.remove("hidden");
    card.classList.add("has-result");

  } catch (error) {
    alert("Prediction failed. Please try again.");
    console.error(error);
  } finally {
    loader.classList.add("hidden");
  }
});
