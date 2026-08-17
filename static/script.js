let selectedFile = null;
let cameraStream = null;
let prediction = null;
let previewUrl = null;

const $ = (id) => document.getElementById(id);
const deviceInput = $("deviceInput");
const uploadBtn = $("uploadBtn");
const cameraBtn = $("cameraBtn");
const previewBox = $("previewBox");
const previewImage = $("previewImage");
const emptyState = $("emptyState");
const fileName = $("fileName");
const removeImageBtn = $("removeImageBtn");
const predictBtn = $("predictBtn");
const loader = $("loader");
const formMessage = $("formMessage");
const resultSection = $("resultSection");
const diseaseName = $("diseaseName");
const confidence = $("confidence");
const confidenceBar = $("confidenceBar");
const educationDisease = $("educationDisease");
const educationCards = $("educationCards");
const resetBtn = $("resetBtn");
const reportBtn = $("reportBtn");
const cameraModal = $("cameraModal");
const cameraPreview = $("cameraPreview");
const cameraCanvas = $("cameraCanvas");
const cameraStatus = $("cameraStatus");
const takePhotoBtn = $("takePhotoBtn");
const cancelCameraBtn = $("cancelCameraBtn");
const reportModal = $("reportModal");
const reportForm = $("reportForm");
const patientName = $("patientName");
const patientAge = $("patientAge");
const reportMessage = $("reportMessage");
const closeReportBtn = $("closeReportBtn");
const cancelReportBtn = $("cancelReportBtn");

function showMessage(message = "") {
  formMessage.textContent = message;
}

function setPreview(file) {
  if (!file || !file.type.startsWith("image/")) {
    showMessage("Please select a valid image file.");
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewBox.classList.remove("hidden");
  emptyState.classList.add("hidden");
  fileName.textContent = file.name || "Camera photo";
  predictBtn.disabled = false;
  resultSection.classList.add("hidden");
  prediction = null;
  showMessage("");
}

function clearImage() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  prediction = null;
  deviceInput.value = "";
  previewImage.removeAttribute("src");
  previewBox.classList.add("hidden");
  emptyState.classList.remove("hidden");
  resultSection.classList.add("hidden");
  predictBtn.disabled = true;
  showMessage("");
}

function stopCamera() {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
}

function closeCamera() {
  stopCamera();
  cameraModal.classList.add("hidden");
  cameraStatus.textContent = "";
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showMessage("Camera access is not available in this browser. Please upload an image instead.");
    return;
  }
  cameraModal.classList.remove("hidden");
  cameraStatus.textContent = "Requesting camera permission...";
  takePhotoBtn.disabled = true;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    cameraStatus.textContent = "Position the affected area in the frame, then take the photo.";
    takePhotoBtn.disabled = false;
  } catch (error) {
    closeCamera();
    showMessage(error.name === "NotAllowedError" ? "Camera permission was denied. Allow access and try again." : "Unable to open the camera. Check that it is connected and not in use.");
  }
}

function takePhoto() {
  const { videoWidth: width, videoHeight: height } = cameraPreview;
  if (!width || !height) return;
  cameraCanvas.width = width;
  cameraCanvas.height = height;
  cameraCanvas.getContext("2d").drawImage(cameraPreview, 0, 0, width, height);
  cameraCanvas.toBlob((blob) => {
    if (blob) setPreview(new File([blob], "dermavision-camera-photo.jpg", { type: "image/jpeg" }));
    closeCamera();
  }, "image/jpeg", 0.92);
}

function createEducationCards(education) {
  const cards = [
    ["Probable Medical Reasons", "Possible common causes or contributing factors", education.reasons],
    ["Prevention", "General preventive measures", education.prevention],
    ["General Care / Possible Solutions", "Educational guidance only", education.care],
  ];
  educationCards.replaceChildren(...cards.map(([title, subtitle, content]) => {
    const card = document.createElement("article");
    card.className = "education-card";
    card.innerHTML = `<span>${subtitle}</span><h3>${title}</h3><p>${content}</p>`;
    return card;
  }));
}

async function predictDisease() {
  if (!selectedFile) return showMessage("Please select an image before prediction.");
  showMessage("");
  loader.classList.remove("hidden");
  predictBtn.disabled = true;
  const data = new FormData();
  data.append("image", selectedFile);
  try {
    const response = await fetch("/predict", { method: "POST", body: data });
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || "Prediction failed.");
    prediction = result;
    diseaseName.textContent = result.disease;
    confidence.textContent = `${result.confidence}%`;
    confidenceBar.style.width = `${Math.min(100, Math.max(0, Number(result.confidence)))}%`;
    educationDisease.textContent = result.disease;
    createEducationCards(result.education);
    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage(error.message || "Prediction failed. Please try again.");
  } finally {
    loader.classList.add("hidden");
    predictBtn.disabled = !selectedFile;
  }
}

function openReport() {
  if (!prediction || !selectedFile) return;
  reportMessage.textContent = "";
  reportModal.classList.remove("hidden");
  patientName.focus();
}

function closeReport() {
  reportModal.classList.add("hidden");
  reportForm.reset();
  reportMessage.textContent = "";
}

async function generateReport(event) {
  event.preventDefault();
  const name = patientName.value.trim();
  const age = Number(patientAge.value);
  if (!name || !Number.isInteger(age) || age < 0 || age > 130) {
    reportMessage.textContent = "Enter a valid patient name and age between 0 and 130.";
    return;
  }
  const button = reportForm.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = "Generating...";
  try {
    const data = new FormData();
    data.append("patient_name", name);
    data.append("patient_age", String(age));
    data.append("disease", prediction.disease);
    data.append("confidence", prediction.confidence);
    data.append("image", selectedFile);
    const response = await fetch("/generate-report", { method: "POST", body: data });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Unable to generate the report.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "dermavision-report.pdf";
    link.click();
    URL.revokeObjectURL(url);
    closeReport();
  } catch (error) {
    reportMessage.textContent = error.message || "Unable to generate the report.";
  } finally {
    button.disabled = false;
    button.textContent = "Generate PDF";
  }
}

uploadBtn.addEventListener("click", () => deviceInput.click());
deviceInput.addEventListener("change", (event) => setPreview(event.target.files[0]));
removeImageBtn.addEventListener("click", clearImage);
cameraBtn.addEventListener("click", openCamera);
cancelCameraBtn.addEventListener("click", closeCamera);
takePhotoBtn.addEventListener("click", takePhoto);
predictBtn.addEventListener("click", predictDisease);
resetBtn.addEventListener("click", clearImage);
reportBtn.addEventListener("click", openReport);
closeReportBtn.addEventListener("click", closeReport);
cancelReportBtn.addEventListener("click", closeReport);
reportForm.addEventListener("submit", generateReport);
window.addEventListener("beforeunload", stopCamera);
