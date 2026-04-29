let selectedFile = null;

const deviceInput = document.getElementById("deviceInput");
const cameraInput = document.getElementById("cameraInput");
const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("resultBox");
const diseaseName = document.getElementById("diseaseName");
const confidence = document.getElementById("confidence");

function handleFileSelect(event) {
  const file = event.target.files[0];

  if (!file) return;

  selectedFile = file;

  previewImage.src = URL.createObjectURL(file);
  previewBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
}

deviceInput.addEventListener("change", handleFileSelect);
cameraInput.addEventListener("change", handleFileSelect);

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

  } catch (error) {
    alert("Prediction failed. Please try again.");
    console.error(error);
  } finally {
    loader.classList.add("hidden");
  }
});