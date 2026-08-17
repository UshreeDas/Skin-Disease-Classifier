# Skin Disease Classifier

This project is a complete skin disease classification web app built with Flask, HTML, CSS, JavaScript, and TensorFlow.
It supports two image input modes:

- upload an existing image from the device
- take a new image using the device camera

The model is designed to train on the Kaggle dataset from your screenshot:
[Skin diseases image dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)

The training code does not hardcode class names. It reads the disease classes directly from the dataset folders inside `IMG_CLASSES`, which is the safest way to stay aligned with the dataset structure you download.

## Project Structure

```text
Skin-Disease-Classifier/
|-- app.py
|-- train_model.py
|-- download_dataset.py
|-- requirements.txt
|-- README.md
|-- model/
|   |-- skin_model.keras
|   |-- class_names.json
|-- templates/
|   |-- index.html
|-- static/
|   |-- style.css
|   |-- script.js
```

## Step-by-Step Implementation

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install the required libraries

```powershell
pip install -r requirements.txt
```

Hardware acceleration is automatic when TensorFlow can see a supported GPU.

For Apple Silicon Macs, TensorFlow uses the Apple GPU through Metal, not the Neural Engine directly:

```powershell
pip install tensorflow-metal==1.2.0
```

Avoid TensorFlow `2.20.x` with `tensorflow-metal` for now; it can fail while loading `libmetal_plugin.dylib`.

For PCs, install the TensorFlow build and GPU drivers/CUDA stack appropriate for your GPU. If no supported GPU is visible, the app and training script fall back to CPU.

### 3. Download the Kaggle dataset

You have two options.

Option A: use the helper script in this project

```powershell
python download_dataset.py
```

Option B: download it manually from Kaggle using the page in your screenshot.

After download, locate the folder named `IMG_CLASSES`.

### 4. Train the deep learning model

Replace the path below with your real dataset path:

```powershell
python train_model.py --data_dir "dataset\IMG_CLASSES" --epochs 10
```

What this does:

- reads images from the Kaggle dataset folders
- creates training and validation splits automatically
- uses MobileNetV2 transfer learning
- saves the model in `model/skin_model.keras`
- saves class names in `model/class_names.json`

### 5. Start the Flask application

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

To check what TensorFlow is using:

```text
http://127.0.0.1:5000/runtime
```

### 6. Use the web app

- Click `Upload from Device` to upload an image from the device.
- Click `Capture Photo` to capture a new image.
- Click `Predict Disease` to get the predicted disease class and confidence score.

## How the Backend Works

- `app.py` loads the trained model and label list from the `model` folder.
- `runtime_config.py` configures TensorFlow to prefer a visible GPU and fall back to CPU.
- The `/predict` route accepts an uploaded image.
- The image is resized to `224 x 224`.
- The model returns probabilities for each disease class.
- The app sends the best prediction and confidence score back as JSON.

## How the Frontend Works

- `templates/index.html` contains the upload UI and result sections.
- `static/style.css` handles the responsive design and visual layout.
- `static/script.js` previews the image and sends it to Flask using `fetch`.

## Deployment (Render)

The app is set up to deploy on [Render](https://render.com) as a Python web service running behind `gunicorn`.

1. Push this repo to GitHub (the model files in `model/` are small enough to commit directly, no LFS needed).
2. In Render, choose **New > Blueprint** and point it at this repo. Render will read [render.yaml](render.yaml) and create the service automatically.
   - Alternatively, choose **New > Web Service** manually and set:
     - Build command: `pip install -r requirements.txt`
     - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
3. Deploy. First boot is slow (TensorFlow import + model load), subsequent requests are fast.

Notes:

- `render.yaml` defaults to Render's **free** plan (512 MB RAM). TensorFlow plus the model can be tight on that plan — if the service fails to boot or crashes under load, bump the plan to **Starter** or higher in the Render dashboard.
- The start command uses a single worker with multiple threads on purpose, so only one copy of the model is loaded into memory. Don't raise `--workers` without also raising the plan's RAM.
- The free plan spins the service down after inactivity; the next request will be slow while it wakes up and reloads the model.
- A generic `Procfile` is also included, so the same setup works on Railway, Heroku, or any platform that reads it.

## Important Note

This project is for learning and demonstration purposes only. Skin disease prediction from images should not be treated as a medical diagnosis.
