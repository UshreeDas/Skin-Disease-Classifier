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
New project/
|-- app.py
|-- train_model.py
|-- download_dataset.py
|-- requirements.txt
|-- README.md
|-- templates/
|   |-- index.html
|-- static/
|   |-- css/
|   |   |-- style.css
|   |-- js/
|       |-- app.js
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
python train_model.py --data_dir "C:\path\to\IMG_CLASSES" --epochs 8
```

What this does:

- reads images from the Kaggle dataset folders
- creates training and validation splits automatically
- uses MobileNetV2 transfer learning
- saves the model in `models/skin_disease_classifier.keras`
- saves class names in `models/class_names.json`

### 5. Start the Flask application

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### 6. Use the web app

- Click `Choose Existing Photo` to upload an image from the device.
- Click `Take Photo With Camera` to capture a new image.
- Click `Analyze Image` to get the predicted disease class and top 3 confidence scores.

## How the Backend Works

- `app.py` loads the trained model and label list from the `models` folder.
- The `/predict` route accepts an uploaded image.
- The image is resized to `224 x 224`.
- The model returns probabilities for each disease class.
- The app sends the best prediction and top 3 results back as JSON.

## How the Frontend Works

- `templates/index.html` contains the upload UI and result sections.
- `static/css/style.css` handles the responsive design and visual layout.
- `static/js/app.js` previews the image and sends it to Flask using `fetch`.

## Important Note

This project is for learning and demonstration purposes only. Skin disease prediction from images should not be treated as a medical diagnosis.
