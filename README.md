# Diabetic-Retinopathy-Detection

A Python-/Deep-Learning-based project for automatic detection and grading of diabetic retinopathy (DR) from retinal fundus images.

## 🚀 Overview

This repository implements methods to detect (and possibly grade) diabetic retinopathy in eye-fundus/fundus images. The key steps include:

* Pre-processing of retinal fundus images (cropping, resizing, contrast/normalisation)
* Image augmentation and dataset balancing
* Training deep convolutional neural networks (CNNs) and/or transfer-learning models to classify retina images into DR severity classes (e.g., no DR, mild, moderate, severe, proliferative)
* Evaluation of model performance (accuracy, Cohen’s Kappa, precision/recall, confusion matrices)
* Optionally: model deployment or visualisation of results for interpretation.

## 🗂️ Repository Structure

```
Diabetic-Retinopathy-Detection/
│
├── data/                    # (optional) raw & pre-processed image data
├── notebooks/               # exploratory Jupyter notebooks (data-analysis, EDA)
├── models/                  # saved/trained model checkpoints
├── src/                     # source code: preprocessing, training, evaluation
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # License file
```

*(Adjust actual structure as appropriate for your repo.)*

## 🛠️ Features

* Accepts retinal fundus image input and classifies into DR severity levels
* Data-preprocessing: resizing, cropping, safe normalization of colour channels
* Data augmentation: rotation, flip, brightness/contrast adjustments, etc.
* Supports transfer-learning backbones (e.g., ResNet, EfficientNet) for feature extraction
* Evaluation metrics: accuracy, Cohen’s Kappa (weighted/unweighted), confusion matrix, ROC/AUC where applicable
* Flexible configuration so you can experiment with different architectures, hyper-parameters, and dataset splits
* (Optional) Visualisation tools such as Grad-CAM or saliency mapping for model interpretability

## 📦 Prerequisites & Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/Taher284/Diabetic-Retinopathy-Detection.git
   cd Diabetic-Retinopathy-Detection
   ```
2. (Recommended) Create a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Prepare your dataset:

   * Download a suitable DR fundus-image dataset (e.g., APTOS 2019 Blindness Detection, EyePACS, or another public DR dataset)
   * Place your images/folders in a `data/` directory (or update config paths accordingly)
5. Configure the run parameters (in maybe a config file or direct constants in code): dataset path, image sizes, backbone architecture, number of classes, hyper-parameters (learning rate, batch size, epochs).
6. To train a model:

   ```bash
   python train.py --config config.yaml
   ```

   *(Replace with actual command if your script differs.)*
7. To evaluate a trained model:

   ```bash
   python evaluate.py --model models/my_model.pth --data data/val/
   ```

   *(Replace accordingly.)*

## 📊 Workflow

Typical step-by-step pipeline:

1. **Load & Pre-process images**: Read raw fundus images, apply cropping (e.g., remove black borders), resize to common input size (e.g., 224×224 or 512×512), normalise pixel values.
2. **Split dataset**: Train / Validation / Test splits ensuring class-balance as far as possible.
3. **Data Augmentation**: Apply on-the-fly or offline augmentation to increase data variety (rotations, flips, brightness/contrast change).
4. **Model training**: Use a CNN (or transfer-learning backbone) to learn features, fine-tune on DR dataset. Possibly use multi-class classification (0–4) or binary classification (no DR vs DR).
5. **Evaluation**: Compute metrics: accuracy, Cohen’s Kappa (important for ordinal classification of DR), confusion matrix, ROC/AUC (binary).
6. **Interpretation & Visualisation**: Optionally visualise what the network is “looking at” via Grad-CAM or saliency maps.
7. **Deployment (optional)**: Use saved model weights to build an inference script or web application for uploading a fundus image and receiving a DR-severity prediction.

## ✅ Usage & Examples

* For a quick test, you may run on a small subset of images to verify the pipeline works.
* Examine the logs or printed metrics to understand where your model is performing well or poorly (common issues: class imbalance, low data quality, over-fitting).
* Extend to other datasets or architectures by modifying config/hyperparameters.

## 📈 Evaluation Metrics (Important for DR)

Since DR classification often has an ordinal progression (0 → 4), some key metrics:

* **Accuracy**: proportion correct.
* **Cohen’s Kappa / Quadratic Weighted Kappa (QWK)**: measures agreement between predicted & true labels, penalises large misclassifications more.
* **Precision / Recall / F1-Score** for each class.
* **Confusion Matrix** to see where the model tends to mis-classify.
* **ROC/AUC** (binary case: no DR vs any DR) for screening tasks.

## 🧩 Extending the Project

* Add support for **multi-class grading** (e.g., 0–4 levels), not just binary classification.
* Introduce **explainability** via saliency maps / Grad-CAM to highlight lesions/areas influencing prediction.
* Integrate **better data-augmentation strategies** or domain-specific preprocessing (e.g., cropping optic disc, removing artefacts).
* Incorporate **ensemble models** to improve performance (e.g., combining outputs of ResNet, EfficientNet, DenseNet).
* Build **web-/mobile-app interface** to allow fundus image upload and DR-prediction by clinicians or patients.
* Add **data-quality checking** (many fundus images are blurred or low-quality, affect model performance).
* Explore **semi-supervised / self-supervised learning** if labelled data is limited.

## 👥 Contributing

Contributions are welcome! If you’d like to:

1. Fork the repository and create your branch.
2. Make your changes (bug-fix, new augmentation, new model).
3. Add or update documentation/tests as needed.
4. Submit a Pull Request with clear description of your changes.


Let me know if you’d like sections added or removed (for example: dataset download instructions, architecture diagram, result plots, limitations section) or want the README tailored for a specific audience (e.g., academic, industry, collaborators).
