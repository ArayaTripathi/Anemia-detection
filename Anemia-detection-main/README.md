
An AI-powered clinical hematology web application that classifies types of anemia from microscopic peripheral blood smear images and complete blood count (CBC) parameters using deep transfer learning.

---

## 🔬 Key Features

- **Multi-Class Anemia Classification**: Accurately classifies 5 clinical categories of anemia:
  1. **Iron Deficiency Anemia (IDA)**
  2. **Sickle Cell Anemia (SCA)**
  3. **Thalassemia (TH)**
  4. **Vitamin B12 Deficiency (Megaloblastic)**
  5. **Dimorphic Anemia (DA)**
- **Deep Neural Network**: Trained VGG16 backbone with custom dense classification layers for high-precision morphology recognition.
- **Microscopic Image Integration**: Interactive drag-and-drop file upload with live image preview, clear button, and quick-select sample blood smear library.
- **CBC Parameter Corroboration**: Evaluates Hemoglobin, MCV, RBC Count, and MCH against clinical criteria to validate morphological smear findings.
- **Automated Diagnostic Lab Reports**: Dynamically generates comprehensive PDF reports formatted with clinic branding, patient details, CBC profile, and pathologist sign-off.
- **Modern Responsive Web UI**: Developed with Flask, Bootstrap 5, Bootstrap Icons, and dedicated CSS/JS styling.

---

## 📁 Project Architecture

```
Anemia-detection-main/
│
├── M_Anemia.h5             # Pre-trained deep learning model weights
├── app.py                  # Flask web server and routing logic
├── model_service.py        # Model loading, image preprocessing, and inference engine
├── pdf.py                  # Automated PDF diagnostic report generation
├── requirements.txt        # Python package dependencies
│
├── static/                 # Static web assets
│   ├── css/
│   │   └── style.css       # Custom modern healthcare dashboard styles
│   ├── js/
│   │   └── script.js       # Drag-and-drop, preview, sample loader, and form handling
│   ├── samples/            # Verified sample blood smear images for 1-click testing
│   │   ├── IDA15.png       # Iron Deficiency Anemia sample
│   │   ├── SCA0106.jpg     # Sickle Cell Anemia sample
│   │   ├── TH0004.png      # Thalassemia sample
│   │   ├── VB12_0000.jpg   # Vitamin B12 Deficiency sample
│   │   └── DA0061.jpg      # Dimorphic Anemia sample
│   └── logo.png            # Laboratory branding logo
│
├── templates/              # HTML frontend templates
│   ├── index.html          # Main diagnostic input and prediction results interface
│   ├── home.html           # Disease guide and educational overview
│   └── about.html          # Team and technical pipeline overview
│
├── uploads/                # Temporary storage for analyzed smear images
└── reports/                # Output directory for generated PDF lab reports
```

---



