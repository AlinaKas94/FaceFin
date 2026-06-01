# 🔍 FaceFin v0.9.8b

**FaceFin** іs a fully local system for face recognition inside photos and documents (PDF, DOCX). The application operates 100% offline, requires no internet connection. 
Made by me for fun and home using. 

---

### ✨ Key Features
* **Local AI (Computer Vision):** Powered by *RetinaFace* for ultra-precise face detection and *ArcFace (ResNet-50)* for extracting 512-dimensional biometric embeddings.
* **Blazing Fast Search (Meta FAISS):** Vector indexing built on Facebook AI technologies provides sub-millisecond search across millions of records.
* **In-Memory Document Parsing:** Native support for multi-page PDFs (via *Google PDFium*) and MS Word documents (via *libzip*) processing entirely in RAM with no temporary disk overhead.
* **Smart Duplicates Grouping:** The AI automatically groups 100% identical photos from different folders into a single card with an expandable context menu to open specific copies.
* **Shadow Enhancement (CLAHE):** Adaptive histogram equalization brings faces out of deep shadows, allowing the AI to "see" even in low-light conditions.
* **Multilingual & Customizable:** On-the-fly language switching (Ukrainian / English) and beautifully tailored themes (Dark, Light, Nord, Solarized).

---

### 🚀 Quick Start (For Users)
1. Go to the **[Releases](https://github.com/AlinaKas94/FaceFin/releases)** tab and download `FaceFin_v0.9.8b.zip`.
2. Extract the ZIP archive into any folder.
3. Run `FaceFin.exe`.

---

## 🛠️ Developer Guide

### Dependencies
The project is built using **CMake** and the **vcpkg** package manager (`x64-windows` triplet).

Install required packages via vcpkg:
powershell
vcpkg install qt5-base:x64-windows opencv4:x64-windows unofficial-sqlite3:x64-windows nlohmann-json:x64-windows onnxruntime:x64-windows faiss:x64-windows libzip:x64-windows

Google PDFium Setup (Manual)
Since there is no stable official port for PDFium in vcpkg, you need to download precompiled binaries manually:

1. Download pdfium-windows-x64.zip from the latest bblanchon/pdfium-binaries release.
2. Extract it into your project root directory under: third_party/pdfium/.

Building the Project
cmake -B build -DCMAKE_TOOLCHAIN_FILE="C:/YOUR_VCPKG_PATH/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release

🧠 AI Models
The application expects ONNX models to be placed in the /models folder right next to the executable:

det_10g.onnx (RetinaFace)

w600k_r50.onnx (ArcFace)

Note: These models belong to the open-source InsightFace community and are provided strictly for Non-commercial / Research use only.

📄 License
This project is licensed under the MIT License (see the LICENSE file).
Third-party libraries used (OpenCV, Qt, FAISS, ONNX, PDFium) are licensed by their respective authors under Apache 2.0, MIT, BSD, and LGPL licenses.

Author / Developer: AlinaKas94

