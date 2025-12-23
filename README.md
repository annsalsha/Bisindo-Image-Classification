<h1 align="center">
<b>Klasifikasi Citra Bahasa Isyarat BISINDO</b>
</h1>

Table of Content


---

<h2 align="center">
<b>🚀Deskripsi Proyek🚀</b>
</h2>
Proyek ini bertujuan untuk mengklasifikasikan citra bahasa isyarat BISINDO (Bahasa Isyarat Indonesia) menggunakan pendekatan Deep Learning. Sistem dikembangkan dalam bentuk website interaktif menggunakan Streamlit.

## Latar Belakang
Bahasa Isyarat Indonesia (BISINDO) merupakan media komunikasi penting bagi penyandang tunarungu dan tunawicara, namun masih banyak masyarakat yang belum memahaminya. Proyek ini bertujuan mengembangkan sistem klasifikasi citra BISINDO menggunakan deep learning untuk mengenali gesture tangan dari gambar, sehingga dapat mempermudah komunikasi, mendukung edukasi, dan menjadi dasar pengembangan aplikasi pengenalan bahasa isyarat di Indonesia.

## Tujuan Pengembangan
- Menciptakan sistem klasifikasi citra Bahasa Isyarat Indonesia (BISINDO) yang mampu mengenali gesture tangan dari gambar.
- Membantu komunikasi antara penyandang tunarungu dan masyarakat umum.
- Mendukung proses pembelajaran bahasa isyarat.
- Menjadi dasar bagi pengembangan aplikasi berbasis computer vision untuk edukasi dan aksesibilitas.

---
<h2 align="center">
<b>📂Dataset📂</b>
</h2>

- Nama Dataset: Indonesian Sign Language - BISINDO
- Sumber: Kaggle
- Jumlah Kelas: 26 kelas alfabet
- Jumlah Gambar: ±7.700 gambar
- Deskripsi
  Dataset ini merupakan hasil kolaborasi dengan sejumlah relawan di Universitas Budi Luhur, di mana para relawan secara sukarela menyumbangkan gambar isyarat untuk proyek akhir
  Memberikan kontribusi penting untuk pemahaman bahasa isyarat Indonesia melalui representasi visual alfabetnya.

🔗 [https://www.kaggle.com/](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo) 

Struktur dataset:
```bash
Bisindo/
├── Image/           
│   ├── Train/
│   │   ├── A
│   │   ├── B
│   │   └── Dst.
│   ├── Val/
│   │   ├── A
│   │   ├── B
│   │   └── Dst.
└── Labels/               
    ├── Train/
    │   ├── A
    │   ├── B
    │   └── Dst.
    ├── Val/
    │   ├── A
    │   ├── B
    │   └── Dst.
```
---
<h2 align="center">
<b>⚙️Preprocessing dan Pemodelan⚙️</b>
</h2>

## Preprocessing Data
Tahapan preprocessing yang dilakukan:
- Resize citra menjadi 224×224 piksel agar sesuai dengan input layer semua model (CNN, VGG16, ResNet50).
- Normalisasi pixel untuk menyesuaikan nilai piksel ke skala 0–1, sehingga mempercepat konvergensi model.
- Preprocessing khusus sesuai model:
  - CNN: rescale pixel /255
  - VGG16: menggunakan fungsi preprocess_input dari Keras untuk menyesuaikan input dengan pretrained model
  - ResNet50: menggunakan fungsi preprocess_input untuk menyesuaikan input dengan pretrained model


## Pemodelan

### 1. Non-Pretrained
CNN adalah jenis neural network Non-Pretrained khusus untuk data citra (gambar) yang dirancang untuk mengekstrak fitur secara otomatis. CNN sangat efektif dalam mengenali pola visual, seperti bentuk tangan atau gerakan jari pada dataset Bisindo.
Arsitektur utama:
- Conv2D, Layer convolution untuk mengekstrak fitur lokal dari gambar (misal tepi, bentuk tangan, gerakan jari).
- MaxPooling, Mengurangi dimensi feature map sekaligus mempertahankan informasi penting.
- GlobalAveragePooling2D, mengubah feature map 2D menjadi vektor 1D.
- Dense, Fully connected layer untuk klasifikasi akhir berdasarkan fitur yang diekstrak.
- Dropout, Regularisasi untuk mengurangi overfitting, dengan mematikan sebagian neuron secara acak saat training.

Tujuan dan penggunaan:
- CNN digunakan sebagai baseline model untuk melihat performa awal pada dataset Bisindo.
- Memberikan referensi akurasi awal sebelum menggunakan model pretrained yang lebih kompleks.
- Training cukup satu tahap karena semua layer dibangun dari awal, tidak ada tahap feature extraction atau fine-tuning.

Kelebihan
- Sederhana dan mudah dimodifikasi.
- Dapat dijalankan tanpa membutuhkan pretrained weights.
- Memberikan gambaran performa awal dataset sebelum strategi transfer learning diterapkan.

### 2. Transfer Learninga
a. VGG16
VGG16 adalah model pretrained populer yang menggunakan arsitektur convolutional dalam-dalam. Dalam proyek ini, strategi training dilakukan dua tahap:
- Feature Extraction (Frozen Convolutional Base)
  Bagian convolutional dari VGG16 dikunci (frozen) sehingga hanya classifier (Dense layer) yang dilatih.
  Tujuannya: memanfaatkan fitur yang sudah dipelajari dari ImageNet tanpa merusak pretrained weights.

- Fine-Tuning (Unfreeze Beberapa Layer)
  Beberapa layer convolutional terakhir di VGG16 di-unfreeze agar dapat dilatih kembali.
  Memberikan fleksibilitas model untuk menyesuaikan fitur spesifik dari dataset Bisindo.

Kelebihan strategi ini:
- Menghemat waktu training dibanding melatih dari awal.
- Memanfaatkan fitur umum dari ImageNet sekaligus menyesuaikan fitur khusus dataset Bisindo.

b. ResNet50
ResNet50 adalah model pretrained yang terkenal dengan residual connection, sehingga mampu mengatasi masalah vanishing gradient pada model sangat dalam. Strategi training di proyek ini juga menggunakan dua tahap:
- Feature Extraction (Frozen Convolutional Base)
  Convolutional base dikunci, hanya classifier yang dilatih.
  Memanfaatkan fitur yang sudah ada dari pretrained ResNet50.

- Fine-Tuning (Unfreeze Beberapa Layer Terakhir)
- Beberapa layer terakhir convolutional di-unfreeze untuk di-train ulang.
- Membantu model belajar fitur spesifik untuk Bisindo, meningkatkan akurasi.

Kelebihan strategi ini:
ResNet50 lebih tahan terhadap degradasi performa saat model sangat dalam.
Residual connection memungkinkan fine-tuning lebih stabil dibanding VGG16.

---

<h2 align="center">
<b>🔍Hasil dan Analisis🔍</b>
</h2>

## Classification Report
- Accuracy → seberapa tepat prediksi model secara keseluruhan
- Precision → akurasi prediksi per kelas
- Recall → kemampuan model mendeteksi semua sampel dari kelas tertentu
- F1-score → kombinasi precision dan recall

Tabel Perbandingan :
| Metric   | CNN | VGG16 | ResNet50 |
|----------|-----|-------|----------|
| Precision| 86% | 84%   | 98%      |
| Recall   | 85% |84%    |98%       |
| F1-Score | 85% |84%    |98%       |
| Accuracy | 85% | 84%   | 98%      |

## Confusion Matrix
| CNN | VGG16 | ResNet50 |
|-----|-------|----------|
| <img width="919" height="855" alt="cm_cnn" src="https://github.com/user-attachments/assets/0fbfec2d-c6f1-4946-af56-ede8cb186aba" /> | <img width="928" height="855" alt="cm_vgg" src="https://github.com/user-attachments/assets/5a3dda51-8119-48c3-9b9c-03d6a434bfd7" /> | <img width="919" height="855" alt="cm_resnet" src="https://github.com/user-attachments/assets/45f1dbc0-b835-4eda-952c-1920df2f9e30" /> |

## Learning Curve
| CNN | VGG16 | ResNet50 |
|-----|-------|----------|
| <img width="1189" height="490" alt="loss_cnn" src="https://github.com/user-attachments/assets/3057a551-f1e9-48e0-ad95-d004db48e650" /> | <img width="1189" height="490" alt="loss_vgg" src="https://github.com/user-attachments/assets/51036f77-2324-47d6-a69f-fc52686e5a89" /> | <img width="1189" height="490" alt="loss_resnet" src="https://github.com/user-attachments/assets/0aa3744e-4bec-4512-970c-dcbb5b6e3b97" /> |

---

<h2 align="center">
<b>🤖Sistem Klasifikasi Sederhana🤖</b>
</h2>

Sistem klasifikasi sederhana ini dirancang untuk mengenali gesture tangan BISINDO, memproses gambar input, mengekstrak fitur penting, dan memprediksi huruf alfabet yang sesuai. Sistem ini juga berfungsi sebagai baseline sebelum menggunakan model yang lebih kompleks.
Sistem web menyediakan 4 menu:
- EDA Dataset
- CNN
- VGG16
- ResNet50
- Evaluation

## Tampilan EDA
Tampilan EDA menampilkan contoh gambar, distribusi kelas, serta statistik dataset seperti jumlah kelas, total data, dan rata-rata sampel per kelas, untuk memahami karakteristik dataset BISINDO sebelum preprocessing dan pelatihan model.
<img width="1366" height="635" alt="image" src="https://github.com/user-attachments/assets/9073ee27-aba0-4a08-a426-a202af607ed5" />

## Tampilan Model
Pengguna dapat mengunggah gambar gesture tangan, lalu sistem akan melakukan prediksi untuk menampilkan huruf alfabet BISINDO yang sesuai beserta tingkat keyakinan (confidence) prediksi model.
<img width="1366" height="465" alt="image" src="https://github.com/user-attachments/assets/dd2d74f0-84fd-4336-9880-b78e38d7cf98" />

## Tampilan Evaluation
Sistem menampilkan confusion matrix untuk melihat kesalahan klasifikasi per kelas, serta learning curve berupa grafik loss dan accuracy selama proses training dan validation, guna mengevaluasi performa model secara keseluruhan.
<img width="1366" height="590" alt="image" src="https://github.com/user-attachments/assets/3087c6e5-a035-418e-bfc5-5e48917dc3b2" />

---

<h2 align="center">
<b>🛠️Langkah Instalasi🛠️</b>
</h2>

1. Clone Repository
    ```bash
    git clone https://github.com/username/bisindo-image-classification.git
    cd project
    ```
2. Download file model & kode sumber dari Google Drive

   Link: https://drive.google.com/drive/folders/1VzozMIRV-RiY-gf_ZcVfQ94Bv9XTqaYW?usp=sharing
   
   Simpan ke folder yang sesuai di lokal:
    ```
   project/
    ├── src/
    ├── modelling/
    ├── training/
    └── requirements.txt
    ```
4. Siapkan virtual environment dan install dependencies
    ```
   # Windows
    python -m venv env
    env\Scripts\activate
    
    # macOS / Linux
    python3 -m venv env
    source env/bin/activate
    
    pip install -r requirements.txt
    ```
5. Jalankan sistem
    ```bash
    streamlit run app.py
    ```
