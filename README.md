# Klasifikasi Citra Bahasa Isyarat BISINDO

## Deskripsi Proyek
Proyek ini bertujuan untuk mengklasifikasikan citra bahasa isyarat BISINDO (Bahasa Isyarat Indonesia) menggunakan pendekatan Deep Learning. Sistem dikembangkan dalam bentuk website interaktif menggunakan Streamlit.

Tiga model yang digunakan:
- Convolutional Neural Network (CNN)
- VGG16
- ResNet50

---

## Dataset
Dataset yang digunakan adalah dataset BISINDO yang diperoleh dari Kaggle:

🔗 [https://www.kaggle.com/](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo) 

Dataset berisi citra tangan bahasa isyarat huruf A–Z dengan total:
- 26 kelas
- ±7.700 gambar

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

## Preprocessing Data
Tahapan preprocessing yang dilakukan:
- Resize citra menjadi 224×224
- Normalisasi pixel
- Preprocessing khusus sesuai model:
  - CNN: rescale /255
  - VGG16: `preprocess_input`
  - ResNet50: `preprocess_input`

---

## Model yang Digunakan

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

## Evaluasi Model
Evaluasi dilakukan menggunakan:
- Classification Report (Accuracy, Precision, Recall, F1-Score)
- Confusion Matrix
- Grafik Loss & Accuracy

Ringkasan hasil:
| Model   | Accuracy |
|--------|----------|
| CNN    | 85%     |
| VGG16  | 84%     |
| ResNet | 98%     |

ResNet50 memberikan performa terbaik pada dataset BISINDO.

---

## Website (Streamlit)
Aplikasi web menyediakan:
- Upload citra
- Prediksi huruf BISINDO
- Confidence score
- EDA Dataset
- Evaluasi Model

---

## Cara Menjalankan Sistem Secara Lokal

1. Clone repository
```bash
git clone https://github.com/username/bisindo-image-classification.git
cd bisindo-image-classification
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
