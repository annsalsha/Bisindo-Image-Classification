import streamlit as st
from eda import eda_citra
from cnn import cnn_page
from VGG import vgg_page
from resnet import resnet_page
from evaluation import evaluation_page

st.set_page_config(
    page_title="Klasifikasi Citra",
    page_icon="🖼️",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>Dashboard Klasifikasi Citra</h1>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5= st.tabs(
    ["EDA", "CNN", "VGG16", "ResNet50", "Evaluation"]
)

with tab1:
    st.subheader("Exploratory Data Analysis (EDA) - Data Citra")
    st.write("Dataset yang digunakan dalam penelitian ini adalah dataset BISINDO (Bahasa Isyarat Indonesia) yang diperoleh dari platform Kaggle. Dataset ini berisi citra tangan yang merepresentasikan alfabet BISINDO (A–Z), sehingga total terdapat 26 kelas. Setiap kelas terdiri dari sejumlah gambar dengan variasi posisi tangan, sudut pengambilan, dan kondisi pencahayaan. Dataset ini digunakan untuk membangun dan mengevaluasi model klasifikasi citra berbasis deep learning seperti CNN, VGG, dan ResNet.")
    st.markdown("---")
    eda_citra("../bisindo/images/train")

with tab2:
    cnn_page()

with tab3:
    vgg_page()

with tab4:
    resnet_page()

with tab5:
    evaluation_page()
