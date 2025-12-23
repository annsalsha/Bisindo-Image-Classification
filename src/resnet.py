import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from tensorflow.keras.applications.resnet50 import preprocess_input

def resnet_page():
    resnet_model = tf.keras.models.load_model("../modelling/resnet_model.h5")

    with open("../modelling/label_resnet.json", "r") as f:
        label_dict = json.load(f)

    # index → label
    class_names = {v: k for k, v in label_dict.items()}

    st.subheader("Klasifikasi Citra BISINDO (ResNet50)")

    col_input, col_output = st.columns(2)

    with col_input:
        uploaded_file = st.file_uploader(
            "Pilih gambar tangan",
            type=["jpg", "jpeg", "png"],
            key="resnet_uploader"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Gambar Input",
                use_container_width=True
            )

    with col_output:
        st.markdown(
        """
        <h2 style="text-align:center;">
            📊 Hasil Prediksi
        </h2>
        """,
        unsafe_allow_html=True
        )

        if uploaded_file is not None:
            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img)

            img_array = preprocess_input(img_array)

            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            predictions = resnet_model.predict(img_array)
            pred_class = int(np.argmax(predictions))
            confidence = float(np.max(predictions))

            col_pred, col_conf = st.columns(2)

            with col_pred:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1e293b;
                        padding: 20px;
                        border-radius: 12px;
                        text-align: center;
                    ">
                        <h3 style="color:#94a3b8;">Prediksi Huruf</h3>
                        <h4 style="color:white; margin:0; text-align: center;">
                            {class_names[pred_class]}
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_conf:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #0f766e;
                        padding: 20px;
                        border-radius: 12px;
                        text-align: center;
                    ">
                        <h3 style="color:#ccfbf1;">Confidence</h3>
                        <h4 style="color:white; margin:0;text-align: center;">
                            {confidence:.2%}
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.info("Silakan upload gambar terlebih dahulu.")
