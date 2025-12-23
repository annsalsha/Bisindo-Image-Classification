import os
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
import numpy as np

def eda_citra(base_dir, img_size=(224, 224)):

    classes = sorted([
        cls for cls in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, cls))
    ])

    counts = [
        len(os.listdir(os.path.join(base_dir, cls)))
        for cls in classes
    ]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Statistik Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Jumlah Kelas", len(classes))

        with col2:
            st.metric("Total Gambar", sum(counts))

        with col3:
            st.metric(
                "Rata-rata / Kelas",
                round(np.mean(counts), 2)
            )

        st.markdown("### Distribusi Jumlah Gambar per Kelas")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(classes, counts)
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Jumlah Gambar")
        ax.set_title("Distribusi Gambar")
        ax.set_xticklabels(classes, rotation=45, ha="right")
        st.pyplot(fig)

    with col_right:
        st.markdown("### Contoh Gambar per Kelas")

        max_display = 6 
        display_classes = classes[:max_display]

        rows = 2
        cols = 3

        fig, axes = plt.subplots(rows, cols, figsize=(8, 5))
        axes = axes.flatten()

        for i, cls in enumerate(display_classes):
            class_dir = os.path.join(base_dir, cls)
            img_name = os.listdir(class_dir)[0]
            img_path = os.path.join(class_dir, img_name)

            img = load_img(img_path, target_size=img_size)
            axes[i].imshow(img)
            axes[i].set_title(cls)
            axes[i].axis("off")

        # Kosongkan subplot sisa
        for j in range(len(display_classes), len(axes)):
            axes[j].axis("off")

        st.pyplot(fig)

