import streamlit as st

def evaluation_page():
    # ===============================
    # HEADER
    # ===============================
    st.markdown(
        "<h3 style='text-align:center;'>Evaluasi Model Klasifikasi BISINDO</h3>",
        unsafe_allow_html=True
    )

    col_cnn, col_vgg, col_resnet = st.columns(3)

    with col_cnn:
        st.markdown(
            """
            <h4 style='text-align:center;'>CNN</h4>
            <hr style='border:1px solid #cccccc; margin-top:5px; margin-bottom:15px;'>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Confusion Matrix**")
        st.image("../modelling/evaluation/cm_cnn.png", use_container_width=True)

        st.markdown("**Grafik Loss**")
        st.image("../modelling/evaluation/loss_cnn.png", use_container_width=True)

    with col_vgg:
        st.markdown(
            """
            <h4 style='text-align:center;'>VGG16</h4>
            <hr style='border:1px solid #cccccc; margin-top:5px; margin-bottom:15px;'>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Confusion Matrix**")
        st.image("../modelling/evaluation/cm_vgg.png", use_container_width=True)

        st.markdown("**Grafik Loss**")
        st.image("../modelling/evaluation/loss_vgg.png", use_container_width=True)

    with col_resnet:
        st.markdown(
            """
            <h4 style='text-align:center;'>ResNet50</h4>
            <hr style='border:1px solid #cccccc; margin-top:5px; margin-bottom:15px;'>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Confusion Matrix**")
        st.image("../modelling/evaluation/cm_resnet.png", use_container_width=True)

        st.markdown("**Grafik Loss**")
        st.image("../modelling/evaluation/loss_resnet.png", use_container_width=True)
