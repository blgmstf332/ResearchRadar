import streamlit as st

st.set_page_config(
    page_title="ResearchRadar",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ResearchRadar")
st.write("Araştırma konularının yükseliş ve düşüş eğilimlerini keşfedin.")

konu = st.text_input(
    "Araştırmak istediğiniz konuyu girin:",
    placeholder="Örn. Artificial Intelligence"
)

if st.button("Analiz Et"):
    if konu.strip():
        st.subheader(f"📊 {konu} — Analiz Sonucu")

        st.warning(
            "Bu sonuçlar yalnızca arayüz taslağıdır. "
            "Gerçek analiz henüz bağlı değil."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Konu Eğilimi", "📈 Yükselişte")

        with col2:
            st.metric("Değişim Oranı", "+%24,5")

    else:
        st.warning("Lütfen bir araştırma konusu girin.")