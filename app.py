import streamlit as st
from ultralytics import YOLO
from PIL import Image
import re

st.set_page_config(page_title="Türk Lirası Tanıma Sistemi", layout="centered")

st.title("YOLOv11 Tabanlı Türk Lirası Banknot Tanıma")
st.write("Sisteme bir banknot görseli yükleyin, model paraları tespit edip toplam değerini hesaplasın.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Banknot görselini yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Orijinal Görsel", use_container_width=True)
    
    if st.button("Paraları Analiz Et"):
        with st.spinner('Model görüntüyü işliyor...'):
            results = model(image)
            
            res_plotted = results[0].plot()
            res_rgb = res_plotted[:, :, ::-1] 
            
            st.image(res_rgb, caption="Tespit Sonucu", use_container_width=True)
            
            toplam_deger = 0
            boxes = results[0].boxes
            names = results[0].names
            
            for c in boxes.cls:
                class_name = names[int(c)]
                sayi_degeri = re.sub(r'\D', '', class_name)
                if sayi_degeri:
                    toplam_deger += int(sayi_degeri)
                    
            st.success(f"Tespit Edilen Toplam Tutar: {toplam_deger} TL")