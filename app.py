import streamlit as st 
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

#  YOLO
model_cus = YOLO('best.pt')

cola, colb = st.columns([1, 9])  
logo = Image.open("logo.jpg")
with cola:
    st.write(" ")
    st.image(logo, width=150)  # 
with colb:
    st.write("# AI Defects Detection PRO")  



st.markdown("""
<style>
    .container {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .left-column, .middle-column, .right-column {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    
</style>
""", unsafe_allow_html=True)

#st.write("""# YOLO Model: AI Welding Defects Detection PRO 👨‍🏭""")


uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Create three columns
    col_left, col_middle, col_right = st.columns([3, 2, 3])

    # Left column
    with col_left:
        st.markdown('<div class="left-column">', unsafe_allow_html=True)
        st.image(image, width=100)
        st.write("Image selected ✅")


        # Button to start
        if st.button('Find welding defects!'):
            st.session_state["detection_started"] = True


        st.markdown('</div>', unsafe_allow_html=True)

    # Middle column (only for arrow)
    with col_middle:
        st.markdown('<div class="middle-column">', unsafe_allow_html=True)
        if st.session_state.get("detection_started", False):

            st.image("arrow-right.png", caption="Model is running...", width=100)  
        st.markdown('</div>', unsafe_allow_html=True)

    # Right arrow
    with col_right:
        st.markdown('<div class="right-column">', unsafe_allow_html=True)
        if st.session_state.get("detection_started", False):


            # for bounding boxes
            def draw_boxes(image, boxes, classes, confidences):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(image)
                num = 0
                for box, cls, conf in zip(boxes, classes, confidences):
                    if conf > 0.2:
                        num += 1
                        x, y, w, h = box
                        x_min = (x - w / 2) * image.width
                        y_min = (y - h / 2) * image.height
                        width = w * image.width
                        height = h * image.height

                        rect = plt.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                        class_name = {
                            0: 'Burn-through', 1: 'Crack', 2: 'Crater', 3: 'Incomplete Penetration',
                            4: 'Overflow', 5: 'Porosity', 6: 'Spatter', 7: 'Undercut',
                            8: 'Normal Welding Line', 9: 'Irregular Welding Line'
                        }.get(int(cls), "Unknown")
                        ax.text(x_min, y_min - 5, f'{class_name} \nConfidence: {conf:.2f}',
                                bbox=dict(facecolor='red', alpha=0.5), fontsize=10, color='white')

                if num > 0:
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.write("Nothing was found!")

            try:
                results = model_cus(np.asarray(image))  
                boxes = results[0].boxes.xywhn.cpu().detach().numpy()
                classes = results[0].boxes.cls.cpu().detach().numpy()
                confidences = results[0].boxes.conf.cpu().detach().numpy()

                if len(classes) > 0:
                    draw_boxes(image, boxes, classes, confidences)
                else:
                    st.write("Nothing was found!")
            except Exception as e:
                st.write(f"Error during detection: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
