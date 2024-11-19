import streamlit as st
import io
import base64
from PIL import Image, ImageDraw


from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="1yftlULBWV4Wy0xGw58x"
)


def nail_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongle")

    if "images" not in st.session_state:
        st.session_state["images"] = None

    images = st.file_uploader(
        "Upload vos images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
    )

    if images is not None:
        st.session_state["images"] = images

    st.button("Afficher les images", on_click=show_images, disabled=len(images) == 0)

    st.button(
        "Détecter les ongles", on_click=on_images_uploaded, disabled=len(images) == 0
    )

    if "result_images" in st.session_state:
        show_results()


def show_images():
    images = st.session_state["images"]
    if images:
        cols = st.columns(len(images))
        for col, image in zip(cols, images):
            col.image(image, caption="Uploaded Image.", use_column_width=True)


def show_results():
    for img_base64 in st.session_state["result_images"]:
        st.markdown(
            f'<img src="data:image/jpeg;base64,{img_base64}">',
            unsafe_allow_html=True,
        )


def on_images_uploaded():

    uploaded_files = st.session_state["images"]
    if uploaded_files is not None:
        results = []
        result_images = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()

            response = CLIENT.infer(img_base64, model_id="nails-detection-hs7q7/1")
            results.append(response)

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)
            for prediction in response["predictions"]:
                x = prediction["x"]
                y = prediction["y"]
                width = prediction["width"]
                height = prediction["height"]
                left = x - width / 2
                top = y - height / 2
                right = x + width / 2
                bottom = y + height / 2
                draw.rectangle([left, top, right, bottom], outline="red", width=2)

            # Save the image with bounding boxes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()
            result_images.append(img_base64)

        st.session_state["result"] = results
        st.session_state["result_images"] = result_images
