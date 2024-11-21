import streamlit as st
import io
import base64
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

custom_configuration = InferenceConfiguration(confidence_threshold=0.90)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="Gk3YGuQPvdT8XhCh9HCs"
).configure(custom_configuration)


def nail_page():
    """
    # Nail Page Flow

    🟢 init session variable to store uploaded images

    📂 Show file input & retrieve files once uploaded

    📝 User clicks on buttons which perform a callback

    📈 Show results once `result_images` session state has been updated


    ## Buttons
    1. show images
        2. Call roboflow api on click
    """
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongle")

    # init session variable to store uploaded images
    if "images" not in st.session_state:
        st.session_state["images"] = None

    # file input
    images = st.file_uploader(
        "Upload vos images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
    )

    if images is not None:
        st.session_state["images"] = images

    ### Control buttons ###

    # 1. show images
    st.button(
        "Afficher les images",
        on_click=show_images,
        disabled=len(images) == 0,
        # label="show uploaded images",
    )

    # grab user input for confidence threshold
    user_threshold = st.number_input(
        "Confidence Threshold", value=0.90, step=0.01, max_value=1.0
    )
    st.session_state["confidence_threshold"] = user_threshold

    # 2. Call roboflow api on click (on_click callback)
    st.button(
        "Détecter les ongles",
        on_click=on_images_uploaded,
        disabled=len(images) == 0,
        # label="Process and detect nails from uploaded images",
    )

    if "result_images" in st.session_state:
        show_prediction_results()


def show_images():
    """
    Callback for button `afficher les images`
    """
    images = st.session_state["images"]
    if images:
        cols = st.columns(len(images))
        for col, image in zip(cols, images):
            col.image(image, caption="Uploaded Image.", use_column_width=True)


def show_prediction_results():
    """
    Update GUI with modified images that show predictions.
    """
    for img_base64 in st.session_state["result_images"]:
        st.markdown(
            f'<img src="data:image/jpeg;base64,{img_base64}">',
            unsafe_allow_html=True,
        )


def on_images_uploaded():
    """
    For each uploaded image:

    1. Call roboflow model
    2. draw prediction from inference response predictions
    3. save result to `result_images` list
    """

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

            threshold = st.session_state["confidence_threshold"]

            if threshold:
                CLIENT.configure(InferenceConfiguration(confidence_threshold=threshold))

            response = CLIENT.infer(img_base64, model_id="nail-detection-iqigg/1")
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
