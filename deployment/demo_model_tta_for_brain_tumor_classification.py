import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import applications
import matplotlib.cm as cm

st.set_page_config(layout="wide", page_title="Brain Tumor Analysis")

st.markdown("""
<style>
    .main-title {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        color: inherit; 
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        height: 850px; /* Chiều cao cố định */
        padding: 20px;
        border-radius: 10px;
        
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        gap: 15px;
    }

    div[data-testid="stFileUploader"], .stButton {
        width: 100%;
    }

    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
        margin-top: auto;
    }
    
    div[data-testid="stVerticalBlock"] button[kind="primary"] {
        background-color: #ff4b4b;
        color: white;
        border: none;
    }

    .result-text {
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
        text-align: center;
        width: 100%;
    }
    
    .confidence-text {
        font-size: 16px;
        opacity: 0.8;
        text-align: center;
        margin-bottom: 10px;
        width: 100%;
    }

    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    
    div[data-testid="stImage"] > img {
        object-fit: contain;
    }

    .placeholder-text {
        opacity: 0.5;
        font-style: italic;
        text-align: center;
        margin-top: 250px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
IMG_SIZE = (224, 224)

@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    path_vgg = os.path.join(project_root, 'models', 'VGG19_Best_Weights', 'VGG19_block_1_fold_1.keras')
    path_mobilenet = os.path.join(project_root, 'models', 'MobileNet_Best_Weights', 'MobileNet_block_1_fold_1.keras')

    try:
        model1 = tf.keras.models.load_model(path_vgg, compile=False)
        model2 = tf.keras.models.load_model(path_mobilenet, compile=False)
        return model1, model2
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return None, None

model_vgg, model_mobilenet = load_models()

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            elif hasattr(layer, 'output'):
                output_shape = layer.output.shape
            else:
                continue
            if output_shape is None: continue
            if len(output_shape) == 4:
                return layer.name
        except (AttributeError, ValueError):
            continue
    raise ValueError("Không tìm thấy layer feature map (4D) nào trong model!")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    x = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_output = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_overlay_image(heatmap, original_img_array):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img_array.shape[1], original_img_array.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + original_img_array
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

def tta_augmentation(img_tensor):
    img = tf.image.random_flip_left_right(img_tensor)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img

def predict_and_gradcam(image, tta_steps=5):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = tf.expand_dims(img_array, 0)
    
    predictions_sum = np.zeros((1, len(CLASS_NAMES)))
    vgg_score_sum = np.zeros((1, len(CLASS_NAMES)))
    mobilenet_score_sum = np.zeros((1, len(CLASS_NAMES)))

    for _ in range(tta_steps):
        aug_img = tta_augmentation(img_batch)
        input_vgg = applications.vgg19.preprocess_input(tf.identity(aug_img))
        input_mobilenet = applications.mobilenet.preprocess_input(tf.identity(aug_img))
        
        pred1 = model_vgg.predict(input_vgg, verbose=0)
        pred2 = model_mobilenet.predict(input_mobilenet, verbose=0)
        
        vgg_score_sum += pred1
        mobilenet_score_sum += pred2
        predictions_sum += (pred1 + pred2) / 2.0

    final_pred = predictions_sum / tta_steps
    avg_vgg = vgg_score_sum / tta_steps
    avg_mobilenet = mobilenet_score_sum / tta_steps
    max_vgg = np.max(avg_vgg)
    max_mobile = np.max(avg_mobilenet)
    top_pred_index = np.argmax(final_pred[0])

    clean_input_vgg = applications.vgg19.preprocess_input(tf.identity(img_batch))
    clean_input_mobilenet = applications.mobilenet.preprocess_input(tf.identity(img_batch))
    
    heatmap = None
    target_model_name = ""

    if max_vgg > max_mobile:
        target_model_name = "VGG19"
        heatmap = make_gradcam_heatmap(clean_input_vgg, model_vgg, pred_index=top_pred_index)
    else:
        target_model_name = "MobileNet"
        heatmap = make_gradcam_heatmap(clean_input_mobilenet, model_mobilenet, pred_index=top_pred_index)
        
    overlay = create_overlay_image(heatmap, img_array)
    return final_pred, overlay, target_model_name

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

st.markdown('<div class="main-title">BRAIN TUMOR CLASSIFICATION AI</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### Upload MRI Image")
    
    with st.container(border=True):
        upload_img = st.file_uploader("", type=["png", "jpg", "jpeg"], key="uploader")
        
        if upload_img is not None:
            image = Image.open(upload_img).convert('RGB')
            display_img = image.resize((512, 512))

            st.image(display_img, caption="Original Image (512x512)")

            if st.button("Analyze Image", type="primary"):
                with st.spinner('Processing...'):
                    ensemble_pred, overlay_img, used_model = predict_and_gradcam(image, tta_steps=5)
                    
                    class_index = np.argmax(ensemble_pred, axis=1)[0]
                    confidence = ensemble_pred[0][class_index]
                    predicted_label = CLASS_NAMES[class_index]
                    
                    st.session_state.analyzed = True
                    st.session_state.overlay_img = overlay_img.resize((512, 512))
                    st.session_state.predicted_label = predicted_label
                    st.session_state.confidence = confidence
                    st.session_state.used_model = used_model
        else:
            st.markdown('<div class="placeholder-text">Please upload an image to start</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("### Analysis Results")
    
    with st.container(border=True):
        if st.session_state.analyzed:
            st.markdown('<div style="height: 105px;"></div>', unsafe_allow_html=True)
            
            st.image(st.session_state.overlay_img, caption=f"Explainable AI ({st.session_state.used_model})")
            
            st.markdown(f"""
                <div class="result-text">Prediction: <span style='color:#3b82f6'>{st.session_state.predicted_label}</span></div>
                <div class="confidence-text">Confidence: <b>{st.session_state.confidence:.2%}</b></div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
                <div style='height: 100%; display: flex; align-items: center; justify-content: center; opacity: 0.3;'>
                    <div style='border: 2px dashed #ddd; border-radius: 10px; width: 100%; height: 512px; display: flex; align-items: center; justify-content: center;'>
                        Result will appear here
                    </div>
                </div>
            """, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)