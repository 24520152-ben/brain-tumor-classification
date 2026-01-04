import os
import io
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras import applications
import matplotlib.cm as cm
import uvicorn

app = FastAPI(title="Brain Tumor Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
IMG_SIZE = (224, 224)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
path_vgg = os.path.join(project_root, 'models', 'VGG19_Best_Weights', 'VGG19_block_1_fold_1.keras')
path_mobilenet = os.path.join(project_root, 'models', 'MobileNet_Best_Weights', 'MobileNet_block_1_fold_1.keras')

try:
    model_vgg = tf.keras.models.load_model(path_vgg, compile=False)
    model_mobilenet = tf.keras.models.load_model(path_mobilenet, compile=False)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model_vgg, model_mobilenet = None, None

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output_shape if hasattr(layer, 'output_shape') else layer.output.shape
            if output_shape is not None and len(output_shape) == 4:
                return layer.name
        except: continue
    raise ValueError("Feature map layer not found.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    x = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_output = None
        current_x = x
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer): continue
            current_x = layer(current_x)
            if layer.name == last_conv_layer_name:
                conv_output = current_x
                tape.watch(conv_output)
        preds = current_x
        if pred_index is None: pred_index = tf.argmax(preds[0])
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

@app.get("/")
async def read_index():
    index_path = os.path.join(current_dir, "index.html")
    return FileResponse(index_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model_vgg is None or model_mobilenet is None:
        raise HTTPException(status_code=500, detail="Models not loaded on server.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_batch = tf.expand_dims(img_array, 0)

        tta_steps = 5
        predictions_sum = np.zeros((1, len(CLASS_NAMES)))
        vgg_score_sum = np.zeros((1, len(CLASS_NAMES)))
        mobilenet_score_sum = np.zeros((1, len(CLASS_NAMES)))

        for _ in range(tta_steps):
            aug_img = tta_augmentation(img_batch)
            input_vgg = applications.vgg19.preprocess_input(tf.identity(aug_img))
            input_mobilenet = applications.mobilenet.preprocess_input(tf.identity(aug_img))
            
            vgg_score_sum += model_vgg.predict(input_vgg, verbose=0)
            mobilenet_score_sum += model_mobilenet.predict(input_mobilenet, verbose=0)

        avg_vgg = vgg_score_sum / tta_steps
        avg_mobilenet = mobilenet_score_sum / tta_steps
        final_pred = (avg_vgg + avg_mobilenet) / 2.0
        
        top_index = np.argmax(final_pred[0])
        confidence = float(final_pred[0][top_index])
        label = CLASS_NAMES[top_index]

        if np.max(avg_vgg) > np.max(avg_mobilenet):
            target_model, target_input, model_name = model_vgg, applications.vgg19.preprocess_input(tf.identity(img_batch)), "VGG19"
        else:
            target_model, target_input, model_name = model_mobilenet, applications.mobilenet.preprocess_input(tf.identity(img_batch)), "MobileNet"

        heatmap = make_gradcam_heatmap(target_input, target_model, pred_index=top_index)
        overlay_img = create_overlay_image(heatmap, img_array)

        buffered = io.BytesIO()
        overlay_img.save(buffered, format="JPEG")
        overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "prediction": label,
            "confidence": confidence,
            "used_model_for_gradcam": model_name,
            "overlay_image_base64": overlay_base64
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)