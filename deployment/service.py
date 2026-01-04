import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from tensorflow.keras import applications
from PIL import Image
import io
import base64
from config import PATH_VGG, PATH_MOBILENET, CLASS_NAMES, IMG_SIZE

class Ensemble_VGG19_MobileNet:
    def __init__(self):
        try:
            self.model_vgg = tf.keras.models.load_model(PATH_VGG, compile=False)
            self.model_mobilenet = tf.keras.models.load_model(PATH_MOBILENET, compile=False)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def get_last_conv_layer_name(self, model):
        for layer in reversed(model.layers):
            try:
                output_shape = layer.output_shape if hasattr(layer, 'output_shape') else layer.output.shape
                if output_shape is not None and len(output_shape) == 4:
                    return layer.name
            except: continue
        raise ValueError("Feature map layer not found.")
    
    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name=None, pred_index=None):
        if last_conv_layer_name is None:
            last_conv_layer_name = self.get_last_conv_layer_name(model)

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
    
    def create_overlay_image(self, heatmap, original_img_array):
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((original_img_array.shape[1], original_img_array.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 0.4 + original_img_array
        return tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    def tta_augmentation(self, img_tensor):
        img = tf.image.random_flip_left_right(img_tensor)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img

    def predict_and_explain(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_batch = tf.expand_dims(img_array, 0)

        tta_steps = 5
        predictions_sum = np.zeros((1, len(CLASS_NAMES)))
        vgg_score_sum = np.zeros((1, len(CLASS_NAMES)))
        mobilenet_score_sum = np.zeros((1, len(CLASS_NAMES)))

        for _ in range(tta_steps):
            aug_img = self.tta_augmentation(img_batch)
            input_vgg = applications.vgg19.preprocess_input(tf.identity(aug_img))
            input_mobilenet = applications.mobilenet.preprocess_input(tf.identity(aug_img))
            
            vgg_score_sum += self.model_vgg.predict(input_vgg, verbose=0)
            mobilenet_score_sum += self.model_mobilenet.predict(input_mobilenet, verbose=0)

        avg_vgg = vgg_score_sum / tta_steps
        avg_mobilenet = mobilenet_score_sum / tta_steps
        final_pred = (avg_vgg + avg_mobilenet) / 2.0
        
        top_index = np.argmax(final_pred[0])
        confidence = float(final_pred[0][top_index])
        label = CLASS_NAMES[top_index]

        if np.max(avg_vgg) > np.max(avg_mobilenet):
            target_model, target_input, model_name = self.model_vgg, applications.vgg19.preprocess_input(tf.identity(img_batch)), "VGG19"
        else:
            target_model, target_input, model_name = self.model_mobilenet, applications.mobilenet.preprocess_input(tf.identity(img_batch)), "MobileNet"

        heatmap = self.make_gradcam_heatmap(target_input, target_model, pred_index=top_index)
        overlay_img = self.create_overlay_image(heatmap, img_array)

        buffered = io.BytesIO()
        overlay_img.save(buffered, format="JPEG")
        overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "prediction": label,
            "confidence": confidence,
            "used_model_for_gradcam": model_name,
            "overlay_image_base64": overlay_base64
        }
    
model = Ensemble_VGG19_MobileNet()