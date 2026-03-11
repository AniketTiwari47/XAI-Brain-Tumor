# ml_core.py
# Refactored core logic from helper2.py, adapted for Flask:
# Plotting functions now encode images to Base64 strings instead of displaying them.

import os
import io
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

# Suppress warnings
warnings.filterwarnings("ignore")

# Force CPU-only execution (useful for environments without a dedicated GPU)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_EXPLAINER_TYPE = "DeepExplainer"
except Exception:
    SHAP_AVAILABLE = False
    SHAP_EXPLAINER_TYPE = None
    print("⚠ SHAP not installed. SHAP explanations will be skipped.")

# -----------------------
# Config (MUST match your local file structure)
# -----------------------
# NOTE: Replace this with your actual, accessible path when running locally
BASE_DIR = r"C:\Users\KIIT\OneDrive\Desktop\Frontend\brain tumor\brain tumor"
IMG_SIZE = 128
CATEGORIES = ["no", "yes"]
RANDOM_STATE = 42
N_COMPONENTS = 300

CNN_MODEL_FILE = os.path.join(BASE_DIR, "cnn_brain_tumor_model_max.keras")
ANN_MODEL_FILE = os.path.join(BASE_DIR, "ann_brain_tumor.keras")
ML_PIPELINES_FILE = os.path.join(BASE_DIR, "ml_pipelines.joblib")
SCALER_PCA_FILE = os.path.join(BASE_DIR, "scaler_pca_for_ann.joblib")
X_TRAIN_PCA_SAMPLE_FILE = os.path.join(BASE_DIR, "X_train_pca_sample.joblib")

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("🔄 Initializing ML Core...")

# -----------------------
# Utility: Plot to Base64
# -----------------------
def plot_to_base64(fig):
    """Encodes a matplotlib figure to a base64 string for web display."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except Exception as e:
        print(f"Error encoding plot: {e}")
        return ""
    finally:
        plt.close(fig)

# -----------------------
# Global Artifacts Initialization (Run once)
# -----------------------
ml_pipelines = {}
scaler_pca = {}
scaler = None
pca = None
ann_model = None
cnn_model = None
CAM_TARGET_LAYER = None
explainer_ann = None
is_initialized = False

def initialize_models():
    """Load all models and artifacts."""
    global ml_pipelines, scaler_pca, scaler, pca, ann_model, cnn_model, CAM_TARGET_LAYER, explainer_ann, is_initialized

    if is_initialized:
        return True

    try:
        # Load ML Artifacts
        if not (os.path.exists(ML_PIPELINES_FILE) and os.path.exists(SCALER_PCA_FILE)):
            print("❌ ML Artifacts missing! Check BASE_DIR and file names.")
            return False
        ml_pipelines = joblib.load(ML_PIPELINES_FILE)
        scaler_pca = joblib.load(SCALER_PCA_FILE)
        scaler = scaler_pca["scaler"]
        pca = scaler_pca["pca"]
        
        # Load ANN Model
        if not os.path.exists(ANN_MODEL_FILE):
             print("❌ ANN Model missing! Check BASE_DIR and file names.")
             return False
        ann_model = load_model(ANN_MODEL_FILE)
        
        # Load CNN Model
        if not os.path.exists(CNN_MODEL_FILE):
             print("❌ CNN Model missing! Check BASE_DIR and file names.")
             return False
        cnn_model = load_model(CNN_MODEL_FILE)
        cnn_model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        # Identify CAM Target Layer
        conv_layers = [l for l in cnn_model.layers if isinstance(l, Conv2D)]
        if len(conv_layers) > 0:
            CAM_TARGET_LAYER = conv_layers[-1]
        else:
            CAM_TARGET_LAYER = None
            print("⚠ No Conv2D layer found; Score-CAM will be skipped.")

        # Initialize SHAP
        if SHAP_AVAILABLE and os.path.exists(X_TRAIN_PCA_SAMPLE_FILE):
            X_train_pca_sample = joblib.load(X_TRAIN_PCA_SAMPLE_FILE)
            if X_train_pca_sample.ndim == 1:
                X_train_pca_sample = np.expand_dims(X_train_pca_sample, 0)
            try:
                explainer_ann = shap.DeepExplainer(ann_model, X_train_pca_sample)
                # SHAP_EXPLAINER_TYPE is already set to DeepExplainer
            except Exception as e:
                try:
                    explainer_ann = shap.KernelExplainer(ann_model.predict, X_train_pca_sample)
                    global SHAP_EXPLAINER_TYPE
                    SHAP_EXPLAINER_TYPE = "KernelExplainer"
                except Exception as e_kernel:
                    print(f"⚠ Could not initialize SHAP explainer: {e_kernel}")
                    explainer_ann = None
        
        is_initialized = True
        print("✅ ML Core Initialization Complete.")
        return True

    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return False

# -----------------------
# Integrated Gradients (IG)
# -----------------------
def compute_integrated_gradients(model, input_image, target_index, steps=50):
    """
    input_image: tf.Tensor shaped (1,H,W,C)
    returns: numpy array attribution map (H,W,C)
    """
    baseline = tf.zeros_like(input_image)
    input_diff = input_image - baseline
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    scaled_inputs = tf.stack([baseline + a * input_diff for a in alphas], axis=0)
    if scaled_inputs.shape[1] == 1:
        scaled_inputs = tf.squeeze(scaled_inputs, axis=1)

    with tf.GradientTape() as tape:
        tape.watch(scaled_inputs)
        outputs = model(scaled_inputs)
        target_scores = outputs[:, target_index]

    grads = tape.gradient(target_scores, scaled_inputs)
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    integrated_grads = tf.reduce_sum(avg_grads, axis=0)
    input_diff_squeezed = tf.squeeze(input_diff, axis=0)
    ig = integrated_grads * input_diff_squeezed
    return ig.numpy()

def generate_integrated_gradients_plot(img_raw, ig_map, prediction, confidence, sigma=1.5):
    """Creates IG plot and returns base64 string."""
    ig_map = np.squeeze(ig_map)
    smoothed = gaussian_filter(ig_map, sigma=sigma) if 'gaussian_filter' in globals() else ig_map
    abs_map = np.abs(smoothed)
    vmax_signed = np.std(smoothed) * 2.0
    vmin_signed = -vmax_signed
    vmax_abs = np.percentile(abs_map, 95)
    
    attribution_info = f"IG shows focus on {prediction} tissue."
    if "yes" in prediction.lower():
        attribution_info = "Strong positive attribution (red) clustered over the tumor mass."
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    axes[0].set_title("Original MRI")
    axes[0].imshow(img_raw, cmap='gray'); axes[0].axis('off')
    
    axes[1].set_title("IG: Signed Attribution (Red = Positive)")
    axes[1].imshow(img_raw, cmap='gray')
    im2 = axes[1].imshow(smoothed, cmap='seismic', vmin=vmin_signed, vmax=vmax_signed, alpha=0.7)
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', shrink=0.7)
    
    axes[2].set_title("IG: Absolute Focus (Bright = Important)")
    axes[2].imshow(img_raw, cmap='gray')
    im3 = axes[2].imshow(abs_map, cmap='plasma', vmin=0, vmax=vmax_abs, alpha=0.7)
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], orientation='horizontal', shrink=0.7)
    
    fig.suptitle(f"CNN Prediction: *{prediction.upper()}* ({confidence:.2f}%) | {attribution_info}", fontsize=12, y=0.05)
    return plot_to_base64(fig)


# -----------------------
# Score-CAM (dynamic, visual, robust)
# -----------------------
def compute_score_cam_visual(model, img_array, target_layer, target_index, IMG_SIZE=128):
    """
    img_array: numpy or tf tensor shape (1,H,W,C), values in [0,1]
    target_layer: layer object from 'model' (must belong to same model)
    target_index: class index to score (int)
    returns: cam_norm (H,W) normalized [0,1], and optional error message (str)
    """
    if target_layer is None:
        return None, "Score-CAM skipped: target_layer is None"

    # build feature extractor from the same model graph
    try:
        fmap_model = Model(inputs=model.inputs, outputs=target_layer.output)
    except Exception as e:
        return None, f"Failed to build feature extractor: {e}"

    # ensure numpy array
    if isinstance(img_array, tf.Tensor):
        input_np = img_array.numpy()
    else:
        input_np = np.array(img_array)

    if input_np.ndim != 4:
        return None, f"Input must be shaped (1,H,W,C). Got {input_np.shape}"

    H_in, W_in = input_np.shape[1], input_np.shape[2]
    orig_img = input_np[0, :, :, :]
    single_channel = (orig_img.shape[-1] == 1)

    # feature maps
    try:
        fmap_output = fmap_model.predict(input_np, verbose=0)[0]  # (Hf,Wf,C)
    except Exception as e:
        return None, f"Feature model predict() failed: {e}"

    Hf, Wf, C = fmap_output.shape
    fmaps = fmap_output  # (Hf, Wf, C)

    weights = []
    # compute score for each fmap
    for i in range(C):
        fmap = fmaps[:, :, i].astype(np.float32)
        fmap = np.maximum(fmap, 0.0)
        
        if not np.isfinite(fmap).all() or fmap.max() == fmap.min():
            # zero mask
            mask_resized = np.zeros((H_in, W_in), dtype=np.float32)
        else:
            mask = (fmap - fmap.min()) / (fmap.max() - fmap.min())
            mask_resized = resize(mask, (H_in, W_in), anti_aliasing=True).astype(np.float32)

        # apply mask to original single channel image
        if single_channel:
            masked = (orig_img[:, :, 0] * mask_resized).reshape(1, H_in, W_in, 1).astype(np.float32)
        else:
            # for RGB multiply per-channel
            masked = (orig_img * mask_resized[..., np.newaxis]).reshape(1, H_in, W_in, orig_img.shape[-1]).astype(np.float32)

        # ensure finite
        if not np.isfinite(masked).all():
            weights.append(0.0)
            continue

        try:
            pred = model.predict(masked, verbose=0)[0]
            score = float(pred[target_index])
        except Exception:
            score = 0.0
        weights.append(score)

    weights = np.array(weights, dtype=np.float32)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    denom = weights.sum()
    if denom <= 1e-8:
        weights = np.ones_like(weights, dtype=np.float32) / max(1, len(weights))
    else:
        weights = weights / (denom + 1e-8)

    # weighted combination
    cam = np.zeros((Hf, Wf), dtype=np.float32)
    for i in range(C):
        cam += weights[i] * fmaps[:, :, i]

    cam = np.maximum(cam, 0.0)
    cam_resized = resize(cam, (H_in, W_in), anti_aliasing=True).astype(np.float32)
    cam_resized = np.nan_to_num(cam_resized)

    # normalize
    mn, mx = cam_resized.min(), cam_resized.max()
    if mx - mn <= 1e-8:
        cam_norm = np.zeros_like(cam_resized)
    else:
        cam_norm = (cam_resized - mn) / (mx - mn + 1e-8)

    return cam_norm, None # <-- CORRECTED: Returns normalized CAM map and None for error (if successful)


def generate_score_cam_plot(img_raw, cam_map, prediction, confidence):
    """Creates Score-CAM plot and returns base64 string."""
    H, W = img_raw.shape
    
    # Create heatmap (RGB)
    heatmap_uint8 = np.uint8(255 * cam_map)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Build overlay: convert img_raw to RGB uint8
    orig_rgb = np.uint8(np.stack([img_raw]*3, axis=-1))
    
    # Create overlay
    overlay = cv2.addWeighted(orig_rgb, 0.55, heatmap_color, 0.45, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    axes[0].set_title("Original MRI")
    axes[0].imshow(img_raw, cmap='gray'); axes[0].axis('off')

    axes[1].set_title("Score-CAM Heatmap")
    im2 = axes[1].imshow(cam_map, cmap='jet')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', shrink=0.6)

    axes[2].set_title("Overlay")
    axes[2].imshow(overlay); axes[2].axis('off')

    fig.suptitle(f"CNN Prediction: *{prediction.upper()}* ({confidence:.2f}%) | Score-CAM", fontsize=12, y=0.05)
    return plot_to_base64(fig)


# -----------------------
# SHAP helper (ANN)
# -----------------------
def generate_shap_plot(img_pca, prediction, confidence):
    """Creates SHAP plot and returns base64 string."""
    global explainer_ann, SHAP_EXPLAINER_TYPE
    if explainer_ann is None:
        return "⚠ SHAP skipped (explainer not initialized)."
        
    sample = img_pca.reshape(1, -1).astype(np.float32)
    
    # Predict the class index for use in indexing SHAP values
    pred_proba = ann_model.predict(sample, verbose=0)[0]
    class_index = int(np.argmax(pred_proba))

    try:
        shap_vals_raw = explainer_ann.shap_values(sample)
    except Exception as e:
        return f"❌ SHAP computation error: {e}"
        
    try:
        if isinstance(shap_vals_raw, list):
            class_shap_arr = np.array(shap_vals_raw[class_index])
        else:
            class_shap_arr = np.array(shap_vals_raw)
        
        if class_shap_arr.ndim > 1 and class_shap_arr.shape[0] > 1:
            class_shap = class_shap_arr[0].flatten()
        else:
            class_shap = class_shap_arr.flatten()
            
        input_size = sample.shape[1]
        if class_shap.size != input_size:
            class_shap = class_shap[:input_size]

    except Exception as e:
        return f"❌ Could not extract SHAP vector from array: {e}"
    
    base = getattr(explainer_ann, "expected_value", None)
    base_val = None
    if base is not None:
        try:
            base_arr = np.array(base)
            idx = class_index if base_arr.ndim > 0 and len(base_arr) > 1 else 0
            base_val = float(base_arr[idx])
        except Exception:
            base_val = None
            
    feature_names = [f"PC{i}" for i in range(sample.shape[1])]
    
    try:
        explanation = shap.Explanation(values=class_shap, base_values=base_val, data=sample[0], feature_names=feature_names)
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        # Use waterfall plot
        try:
            shap.plots.waterfall(explanation, show=False)
        except Exception:
            shap.waterfall_plot(explanation, show=False)
        
        fig.suptitle(f"ANN Prediction: *{prediction.upper()}* ({confidence:.2f}%) | SHAP Analysis", fontsize=12)
        
        return plot_to_base64(fig)
        
    except Exception as e:
        return f"❌ Error plotting SHAP: {e}"


# -----------------------
# Preprocess single image
# -----------------------
def preprocess_image(img_path):
    """Loads and preprocesses a single image for all models."""
    global scaler, pca
    if not scaler or not pca:
        raise Exception("Scaler and PCA models not loaded. Initialization failed.")
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
    
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # CNN Input (1, H, W, 1), normalized
    img_cnn_np = np.expand_dims(img_resized / 255.0, axis=(0, -1)).astype(np.float32)
    img_cnn = tf.constant(img_cnn_np)
    
    # ANN/ML Input (1, N_features), scaled and PCA-reduced
    img_flat = img_resized.flatten().reshape(1, -1).astype(np.float32)
    img_scaled = scaler.transform(img_flat)
    img_pca = pca.transform(img_scaled)
    
    # img_raw is the original resized grayscale for plotting base
    return img_flat, img_pca, img_cnn, img_resized

# -----------------------
# Predict wrapper for Flask
# -----------------------
def predict_brain_tumor_web(img_path):
    """The main prediction and XAI function for the web app."""
    
    if not initialize_models():
        return {"error": "Model initialization failed. Check server logs for missing files in BASE_DIR."}
        
    try:
        img_flat, img_pca, img_cnn, img_raw = preprocess_image(img_path)
    except Exception as e:
        return {"error": f"Preprocessing error: {e}"}

    results = {}
    
    # Classical ML
    ml_results = {}
    for name, pipe in ml_pipelines.items():
        try:
            p = pipe.predict(img_flat)[0]
            ml_results[name] = CATEGORIES[int(p)]
        except Exception as e:
            ml_results[name] = f"Error: {e}"
    results["classical_ml"] = ml_results

    # ANN
    try:
        ann_proba = ann_model.predict(img_pca, verbose=0)[0]
        ann_class = int(np.argmax(ann_proba))
        ann_label = CATEGORIES[ann_class]
        ann_conf = ann_proba[ann_class] * 100
        ann_conf = min(ann_conf, 99.99) # Clamp for display
        results["ANN_pred"] = ann_label
        results["ANN_conf"] = f"{ann_conf:.2f}"
    except Exception as e:
        results["ANN_pred"] = "Error"
        results["ANN_conf"] = "N/A"

    # CNN
    try:
        cnn_proba = cnn_model.predict(img_cnn, verbose=0)[0]
        cnn_class = int(np.argmax(cnn_proba))
        cnn_label = CATEGORIES[cnn_class]
        cnn_conf = cnn_proba[cnn_class] * 100
        cnn_conf = min(cnn_conf, 99.99) # Clamp for display
        results["CNN_pred"] = cnn_label
        results["CNN_conf"] = f"{cnn_conf:.2f}"
    except Exception as e:
        results["CNN_pred"] = "Error"
        results["CNN_conf"] = "N/A"
    
    # 1. Integrated Gradients
    try:
        ig_map = compute_integrated_gradients(cnn_model, img_cnn, target_index=cnn_class, steps=50)
        ig_plot_base64 = generate_integrated_gradients_plot(img_raw, ig_map, cnn_label, cnn_conf)
        results["ig_plot"] = ig_plot_base64
    except Exception as e:
        print(f"IG Error: {e}")
        results["ig_plot"] = "Error generating Integrated Gradients plot."

    # 2. Score-CAM
    if CAM_TARGET_LAYER is not None:
        try:
            # CORRECTED: Only unpack cam_map and err
            cam_map, err = compute_score_cam_visual(cnn_model, img_cnn, CAM_TARGET_LAYER, target_index=cnn_class, IMG_SIZE=IMG_SIZE)
            if err:
                results["cam_plot"] = f"Score-CAM computation error: {err}"
            else:
                cam_plot_base64 = generate_score_cam_plot(img_raw, cam_map, cnn_label, cnn_conf)
                results["cam_plot"] = cam_plot_base64
        except Exception as e:
            print(f"Score-CAM Error: {e}")
            results["cam_plot"] = "Error generating Score-CAM plot."
    else:
        results["cam_plot"] = "Score-CAM skipped (Target Layer not found)."

    # 3. SHAP
    try:
        shap_plot_base64 = generate_shap_plot(img_pca, ann_label, float(results.get("ANN_conf", 0)))
        results["shap_plot"] = shap_plot_base64
    except Exception as e:
        print(f"SHAP Error: {e}")
        results["shap_plot"] = "Error generating SHAP plot."

    return results

if __name__ == '__main__':
    initialize_models()
    # This block is for testing ml_core independently
    # Example: print(predict_brain_tumor_web("path_to_a_test_image.jpg"))