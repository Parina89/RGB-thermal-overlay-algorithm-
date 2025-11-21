import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Thermal Image Overlay", layout="wide")

st.title("🌡️ Thermal & RGB Image Overlay Tool")
st.markdown("**Image 1:** Thermal Image (Base) | **Image 2:** RGB Image (Overlay)")

# Blend functions using NumPy
def blend_normal(img1, img2, opacity):
    return img1 * (1 - opacity) + img2 * opacity

def blend_multiply(img1, img2, opacity):
    blended = img1 * img2
    return img1 * (1 - opacity) + blended * opacity

def blend_screen(img1, img2, opacity):
    blended = 1 - (1 - img1) * (1 - img2)
    return img1 * (1 - opacity) + blended * opacity

def blend_overlay(img1, img2, opacity):
    mask = img1 < 0.5
    blended = np.where(
        mask,
        2 * img1 * img2,
        1 - 2 * (1 - img1) * (1 - img2)
    )
    return img1 * (1 - opacity) + blended * opacity

def blend_difference(img1, img2, opacity):
    blended = np.abs(img1 - img2)
    return img1 * (1 - opacity) + blended * opacity

def blend_add(img1, img2, opacity):
    blended = np.clip(img1 + img2, 0, 1)
    return img1 * (1 - opacity) + blended * opacity

def blend_soft_light(img1, img2, opacity):
    mask = img2 < 0.5
    blended = np.where(
        mask,
        img1 - (1 - 2 * img2) * img1 * (1 - img1),
        img1 + (2 * img2 - 1) * (np.sqrt(img1) - img1)
    )
    return img1 * (1 - opacity) + blended * opacity

def process_overlay(img1, img2, blend_mode, opacity):
    """Process the overlay of two images."""
    # Resize img2 to match img1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Normalize to [0, 1]
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2_resized.astype(np.float32) / 255.0
    
    # Apply blend mode
    blend_functions = {
        'Normal': blend_normal,
        'Multiply': blend_multiply,
        'Screen': blend_screen,
        'Overlay': blend_overlay,
        'Difference': blend_difference,
        'Add': blend_add,
        'Soft Light': blend_soft_light
    }
    
    result = blend_functions[blend_mode](img1_norm, img2_norm, opacity)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

# Sidebar controls
st.sidebar.header("⚙️ Settings")
blend_mode = st.sidebar.selectbox(
    "Blend Mode",
    ['Normal', 'Multiply', 'Screen', 'Overlay', 'Difference', 'Add', 'Soft Light']
)
opacity = st.sidebar.slider("Image 2 Opacity", 0.0, 1.0, 0.5, 0.01)

# File uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Image 1 (Thermal)")
    uploaded_file1 = st.file_uploader("Upload thermal image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], key="img1")

with col2:
    st.subheader("📤 Image 2 (RGB)")
    uploaded_file2 = st.file_uploader("Upload RGB image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], key="img2")

# Process images
if uploaded_file1 is not None and uploaded_file2 is not None:
    # Load images
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)
    
    # Convert to numpy arrays (RGB)
    img1_np = np.array(image1.convert('RGB'))
    img2_np = np.array(image2.convert('RGB'))
    
    # Display original images
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image1, caption="Image 1 (Thermal)", use_container_width=True)
    
    with col2:
        st.image(image2, caption="Image 2 (RGB)", use_container_width=True)
    
    # Process overlay
    result = process_overlay(img1_np, img2_np, blend_mode, opacity)
    
    # Display result
    st.markdown("---")
    st.subheader(f"🎨 Overlay Result ({blend_mode}, {int(opacity*100)}% opacity)")
    st.image(result, caption="Image 1 + Image 2 Overlay", use_container_width=True)
    
    # Download button
    result_pil = Image.fromarray(result)
    buf = io.BytesIO()
    result_pil.save(buf, format='PNG')
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Result",
        data=buf,
        file_name="overlay_result.png",
        mime="image/png"
    )
    
    # Show all blend modes comparison
    if st.checkbox("Show all blend modes comparison"):
        st.markdown("---")
        st.subheader("📊 All Blend Modes Comparison")
        
        modes = ['Normal', 'Multiply', 'Screen', 'Overlay', 'Difference', 'Add', 'Soft Light']
        cols = st.columns(4)
        
        for i, mode in enumerate(modes):
            with cols[i % 4]:
                result_mode = process_overlay(img1_np, img2_np, mode, opacity)
                st.image(result_mode, caption=mode, use_container_width=True)

else:
    st.info("👆 Please upload both Image 1 (Thermal) and Image 2 (RGB) to see the overlay result.")
    
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use:
    1. Upload your **thermal image** as Image 1
    2. Upload your **RGB image** as Image 2
    3. Adjust **blend mode** and **opacity** in the sidebar
    4. Download the result
    
    ### 🎨 Blend Modes:
    - **Normal** - Simple transparency blend
    - **Screen** - Brightens, reveals hot spots
    - **Multiply** - Darkens, shows thermal patterns
    - **Overlay** - Combines contrast from both
    - **Difference** - Highlights temperature variations
    - **Add** - Additive blending
    - **Soft Light** - Subtle contrast enhancement
    """)
