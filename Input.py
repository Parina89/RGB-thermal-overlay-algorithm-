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

def resize_images_to_match(img1, img2):
    """Resize both images to exact same dimensions for 100% overlap."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Use the larger dimensions to avoid losing detail
    target_h = max(h1, h2)
    target_w = max(w1, w2)
    
    # Resize both images to exact same size
    img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    return img1_resized, img2_resized

def process_overlay(img1, img2, blend_mode, opacity):
    """Process the overlay of two images with 100% overlap."""
    # Resize both images to exact same dimensions
    img1_resized, img2_resized = resize_images_to_match(img1, img2)
    
    # Normalize to [0, 1]
    img1_norm = img1_resized.astype(np.float32) / 255.0
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
    
    return result, img1_resized, img2_resized

# Sidebar controls
st.sidebar.header("⚙️ Settings")
blend_mode = st.sidebar.selectbox(
    "Blend Mode",
    ['Normal', 'Multiply', 'Screen', 'Overlay', 'Difference', 'Add', 'Soft Light'],
    index=0
)
opacity = st.sidebar.slider("Image 2 Opacity", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Tips:
- **Opacity 0.5** = 50% each image
- **Opacity 0.0** = Only Image 1
- **Opacity 1.0** = Full blend effect
""")

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
    
    # Get original dimensions
    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]
    
    # Process overlay with 100% overlap
    result, img1_resized, img2_resized = process_overlay(img1_np, img2_np, blend_mode, opacity)
    
    # Get final dimensions
    final_h, final_w = result.shape[:2]
    
    # Display image info
    st.markdown("---")
    st.subheader("📐 Image Dimensions (100% Overlap)")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Image 1 Original", f"{w1} x {h1} px")
    with info_col2:
        st.metric("Image 2 Original", f"{w2} x {h2} px")
    with info_col3:
        st.metric("Output Size", f"{final_w} x {final_h} px")
    
    st.success(f"✅ Both images resized to **{final_w} x {final_h}** pixels for 100% overlap")
    
    # Display original images (resized)
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img1_resized, caption="Image 1 (Thermal) - Resized", use_container_width=True)
    
    with col2:
        st.image(img2_resized, caption="Image 2 (RGB) - Resized", use_container_width=True)
    
    # Display result
    st.markdown("---")
    st.subheader(f"🎨 100% Overlay Result ({blend_mode}, {int(opacity*100)}% opacity)")
    st.image(result, caption="Image 1 + Image 2 (100% Overlap)", use_container_width=True)
    
    # Download button
    result_pil = Image.fromarray(result)
    buf = io.BytesIO()
    result_pil.save(buf, format='PNG')
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Overlay Result",
        data=buf,
        file_name="overlay_100_percent.png",
        mime="image/png"
    )
    
    # Show all blend modes comparison
    if st.checkbox("🔍 Show all blend modes comparison"):
        st.markdown("---")
        st.subheader("📊 All Blend Modes (100% Overlap)")
        
        modes = ['Normal', 'Multiply', 'Screen', 'Overlay', 'Difference', 'Add', 'Soft Light']
        cols = st.columns(4)
        
        for i, mode in enumerate(modes):
            with cols[i % 4]:
                result_mode, _, _ = process_overlay(img1_np, img2_np, mode, opacity)
                st.image(result_mode, caption=mode, use_container_width=True)

else:
    st.info("👆 Please upload both Image 1 (Thermal) and Image 2 (RGB) to see the 100% overlay result.")
    
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use:
    1. Upload your **thermal image** as Image 1
    2. Upload your **RGB image** as Image 2
    3. Both images will be **automatically resized** to match for 100% overlap
    4. Adjust **blend mode** and **opacity** in the sidebar
    5. Download the result
    
    ### 🎨 Blend Modes:
    - **Normal** - Simple transparency blend
    - **Screen** - Brightens, reveals hot spots
    - **Multiply** - Darkens, shows thermal patterns
    - **Overlay** - Combines contrast from both
    - **Difference** - Highlights temperature variations
    - **Add** - Additive blending
    - **Soft Light** - Subtle contrast enhancement
    
    ### ✅ 100% Overlap Feature:
    - Both images are automatically resized to the **same dimensions**
    - Every pixel from Image 1 aligns with corresponding pixel in Image 2
    - No cropping or misalignment
    """)
