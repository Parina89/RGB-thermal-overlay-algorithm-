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

def align_images_same_position(img1, img2):
    """
    Align both images to exact same position and size.
    Uses feature matching for precise alignment.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Target size - use Image 1 as reference
    target_h, target_w = h1, w1
    
    # Resize Image 2 to match Image 1 exactly
    img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
    
    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) > 10:
            # Get matched points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
            
            # Find homography matrix
            M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Warp Image 2 to align with Image 1
                img2_aligned = cv2.warpPerspective(img2_resized, M, (target_w, target_h))
                return img1, img2_aligned, True
    
    # If feature matching fails, return resized images
    return img1, img2_resized, False

def process_overlay(img1, img2, blend_mode, opacity, use_alignment):
    """Process the overlay with same position alignment."""
    
    if use_alignment:
        img1_final, img2_final, aligned = align_images_same_position(img1, img2)
    else:
        # Simple resize without alignment
        h1, w1 = img1.shape[:2]
        img1_final = img1
        img2_final = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)
        aligned = False
    
    # Normalize to [0, 1]
    img1_norm = img1_final.astype(np.float32) / 255.0
    img2_norm = img2_final.astype(np.float32) / 255.0
    
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
    
    return result, img1_final, img2_final, aligned

# Sidebar controls
st.sidebar.header("⚙️ Settings")

blend_mode = st.sidebar.selectbox(
    "Blend Mode",
    ['Normal', 'Multiply', 'Screen', 'Overlay', 'Difference', 'Add', 'Soft Light'],
    index=0
)

opacity = st.sidebar.slider("Image 2 Opacity", 0.0, 1.0, 0.5, 0.01)

use_alignment = st.sidebar.checkbox("🎯 Auto-align images (Feature Matching)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Alignment Info:
- **Auto-align ON**: Uses feature matching to align images precisely
- **Auto-align OFF**: Simple resize only
""")

# File uploaders in same row
st.markdown("---")
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
    
    # Process overlay
    result, img1_final, img2_final, was_aligned = process_overlay(
        img1_np, img2_np, blend_mode, opacity, use_alignment
    )
    
    # Get dimensions
    final_h, final_w = result.shape[:2]
    
    # Alignment status
    st.markdown("---")
    if use_alignment and was_aligned:
        st.success(f"✅ Images aligned to same position using feature matching ({final_w} x {final_h} px)")
    elif use_alignment and not was_aligned:
        st.warning(f"⚠️ Feature matching failed - using simple resize ({final_w} x {final_h} px)")
    else:
        st.info(f"ℹ️ Simple resize applied ({final_w} x {final_h} px)")
    
    # Display both input images side by side (same position)
    st.markdown("---")
    st.subheader("📍 Input Images (Same Position & Size)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1_final, caption="Image 1 (Thermal)", use_container_width=True)
    with col2:
        st.image(img2_final, caption="Image 2 (RGB) - Aligned", use_container_width=True)
    
    # Display overlay result
    st.markdown("---")
    st.subheader(f"🎨 Overlay Result ({blend_mode}, {int(opacity*100)}% opacity)")
    st.image(result, caption="Image 1 + Image 2 (Same Position Overlay)", use_container_width=True)
    
    # Side by side comparison with slider
    st.markdown("---")
    st.subheader("🔄 Interactive Comparison")
    
    compare_opacity = st.slider("Slide to compare Image 1 ↔ Image 2", 0.0, 1.0, 0.5, 0.01, key="compare")
    
    # Create comparison image
    img1_norm = img1_final.astype(np.float32) / 255.0
    img2_norm = img2_final.astype(np.float32) / 255.0
    comparison = img1_norm * (1 - compare_opacity) + img2_norm * compare_opacity
    comparison = np.clip(comparison * 255, 0, 255).astype(np.uint8)
    
    st.image(comparison, caption=f"Image 1 ({int((1-compare_opacity)*100)}%) ↔ Image 2 ({int(compare_opacity*100)}%)", use_container_width=True)
    
    # Download buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        buf1 = io.BytesIO()
        Image.fromarray(img1_final).save(buf1, format='PNG')
        buf1.seek(0)
        st.download_button("📥 Download Image 1", buf1, "image1_aligned.png", "image/png")
    
    with col2:
        buf2 = io.BytesIO()
        Image.fromarray(img2_final).save(buf2, format='PNG')
        buf2.seek(0)
        st.download_button("📥 Download Image 2", buf2, "image2_aligned.png", "image/png")
    
    with col3:
        buf3 = io.BytesIO()
        Image.fromarray(result).save(buf3, format='PNG')
        buf3.seek(0)
        st.download_button("📥 Download Overlay", buf3, "overlay_result.png", "image/png")
    
    # Show all blend modes
    if st.checkbox("🔍 Show all blend modes"):
        st.markdown("---")
        st.subheader("📊 All Blend Modes Comparison")
        
        modes = ['Normal', 'Multiply', 'Screen', 'Overlay', 'Difference', 'Add', 'Soft Light']
        cols = st.columns(4)
        
        for i, mode in enumerate(modes):
            with cols[i % 4]:
                result_mode, _, _, _ = process_overlay(img1_np, img2_np, mode, opacity, use_alignment)
                st.image(result_mode, caption=mode, use_container_width=True)

else:
    st.info("👆 Please upload both Image 1 (Thermal) and Image 2 (RGB) to see the overlay.")
    
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use:
    1. Upload **Image 1** (Thermal) and **Image 2** (RGB)
    2. Both images will be **aligned to the same position**
    3. Adjust **blend mode** and **opacity** in sidebar
    4. Use **interactive comparison slider** to compare
    5. Download aligned images or overlay result
    
    ### 🎯 Same Position Alignment:
    - **Feature Matching**: Detects common features and aligns images precisely
    - **Homography Transform**: Warps Image 2 to match Image 1's position
    - **Pixel-Perfect Overlay**: Every feature aligns exactly
    
    ### 🎨 Blend Modes:
    - **Normal** - Simple transparency blend
    - **Screen** - Brightens, reveals hot spots
    - **Multiply** - Darkens, shows thermal patterns
    - **Overlay** - Combines contrast from both
    - **Difference** - Highlights temperature variations
    """)
