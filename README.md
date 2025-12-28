import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --- SAFETY CHECK ---
try:
    import cv2
except ImportError:
    st.error("CRITICAL: 'opencv-python-headless' is missing from requirements.txt")
    st.stop()

st.set_page_config(page_title="Onion Probability AI", layout="wide")
st.title("🧅 Onion AI: Size Probability Dashboard")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Calibration")
    st.info("⚠️ Use a SOLID Green Object.")
    ref_real_size = st.number_input("Reference Diameter (mm)", value=30.0)

    with st.expander("🔧 Reference Tuning"):
        ref_h_min = st.slider("Ref Hue Min", 0, 179, 35)
        ref_h_max = st.slider("Ref Hue Max", 0, 179, 90)
        ref_s_min = st.slider("Ref Saturation Min", 0, 255, 80)
        ref_v_min = st.slider("Ref Brightness Min", 0, 255, 70)

    st.divider()
    st.header("2. Detection Settings")
    sprout_k = st.slider("Sprout Eraser Size", 1, 25, 11, step=2)
    min_area = st.number_input("Min Area (Ignore Dirt)", value=4000, step=500)
    measure_logic = st.radio("Measurement Logic", ["Min Axis (Width)", "Max Axis (Length)", "Enclosing Circle"])
    
    show_masks = st.checkbox("Show Debug Masks", value=False)

# --- PROCESSING ENGINE ---
def analyze_probability(file_bytes, real_ref_mm, 
                        ref_h_min, ref_h_max, ref_s_min, ref_v_min,
                        logic, sprout_k_size, min_area_thresh):
    
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, "Error decoding image."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- REFERENCE DETECTION ---
    lower_green = np.array([ref_h_min, ref_s_min, ref_v_min])
    upper_green = np.array([ref_h_max, 255, 255])
    mask_ref = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = np.ones((5,5), np.uint8)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts_ref: return None, "Green Reference Not Found."
    
    ref_contour = max(cnts_ref, key=cv2.contourArea)
    ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
    px_per_mm = (ref_radius * 2) / real_ref_mm
    
    # --- ONION DETECTION ---
    mask1 = cv2.inRange(hsv, np.array([160, 60, 50]), np.array([179, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([20, 255, 255]))
    mask_onion = cv2.add(mask1, mask2)

    # Fill Holes
    contours_temp, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask_onion)
    big_contours = [c for c in contours_temp if cv2.contourArea(c) > (min_area_thresh / 4)]
    cv2.drawContours(mask_filled, big_contours, -1, 255, thickness=cv2.FILLED)
    
    # Remove Ref
    mask_ref_dilated = cv2.dilate(mask_ref, np.ones((15,15), np.uint8), iterations=1)
    mask_final = cv2.subtract(mask_filled, mask_ref_dilated)
    
    # Sprout Removal
    if sprout_k_size > 1:
        kernel_sprout = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sprout_k_size, sprout_k_size))
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel_sprout)
    
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sizes = []
    result_img = img.copy()
    
    ((rx, ry), rr) = cv2.minEnclosingCircle(ref_contour)
    cv2.circle(result_img, (int(rx), int(ry)), int(rr), (0, 255, 0), 3)

    for c in cnts_final:
        if cv2.contourArea(c) > min_area_thresh:
            if len(c) < 5: continue
            (center, (MA, ma), angle) = cv2.fitEllipse(c)
            axes = sorted([MA, ma])
            
            if logic == "Min Axis (Width)": dia_mm = axes[0] / px_per_mm
            elif logic == "Max Axis (Length)": dia_mm = axes[1] / px_per_mm
            else: 
                ((_, _), r) = cv2.minEnclosingCircle(c)
                dia_mm = (r * 2) / px_per_mm

            sizes.append(dia_mm)
            
            # Simple Visuals: Green circle for all valid onions
            cv2.ellipse(result_img, (center, (MA, ma), angle), (0, 255, 0), 2)
            cv2.putText(result_img, f"{int(dia_mm)}", (int(center[0])-10, int(center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return sizes, result_img, mask_final, mask_ref

# --- UI ---
uploaded_file = st.file_uploader("Upload Sample Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Using sidebar values
    result = analyze_probability(file_bytes, ref_real_size, 
                              st.session_state.get('Ref Hue Min', 35),
                              st.session_state.get('Ref Hue Max', 90),
                              st.session_state.get('Ref Saturation Min', 80),
                              st.session_state.get('Ref Brightness Min', 70),
                              measure_logic, sprout_k, min_area)
    
    if result and len(result) == 4:
        sizes, final_img, mask_o, mask_r = result
        st.image(final_img, channels="BGR", caption="Analyzed Sample", use_container_width=True)
        
        if show_masks:
             st.image(mask_o, caption="Onion Mask", width=300)

        if sizes:
            df = pd.DataFrame(sizes, columns=['mm'])
            
            # --- PROBABILITY METRICS ---
            st.divider()
            st.header("📊 Size Probability Report")
            
            total_onions = len(df)
            
            # 1. Calculate Counts
            count_65_plus = len(df[df['mm'] >= 65])
            count_55_plus = len(df[df['mm'] >= 55])
            
            # 2. Calculate Probabilities
            prob_65 = (count_65_plus / total_onions) * 100
            prob_55 = (count_55_plus / total_onions) * 100
            
            # 3. Display Metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric(
                label="Sample Size", 
                value=f"{total_onions} Onions"
            )
            
            col2.metric(
                label="Probability ≥ 65mm", 
                value=f"{prob_65:.1f}%",
                delta=f"{count_65_plus} onions"
            )
            
            col3.metric(
                label="Probability ≥ 55mm", 
                value=f"{prob_55:.1f}%", 
                delta=f"{count_55_plus} onions"
            )
            
            # 4. Detailed Breakdown Table
            st.write("### Detailed Distribution")
            
            # Binning data for a clean table
            bins = [0, 45, 55, 65, 1000]
            labels = ['Small (<45mm)', 'Medium (45-55mm)', 'Large (55-65mm)', 'Jumbo (>65mm)']
            df['Category'] = pd.cut(df['mm'], bins=bins, labels=labels)
            
            breakdown = df['Category'].value_counts().reset_index()
            breakdown.columns = ['Size Category', 'Count']
            breakdown['Percentage'] = (breakdown['Count'] / total_onions * 100).map('{:.1f}%'.format)
            
            st.dataframe(breakdown, use_container_width=True)

            # 5. Histogram
            fig = px.histogram(df, x="mm", nbins=15, title="Full Size Distribution", labels={'mm':'Size (mm)'})
            fig.add_vline(x=55, line_dash="dash", line_color="orange", annotation_text="55mm Limit")
            fig.add_vline(x=65, line_dash="dash", line_color="green", annotation_text="65mm Limit")
            st.plotly_chart(fig, use_container_width=True)

    elif result:
        st.error(result[1])
