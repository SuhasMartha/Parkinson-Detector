"""
Enhanced Drawing Mode for Parkinson's Detection
Supports both image upload and real-time drawing using canvas
Based on reference projects with SVM features and drawing canvas
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# Try to import drawing canvas
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False
    st.warning("Canvas library not available. Using upload only.")

# ========== DRAWING FEATURE EXTRACTION ==========
class DrawingAnalyzer:
    """Extract features from drawing for Parkinson's detection"""

    @staticmethod
    def extract_tremor_features(image_gray):
        """
        Extract tremor-related features:
        - Stroke width variations
        - Pressure variations
        - Line smoothness
        """
        try:
            # Apply edge detection
            edges = cv2.Canny(image_gray, 100, 200)

            # Calculate line thickness/tremor
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)

            # Tremor indicator: variance in line thickness
            contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return 0.0

            # Calculate contour properties
            contour_widths = []
            for contour in contours:
                if cv2.contourArea(contour) > 10:
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_widths.append(w)

            if contour_widths:
                tremor_score = np.std(contour_widths) / (np.mean(contour_widths) + 1e-10)
                return float(tremor_score)
            return 0.0
        except:
            return 0.0

    @staticmethod
    def extract_pressure_features(image_gray):
        """
        Extract pressure variation features:
        - Intensity variations
        - Darkness/intensity distribution
        """
        try:
            # Calculate intensity distribution
            hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()

            # Entropy - high entropy = pressure variations
            entropy = -np.sum(hist * np.log(hist + 1e-10))

            # Mean intensity (darkness)
            mean_intensity = np.mean(image_gray)

            # Std of intensity
            std_intensity = np.std(image_gray)

            return {
                'entropy': float(entropy),
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity)
            }
        except:
            return {'entropy': 0.0, 'mean_intensity': 0.0, 'std_intensity': 0.0}

    @staticmethod
    def extract_smoothness_features(image_gray):
        """
        Extract line smoothness features:
        - Curvature
        - Direction changes
        - Smoothness score
        """
        try:
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(image_gray, kernel, iterations=2)

            # Calculate Laplacian (curvature)
            laplacian = cv2.Laplacian(dilated, cv2.CV_64F)
            curvature = np.mean(np.abs(laplacian))

            # Calculate variance of laplacian
            curvature_var = np.var(laplacian)

            # Smoothness (inverse of curvature variance)
            smoothness = 1.0 / (1.0 + curvature_var)

            return {
                'curvature': float(curvature),
                'curvature_variance': float(curvature_var),
                'smoothness': float(smoothness)
            }
        except:
            return {'curvature': 0.0, 'curvature_variance': 0.0, 'smoothness': 0.0}

    @staticmethod
    def extract_coverage_features(image_gray):
        """
        Extract spiral coverage features:
        - Coverage percentage
        - Spiral completeness
        """
        try:
            # Binary image
            _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

            # Calculate coverage
            total_pixels = binary.shape[0] * binary.shape[1]
            drawn_pixels = np.count_nonzero(binary)
            coverage = drawn_pixels / total_pixels

            # Find contours to check completeness
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)

            return {
                'coverage': float(coverage),
                'num_contours': int(num_contours)
            }
        except:
            return {'coverage': 0.0, 'num_contours': 0}

    @staticmethod
    def extract_all_features(image_gray):
        """Extract all features from drawing"""
        features = {}

        # Tremor features
        tremor = DrawingAnalyzer.extract_tremor_features(image_gray)
        features['tremor_score'] = tremor

        # Pressure features
        pressure = DrawingAnalyzer.extract_pressure_features(image_gray)
        features.update(pressure)

        # Smoothness features
        smoothness = DrawingAnalyzer.extract_smoothness_features(image_gray)
        features.update(smoothness)

        # Coverage features
        coverage = DrawingAnalyzer.extract_coverage_features(image_gray)
        features.update(coverage)

        return features

# ========== STREAMLIT DRAWING INTERFACE ==========
def create_enhanced_drawing_tab():
    """
    Create enhanced drawing tab with both upload and draw options
    """

    st.markdown("""
        <div class='model-card'>
            <div class='model-title'>✏️ Enhanced Drawing Test</div>
            <p>Spiral/Wave drawing analysis for motor control assessment</p>
        </div>
    """, unsafe_allow_html=True)

    # Two column layout
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("#### ✏️ Input Options")

        # Mode selection
        draw_mode = st.radio(
            "Choose input method:",
            ["📁 Upload Drawing", "🎨 Draw Here"],
            key="draw_mode_select"
        )

        analyzer = DrawingAnalyzer()

        # ========== MODE 1: UPLOAD ==========
        if draw_mode == "📁 Upload Drawing":
            st.markdown("**Upload spiral or wave drawing image**")

            drawing_file = st.file_uploader(
                "Choose drawing image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="drawing_upload"
            )

            if drawing_file:
                image = Image.open(drawing_file)
                st.image(image, caption="Uploaded Drawing", use_column_width=True)

                if st.button("🔍 Analyze Drawing", key="analyze_upload_drawing"):
                    with st.spinner("Analyzing drawing..."):
                        # Convert to grayscale
                        img_array = np.array(image)
                        if len(img_array.shape) == 3:
                            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        else:
                            img_gray = img_array

                        # Extract features
                        features = analyzer.extract_all_features(img_gray)

                        st.session_state.drawing_features = features
                        st.session_state.drawing_image = img_gray
                        st.success("✅ Analysis complete!")

        # ========== MODE 2: DRAW ON CANVAS ==========
        else:
            st.markdown("**Draw spiral or wave pattern**")

            if CANVAS_AVAILABLE:
                # Canvas drawing
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=300,
                    width=300,
                    drawing_mode="freedraw",
                    key="canvas_draw"
                )

                if st.button("🔍 Analyze Drawing", key="analyze_drawn"):
                    if canvas_result.image_data is not None:
                        with st.spinner("Analyzing drawing..."):
                            # Get drawn image
                            img_array = canvas_result.image_data

                            # Convert to grayscale
                            if len(img_array.shape) == 3:
                                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            else:
                                img_gray = img_array

                            # Extract features
                            features = analyzer.extract_all_features(img_gray)

                            st.session_state.drawing_features = features
                            st.session_state.drawing_image = img_gray
                            st.success("✅ Analysis complete!")
                    else:
                        st.warning("Please draw something first!")
            else:
                st.info("📝 Canvas drawing not available. Please use upload mode.")
                st.markdown("""
                To enable drawing, install:
                ```bash
                pip install streamlit-drawable-canvas
                ```
                """)

    # ========== RESULTS DISPLAY (RIGHT COLUMN) ==========
    with col2:
        if 'drawing_features' in st.session_state:
            features = st.session_state.drawing_features

            st.markdown("#### 📊 Analysis Results")

            # Calculate risk score
            risk_score = calculate_risk_score(features)

            # Display prediction
            if risk_score > 0.6:
                st.markdown("""
                <div class='prediction-positive'>
                    <h3>🚨 Abnormal Pattern Detected</h3>
                    <p>Drawing shows signs of motor impairment</p>
                </div>
                """, unsafe_allow_html=True)
                status = "Abnormal"
                confidence = risk_score
            else:
                st.markdown("""
                <div class='prediction-negative'>
                    <h3>✅ Normal Drawing Pattern</h3>
                    <p>Motor control appears normal</p>
                </div>
                """, unsafe_allow_html=True)
                status = "Normal"
                confidence = 1.0 - risk_score

            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Status", status)
            with col_b:
                st.metric("Confidence", f"{confidence*100:.1f}%")

            # Feature analysis
            st.markdown("#### 🔬 Feature Analysis")

            feature_cols = st.columns(2)

            with feature_cols[0]:
                st.markdown("**Motor Features:**")
                st.write(f"🎯 Tremor Score: {features.get('tremor_score', 0):.3f}")
                st.write(f"😐 Smoothness: {features.get('smoothness', 0):.3f}")
                st.write(f"📏 Curvature: {features.get('curvature', 0):.3f}")

            with feature_cols[1]:
                st.markdown("**Drawing Features:**")
                st.write(f"📊 Entropy: {features.get('entropy', 0):.3f}")
                st.write(f"🎨 Coverage: {features.get('coverage', 0):.3f}")
                st.write(f"📐 Contours: {features.get('num_contours', 0)}")

            # Detailed interpretation
            with st.expander("📖 Feature Interpretation"):
                st.markdown(f"""
                **Tremor Score**: {features.get('tremor_score', 0):.3f}
                - Measures line thickness variations
                - Higher = more tremor = more abnormal
                - Healthy: < 0.3, Parkinson's: > 0.5

                **Smoothness**: {features.get('smoothness', 0):.3f}
                - Measures how smooth the drawing is
                - Higher = smoother (normal)
                - Lower = jagged lines (abnormal)

                **Entropy**: {features.get('entropy', 0):.3f}
                - Measures pressure/intensity variations
                - Higher = more pressure variation
                - Healthy: moderate, PD: varies

                **Coverage**: {features.get('coverage', 0):.3f}
                - Percentage of spiral filled
                - Lower = incomplete spiral (concern)
                - Healthy: > 0.3
                """)

            # Display image with analysis
            if 'drawing_image' in st.session_state:
                st.markdown("#### 🖼️ Drawing Image")
                st.image(st.session_state.drawing_image, caption="Analyzed Drawing")
        else:
            st.info("👆 Upload or draw a spiral to see analysis results")

def calculate_risk_score(features):
    """
    Calculate Parkinson's risk score from features
    Based on clinical literature
    """
    try:
        score = 0.0
        weights = 0.0

        # Tremor component (30% weight)
        if 'tremor_score' in features:
            tremor = min(features['tremor_score'], 1.0)
            score += tremor * 0.3
            weights += 0.3

        # Smoothness component (20% weight)
        if 'smoothness' in features:
            smoothness = features['smoothness']
            score += (1.0 - smoothness) * 0.2  # Lower smoothness = higher risk
            weights += 0.2

        # Curvature component (15% weight)
        if 'curvature' in features:
            curvature = min(features['curvature'] / 10.0, 1.0)
            score += curvature * 0.15
            weights += 0.15

        # Entropy component (15% weight)
        if 'entropy' in features:
            entropy = min(features['entropy'] / 5.0, 1.0)
            score += entropy * 0.15
            weights += 0.15

        # Coverage component (20% weight)
        if 'coverage' in features:
            coverage = features['coverage']
            if coverage < 0.2:
                score += 0.8 * 0.2  # Very incomplete
            elif coverage < 0.3:
                score += 0.5 * 0.2  # Incomplete
            else:
                score += 0.2 * 0.2  # Complete
            weights += 0.2

        # Normalize
        if weights > 0:
            score = score / weights

        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

# ========== USAGE IN APP ==========
# In app_separated_models.py, replace the drawing section with:
# 
# elif model_section == "✏️ Drawing Test":
#     from enhanced_drawing_module import create_enhanced_drawing_tab
#     create_enhanced_drawing_tab()
