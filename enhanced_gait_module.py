"""
Enhanced Gait Analysis Module for Parkinson's Detection
Based on clinical rules and explainable AI (Prodromal_Parkinson + TRIAD2PD)
Wearable sensor gait parameters with decision trees
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ========== GAIT PARAMETERS DATABASE ==========
GAIT_NORMAL_RANGES = {
    'cadence': {'min': 100, 'max': 130, 'unit': 'steps/min', 'name': 'Cadence'},
    'stride_length': {'min': 1.2, 'max': 1.5, 'unit': 'm', 'name': 'Stride Length'},
    'stride_velocity': {'min': 1.2, 'max': 1.5, 'unit': 'm/s', 'name': 'Stride Velocity'},
    'step_width': {'min': 0.08, 'max': 0.12, 'unit': 'm', 'name': 'Step Width'},
    'double_support': {'min': 20, 'max': 30, 'unit': '%', 'name': 'Double Support %'},
    'single_support': {'min': 35, 'max': 45, 'unit': '%', 'name': 'Single Support %'},
    'swing_phase': {'min': 40, 'max': 50, 'unit': '%', 'name': 'Swing Phase %'},
    'stance_phase': {'min': 50, 'max': 60, 'unit': '%', 'name': 'Stance Phase %'},
    'postural_stability': {'min': 0, 'max': 10, 'unit': 'cm', 'name': 'Postural Stability (CoP)'},
    'gait_variability': {'min': 0, 'max': 5, 'unit': '%', 'name': 'Gait Variability'},
}

# ========== CLINICAL DECISION RULES ==========
class GaitAnalyzer:
    """
    Gait analysis using clinical decision trees
    Based on Prodromal Parkinson + TRIAD2PD methodology
    """

    @staticmethod
    def check_bradykinesia_gait(cadence, stride_length, velocity):
        """
        Check for bradykinesia (slow movement)
        - Reduced cadence
        - Reduced stride length
        - Reduced walking velocity
        """
        score = 0
        findings = []

        # Cadence check (normal: 100-130)
        if cadence < 95:
            score += 2
            findings.append("⚠️ Low Cadence (Bradykinesia indicator)")
        elif cadence < 100:
            score += 1
            findings.append("⚠️ Below average cadence")

        # Stride length check (normal: 1.2-1.5m)
        if stride_length < 1.0:
            score += 2
            findings.append("⚠️ Very Short Stride (Bradykinesia indicator)")
        elif stride_length < 1.2:
            score += 1
            findings.append("⚠️ Below average stride length")

        # Velocity check (normal: 1.2-1.5 m/s)
        if velocity < 1.0:
            score += 2
            findings.append("⚠️ Slow Walking Speed (Bradykinesia indicator)")
        elif velocity < 1.2:
            score += 1
            findings.append("⚠️ Below average walking speed")

        return {
            'score': score,
            'findings': findings,
            'severity': 'High' if score >= 4 else 'Moderate' if score >= 2 else 'Low'
        }

    @staticmethod
    def check_rigidity_gait(double_support, step_width, postural_stability):
        """
        Check for rigidity effects on gait
        - Increased double support time
        - Increased step width (balance issues)
        - Poor postural stability
        """
        score = 0
        findings = []

        # Double support (normal: 20-30%)
        if double_support > 35:
            score += 2
            findings.append("⚠️ Prolonged Double Support (Rigidity indicator)")
        elif double_support > 30:
            score += 1
            findings.append("⚠️ Slightly increased double support")

        # Step width (normal: 0.08-0.12m)
        if step_width > 0.15:
            score += 2
            findings.append("⚠️ Wide Step Width (Balance/Rigidity concern)")
        elif step_width > 0.12:
            score += 1
            findings.append("⚠️ Wider than normal step")

        # Postural stability (CoP displacement in cm)
        if postural_stability > 15:
            score += 2
            findings.append("⚠️ Poor Postural Stability (Balance risk)")
        elif postural_stability > 10:
            score += 1
            findings.append("⚠️ Below average postural stability")

        return {
            'score': score,
            'findings': findings,
            'severity': 'High' if score >= 4 else 'Moderate' if score >= 2 else 'Low'
        }

    @staticmethod
    def check_freezing_gait(swing_phase, cadence_variability, stride_variability):
        """
        Check for freezing of gait
        - Reduced swing phase
        - High cadence variability
        - High stride variability
        """
        score = 0
        findings = []

        # Swing phase (normal: 40-50%)
        if swing_phase < 35:
            score += 2
            findings.append("⚠️ Reduced Swing Phase (Freezing indicator)")
        elif swing_phase < 40:
            score += 1
            findings.append("⚠️ Below average swing phase")

        # Cadence variability (normal: < 5%)
        if cadence_variability > 10:
            score += 2
            findings.append("⚠️ High Cadence Variability (Freezing indicator)")
        elif cadence_variability > 7:
            score += 1
            findings.append("⚠️ Elevated cadence variability")

        # Stride variability (normal: < 5%)
        if stride_variability > 10:
            score += 2
            findings.append("⚠️ High Stride Variability (Freezing indicator)")
        elif stride_variability > 7:
            score += 1
            findings.append("⚠️ Elevated stride variability")

        return {
            'score': score,
            'findings': findings,
            'severity': 'High' if score >= 4 else 'Moderate' if score >= 2 else 'Low'
        }

    @staticmethod
    def check_postural_instability(postural_stability, step_width, double_support):
        """
        Check for postural instability
        - Poor CoP stability
        - Wide base of support
        - Balance compensation (increased double support)
        """
        score = 0
        findings = []

        # CoP stability
        if postural_stability > 12:
            score += 2
            findings.append("⚠️ Significant Postural Instability")
        elif postural_stability > 8:
            score += 1
            findings.append("⚠️ Mild postural instability")

        # Wide base
        if step_width > 0.15:
            score += 1
            findings.append("⚠️ Wide base (compensation for instability)")

        # Increased double support
        if double_support > 32:
            score += 1
            findings.append("⚠️ Increased double support (stability seeking)")

        return {
            'score': score,
            'findings': findings,
            'severity': 'High' if score >= 3 else 'Moderate' if score >= 2 else 'Low'
        }

    @staticmethod
    def calculate_overall_risk(gait_params):
        """
        Calculate overall Parkinson's gait risk score
        Using weighted components
        """
        # Bradykinesia analysis (40% weight)
        bradykinesia = GaitAnalyzer.check_bradykinesia_gait(
            gait_params['cadence'],
            gait_params['stride_length'],
            gait_params['stride_velocity']
        )

        # Rigidity analysis (25% weight)
        rigidity = GaitAnalyzer.check_rigidity_gait(
            gait_params['double_support'],
            gait_params['step_width'],
            gait_params['postural_stability']
        )

        # Freezing analysis (20% weight)
        freezing = GaitAnalyzer.check_freezing_gait(
            gait_params['swing_phase'],
            gait_params['cadence_variability'],
            gait_params['stride_variability']
        )

        # Postural instability (15% weight)
        postural = GaitAnalyzer.check_postural_instability(
            gait_params['postural_stability'],
            gait_params['step_width'],
            gait_params['double_support']
        )

        # Calculate weighted score (0-1)
        max_score = (6 * 0.4) + (4 * 0.25) + (4 * 0.2) + (3 * 0.15)
        weighted_score = (
            (bradykinesia['score'] * 0.4) +
            (rigidity['score'] * 0.25) +
            (freezing['score'] * 0.2) +
            (postural['score'] * 0.15)
        ) / max_score

        return {
            'overall_risk': min(weighted_score, 1.0),
            'bradykinesia': bradykinesia,
            'rigidity': rigidity,
            'freezing': freezing,
            'postural': postural,
            'total_findings': (len(bradykinesia['findings']) + 
                             len(rigidity['findings']) +
                             len(freezing['findings']) +
                             len(postural['findings']))
        }

# ========== STREAMLIT INTERFACE ==========
def create_enhanced_gait_tab():
    """Create enhanced gait analysis interface"""

    st.markdown("""
        <div class='model-card'>
            <div class='model-title'>🚶 Clinical Gait Analysis</div>
            <p>Advanced gait parameter assessment based on wearable sensor data and clinical decision trees</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("#### 📊 Gait Parameters")

        # Input mode selection
        input_mode = st.radio(
            "Input method:",
            ["📋 Manual Entry", "📁 Upload CSV", "🎯 Quick Test"],
            key="gait_input_mode"
        )

        gait_params = {}

        if input_mode == "📋 Manual Entry":
            st.markdown("**Enter gait parameters:**")

            gait_params['cadence'] = st.slider(
                "Cadence (steps/min)",
                min_value=50, max_value=150, value=110,
                help="Normal: 100-130 steps/min"
            )

            gait_params['stride_length'] = st.slider(
                "Stride Length (m)",
                min_value=0.5, max_value=2.0, value=1.35, step=0.05,
                help="Normal: 1.2-1.5 m"
            )

            gait_params['stride_velocity'] = st.slider(
                "Stride Velocity (m/s)",
                min_value=0.5, max_value=2.0, value=1.35, step=0.05,
                help="Normal: 1.2-1.5 m/s"
            )

            gait_params['step_width'] = st.slider(
                "Step Width (m)",
                min_value=0.02, max_value=0.3, value=0.1, step=0.01,
                help="Normal: 0.08-0.12 m"
            )

            gait_params['double_support'] = st.slider(
                "Double Support Phase (%)",
                min_value=10, max_value=50, value=25,
                help="Normal: 20-30%"
            )

            gait_params['swing_phase'] = st.slider(
                "Swing Phase (%)",
                min_value=20, max_value=60, value=45,
                help="Normal: 40-50%"
            )

            gait_params['postural_stability'] = st.slider(
                "Postural Stability - CoP (cm)",
                min_value=0, max_value=20, value=5,
                help="Center of Pressure displacement"
            )

            gait_params['cadence_variability'] = st.slider(
                "Cadence Variability (%)",
                min_value=0, max_value=15, value=3,
                help="Stride-to-stride variability"
            )

            gait_params['stride_variability'] = st.slider(
                "Stride Variability (%)",
                min_value=0, max_value=15, value=3,
                help="Stride length variability"
            )

            # Calculate single support
            gait_params['single_support'] = 100 - gait_params['double_support'] - gait_params['swing_phase']

        elif input_mode == "📁 Upload CSV":
            st.markdown("**Upload gait data CSV**")
            csv_file = st.file_uploader("Choose CSV file", type="csv", key="gait_csv")

            if csv_file:
                df = pd.read_csv(csv_file)
                st.write("Preview:", df.head())

                # Extract first row or average
                if len(df) > 0:
                    gait_params['cadence'] = float(df['cadence'].iloc[0] if 'cadence' in df.columns else 110)
                    gait_params['stride_length'] = float(df['stride_length'].iloc[0] if 'stride_length' in df.columns else 1.35)
                    gait_params['stride_velocity'] = float(df['stride_velocity'].iloc[0] if 'stride_velocity' in df.columns else 1.35)
                    gait_params['step_width'] = float(df['step_width'].iloc[0] if 'step_width' in df.columns else 0.1)
                    gait_params['double_support'] = float(df['double_support'].iloc[0] if 'double_support' in df.columns else 25)
                    gait_params['swing_phase'] = float(df['swing_phase'].iloc[0] if 'swing_phase' in df.columns else 45)
                    gait_params['postural_stability'] = float(df['postural_stability'].iloc[0] if 'postural_stability' in df.columns else 5)
                    gait_params['cadence_variability'] = float(df['cadence_variability'].iloc[0] if 'cadence_variability' in df.columns else 3)
                    gait_params['stride_variability'] = float(df['stride_variability'].iloc[0] if 'stride_variability' in df.columns else 3)
                    gait_params['single_support'] = 100 - gait_params['double_support'] - gait_params['swing_phase']

        else:  # Quick Test
            st.markdown("**Select a scenario:**")
            scenario = st.radio(
                "Test scenario:",
                ["Healthy Controls", "Prodromal PD", "Severe PD"],
                key="gait_scenario"
            )

            if scenario == "Healthy Controls":
                gait_params = {
                    'cadence': 115,
                    'stride_length': 1.40,
                    'stride_velocity': 1.40,
                    'step_width': 0.10,
                    'double_support': 25,
                    'swing_phase': 45,
                    'single_support': 50,
                    'postural_stability': 4,
                    'cadence_variability': 2,
                    'stride_variability': 2
                }
            elif scenario == "Prodromal PD":
                gait_params = {
                    'cadence': 105,
                    'stride_length': 1.10,
                    'stride_velocity': 1.15,
                    'step_width': 0.13,
                    'double_support': 32,
                    'swing_phase': 38,
                    'single_support': 48,
                    'postural_stability': 8,
                    'cadence_variability': 5,
                    'stride_variability': 6
                }
            else:  # Severe PD
                gait_params = {
                    'cadence': 85,
                    'stride_length': 0.85,
                    'stride_velocity': 0.85,
                    'step_width': 0.18,
                    'double_support': 40,
                    'swing_phase': 32,
                    'single_support': 45,
                    'postural_stability': 14,
                    'cadence_variability': 10,
                    'stride_variability': 12
                }

        if st.button("🔍 Analyze Gait", key="analyze_gait", use_container_width=True):
            if gait_params:
                st.session_state.gait_analysis = GaitAnalyzer.calculate_overall_risk(gait_params)
                st.session_state.gait_params = gait_params
                st.success("✅ Analysis complete!")

    # ========== RESULTS DISPLAY ==========
    with col2:
        if 'gait_analysis' in st.session_state:
            analysis = st.session_state.gait_analysis
            overall_risk = analysis['overall_risk']

            st.markdown("#### 📈 Analysis Results")

            # Risk classification
            if overall_risk > 0.7:
                st.markdown("""
                <div class='prediction-positive'>
                    <h3>🚨 High Gait Abnormality Risk</h3>
                    <p>Multiple gait parameters suggest Parkinson's disease concerns</p>
                </div>
                """, unsafe_allow_html=True)
                risk_level = "High Risk"
            elif overall_risk > 0.4:
                st.markdown("""
                <div class='model-card'>
                    <h3>⚠️ Moderate Gait Changes Detected</h3>
                    <p>Some gait parameters warrant monitoring</p>
                </div>
                """, unsafe_allow_html=True)
                risk_level = "Moderate Risk"
            else:
                st.markdown("""
                <div class='prediction-negative'>
                    <h3>✅ Normal Gait Pattern</h3>
                    <p>Gait parameters within healthy ranges</p>
                </div>
                """, unsafe_allow_html=True)
                risk_level = "Low Risk"

            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Overall Risk", f"{overall_risk*100:.1f}%")
            with col_b:
                st.metric("Risk Level", risk_level)
            with col_c:
                st.metric("Findings", analysis['total_findings'])

            # Component breakdown
            st.markdown("#### 🔬 Component Analysis")

            component_scores = {
                'Bradykinesia': analysis['bradykinesia']['score'],
                'Rigidity': analysis['rigidity']['score'],
                'Freezing': analysis['freezing']['score'],
                'Postural': analysis['postural']['score']
            }

            # Bar chart
            fig = go.Figure(data=[
                go.Bar(x=list(component_scores.keys()), 
                       y=list(component_scores.values()),
                       marker=dict(color=['#FF6B6B', '#FFA500', '#FFD700', '#FF8C00']))
            ])
            fig.update_layout(
                title="Parkinson's Gait Markers Score",
                yaxis_title="Score",
                xaxis_title="Gait Component",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed findings
            with st.expander("📋 Detailed Findings"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Bradykinesia Findings** (40% weight)")
                    st.markdown(f"Severity: {analysis['bradykinesia']['severity']}")
                    for finding in analysis['bradykinesia']['findings']:
                        st.write(finding)

                    st.markdown("**Freezing Findings** (20% weight)")
                    st.markdown(f"Severity: {analysis['freezing']['severity']}")
                    for finding in analysis['freezing']['findings']:
                        st.write(finding)

                with col2:
                    st.markdown("**Rigidity Findings** (25% weight)")
                    st.markdown(f"Severity: {analysis['rigidity']['severity']}")
                    for finding in analysis['rigidity']['findings']:
                        st.write(finding)

                    st.markdown("**Postural Instability** (15% weight)")
                    st.markdown(f"Severity: {analysis['postural']['severity']}")
                    for finding in analysis['postural']['findings']:
                        st.write(finding)

            # Clinical interpretation
            with st.expander("📖 Clinical Interpretation"):
                st.markdown(f"""
                **Overall Assessment:**

                Based on wearable sensor gait analysis, this subject shows:
                - **Bradykinesia Assessment:** {analysis['bradykinesia']['severity']}
                - **Rigidity Pattern:** {analysis['rigidity']['severity']}
                - **Freezing Risk:** {analysis['freezing']['severity']}
                - **Postural Stability:** {analysis['postural']['severity']}

                **Risk Level:** {risk_level}

                **Recommendation:**
                """ + (
                    "Regular clinical monitoring is recommended. Consider referral to movement disorders specialist."
                    if overall_risk > 0.5
                    else "Continue routine follow-up. No immediate intervention needed."
                ))

            # Parameters table
            st.markdown("#### 📊 Parameter Summary")
            params_df = pd.DataFrame({
                'Parameter': list(st.session_state.gait_params.keys()),
                'Value': list(st.session_state.gait_params.values())
            })
            st.dataframe(params_df, use_container_width=True)

        else:
            st.info("👆 Enter or select gait parameters to see analysis")

# ========== USAGE IN APP ==========
# In app_separated_models.py, replace the gait section with:
# 
# elif model_section == "🚶 Gait Analysis":
#     from enhanced_gait_module import create_enhanced_gait_tab
#     create_enhanced_gait_tab()
