import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf  #type: ignore
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()  # Loading environment variables from .env i.e api key

# --------- Gemini API Setup ---------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --------- Enhanced Page Config ---------
st.set_page_config(
    page_title="Plant Disease Detection 🌿",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="collapsed"
)

# --------- Simple & Clean CSS Styling ---------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call the CSS loader
local_css("D:/plant_disease_detection/styles.css")

# --------- Load Model and Class Indices ---------
model_path = "D:/plant_disease_detection/plant_disease_prediction_model.h5"
class_indices_path = "D:/plant_disease_detection/class_indices.json"

model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# --------- Descriptive Labels ---------
disease_descriptions = {
    "Apple___Apple_scab": "🍎 Apple Scab: Fungal disease with dark lesions on leaves and fruit.",
    "Apple___Black_rot": "🍎 Black Rot: Fungal infection causing leaf spots and fruit rot in apples.",
    "Apple___Cedar_apple_rust": "🍎 Cedar Apple Rust: Causes orange lesions on apple leaves.",
    "Apple___healthy": "🍏 Healthy Apple Leaf: No visible disease symptoms.",
    "Blueberry___healthy": "🫐 Healthy Blueberry Leaf: No signs of disease.",
    "Cherry_(including_sour)___Powdery_mildew": "🍒 Powdery Mildew: White fungal coating on cherry leaves.",
    "Cherry_(including_sour)___healthy": "🍒 Healthy Cherry Leaf: No disease detected.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "🌽 Gray Leaf Spot: Grayish lesions on maize leaves.",
    "Corn_(maize)___Common_rust_": "🌽 Common Rust: Small reddish-brown pustules on maize leaves.",
    "Corn_(maize)___Northern_Leaf_Blight": "🌽 Northern Leaf Blight: Long gray-green lesions on maize leaves.",
    "Corn_(maize)___healthy": "🌽 Healthy Corn Leaf: No disease signs.",
    "Grape___Black_rot": "🍇 Black Rot: Black spots on grape leaves and fruit.",
    "Grape___Esca_(Black_Measles)": "🍇 Esca: Causes leaf scorching and fruit rot in grapes.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "🍇 Leaf Blight: Irregular dark spots on grape leaves.",
    "Grape___healthy": "🍇 Healthy Grape Leaf: No infection.",
    "Orange___Haunglongbing_(Citrus_greening)": "🍊 Citrus Greening: Bacterial disease causing yellow shoots and bitter fruit.",
    "Peach___Bacterial_spot": "🍑 Bacterial Spot: Water-soaked spots on peach leaves and fruits.",
    "Peach___healthy": "🍑 Healthy Peach Leaf: No disease.",
    "Pepper,_bell___Bacterial_spot": "🫑 Bacterial Spot: Brown lesions on bell pepper leaves and fruit.",
    "Pepper,_bell___healthy": "🫑 Healthy Bell Pepper Leaf: No symptoms.",
    "Potato___Early_blight": "🥔 Early Blight: Dark spots with concentric rings on potato leaves.",
    "Potato___Late_blight": "🥔 Late Blight: Brown-black spots on potato leaves.",
    "Potato___healthy": "🥔 Healthy Potato Leaf: No disease.",
    "Raspberry___healthy": "🍓 Healthy Raspberry Leaf: No visible disease.",
    "Soybean___healthy": "🌱 Healthy Soybean Leaf: No symptoms of infection.",
    "Squash___Powdery_mildew": "🎃 Powdery Mildew: White fungal growth on squash leaves.",
    "Strawberry___Leaf_scorch": "🍓 Leaf Scorch: Brown edges and yellowing in strawberry leaves.",
    "Strawberry___healthy": "🍓 Healthy Strawberry Leaf: No disease.",
    "Tomato___Bacterial_spot": "🍅 Bacterial Spot: Small water-soaked spots on tomato leaves and fruit.",
    "Tomato___Early_blight": "🍅 Early Blight: Concentric brown spots on tomato leaves.",
    "Tomato___Late_blight": "🍅 Late Blight: Water-soaked lesions and mold on tomato leaves.",
    "Tomato___Leaf_Mold": "🍅 Leaf Mold: Yellow spots and mold on the underside of leaves.",
    "Tomato___Septoria_leaf_spot": "🍅 Septoria Leaf Spot: Tiny, circular spots on tomato leaves.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "🕷️ Spider Mites: Speckled leaves and fine webbing on tomatoes.",
    "Tomato___Target_Spot": "🎯 Target Spot: Brown lesions with rings on tomato leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "💛 TYLCV: Yellow curled leaves and stunted growth.",
    "Tomato___Tomato_mosaic_virus": "🧬 Mosaic Virus: Mottling and leaf distortion in tomatoes.",
    "Tomato___healthy": "🍅 Healthy Tomato Leaf: No disease symptoms."
}

# --------- Image Preprocessing ---------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# --------- Prediction Function ---------
def predict_image_class(model, image_path, class_indices, disease_descriptions):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    description = disease_descriptions.get(predicted_class_name, predicted_class_name)
    confidence = float(np.max(predictions) * 100)
    return predicted_class_name, description, confidence

# --------- Get Gemini AI Suggestions ---------
def get_ai_suggestions(disease_name):
    try:
        prompt = f"""
        A plant has been diagnosed with: {disease_name}.
        Please give 3 clear, practical suggestions for treatment or prevention.
        Write them in bullet points and simple language suitable for farmers.
        Do not use any bold formatting or asterisks in your response.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Could not get AI suggestions: {e}"

# --------- Main UI Layout ---------

# Header
st.markdown("""
<div class="main-header fade-in">
    <h1 class="title">
        <span class="interactive-icon">🌿</span> 
        Plant Disease Detection
    </h1>
    <p class="subtitle">AI-powered plant health diagnosis for smarter farming</p>
</div>
""", unsafe_allow_html=True)

# Quick Stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-number">38+</div>
        <div class="stat-label">Disease Types</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-number">14</div>
        <div class="stat-label">Plant Species</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-number">91%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-number">⚡</div>
        <div class="stat-label">Instant Results</div>
    </div>
    """, unsafe_allow_html=True)

# Feature Cards
st.markdown('<div class="feature-grid">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3><span class="interactive-icon">🔬</span> Deep Learning AI</h3>
        <p>Advanced neural networks trained on plant leaf images for accurate disease detection</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3><span class="interactive-icon">⚡</span> Quick Analysis</h3>
        <p>Get instant diagnosis results with confidence scores and detailed explanations</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3><span class="interactive-icon">💡</span> Smart Recommendations</h3>
        <p>Receive AI-powered treatment suggestions tailored to your plant's specific condition</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("""
<div class="upload-container">
    <h2><span class="interactive-icon">📤</span> Upload Your Plant leaf Image</h2>
    <p>Take a clear photo of the plant leaf and upload it for instant analysis</p>
</div>
""", unsafe_allow_html=True)

uploaded_image = st.file_uploader(
    "Choose image file",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG (Max size: 200MB)"
)

if uploaded_image is not None:
    # Main analysis section with enhanced layout
    col1, col2 = st.columns([1.3, 1], gap="large")
    
    with col1:
        # Enhanced Image section
        st.markdown("""
        <div class="image-section fade-in">
            <div class="image-content">
                <h3 class="image-header">📸 Uploaded Image</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded plant leaf", use_container_width=True)
    
    with col2:
        # Enhanced Analysis section
        st.markdown("""
        <div class="analysis-section fade-in">
            <div class="analysis-content">
                <h3 class="analysis-header">
                    <span>🔍</span>
                    <span>AI Analysis Center</span>
                </h3>
        """, unsafe_allow_html=True)
        
        # Analysis button with enhanced styling
        if st.button('🚀 Start Analysis', key="analyze", help="Click to analyze your plant image"):
            
            # Progress indication
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                progress_text.text('🔄 Processing image...')
                progress_bar.progress(25)
                
                progress_text.text('🧠 Running AI analysis...')
                progress_bar.progress(50)
                
                predicted_class, result, confidence = predict_image_class(
                    model, uploaded_image, class_indices, disease_descriptions
                )
                
                progress_bar.progress(75)
                progress_text.text('✅ Analysis complete!')
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_text.empty()
                progress_bar.empty()
                
                # Show results
                if "healthy" in predicted_class.lower():
                    st.markdown(f"""
                    <h3 style="color: #27ae60; margin-top: 1.5rem;">🎉 Excellent News!</h3>
                    <div class="healthy-result fade-in">
                        <h4 style="color: #2d3748;">{result}</h4>
                        <div style="margin: 1rem 0;">
                            <strong style="color: #1a202c;">Confidence: {confidence:.1f}%</strong>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence}%"></div>
                            </div>
                        </div>
                        <p style="color: #2d3748;">✅ Your plant looks healthy! Continue with regular care.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <h3 style="color: #e53e3e; margin-top: 1.5rem;">⚠️ Disease Detected</h3>
                    <div class="disease-result fade-in">
                        <h4 style="color: #2d3748;">{result}</h4>
                        <div style="margin: 1rem 0;">
                            <strong style="color: #1a202c;">Confidence: {confidence:.1f}%</strong>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get AI suggestions
                    st.markdown("""
                    <h4 style="color: #ed8936; font-size: 1.2rem; font-weight: 600; margin-top: 1.5rem;">🧠 Treatment Recommendations</h4>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner('Getting AI recommendations...'):
                        suggestions = get_ai_suggestions(predicted_class)
                    
                    st.markdown(f"""
                    <div class="suggestions-container fade-in">
                        <div style="color: #2d3748;">{suggestions}</div>
                        <hr style="margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;">
                        <p style="color: #666;"><small>⚠️ <strong>Note:</strong> These are AI-generated suggestions. 
                        For severe cases, please consult agricultural experts.</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                progress_text.empty()
                progress_bar.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
        
        else:
            # Enhanced ready state
            st.markdown("""
                <div class="ready-state">
                    <h4>📋 Ready for Analysis</h4>
                    <p>Your image is uploaded and ready for AI analysis.</p>
                    <div class="ready-tips">
                        <div class="tip-item">
                            <span>✅</span>
                            <span>Focus on affected areas</span>
                        </div>
                        <div class="tip-item">
                            <span>✅</span>
                            <span>Single leaf preferred</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)

else:
    # Instructions when no image uploaded
    st.markdown("""
    <div class="result-container">
        <h2 style="color: #1a202c;">📋 Getting Started</h2>
        <p>Follow these simple steps to diagnose your plant health:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for enhanced step cards
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-icon">📱</div>
            <h3 style="color: #1a202c; margin: 1rem 0;">Capture Image</h3>
            <p style="flex-grow: 1;">Take a clear, well-lit photo of the plant leaf. Focus on any visible symptoms or healthy areas.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-icon">📤</div>
            <h3 style="color: #1a202c; margin: 1rem 0;">Upload File</h3>
            <p style="flex-grow: 1;">Use the file uploader above to select your plant image from your device.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-icon">🔍</div>
            <h3 style="color: #1a202c; margin: 1rem 0;">Get Results</h3>
            <p style="flex-grow: 1;">Click analyze to receive instant AI diagnosis with treatment recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced supported plants section
    st.markdown("""
    <div style="margin-top: 2rem; padding: 2rem; background: white; 
               border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
               text-align: center;">
        <h3 style="color: #1a202c; margin-bottom: 1rem;">🌱 Supported Plant Species</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                   gap: 1rem; margin-top: 1.5rem;">
            <div style="padding: 1rem; background: #f8fff8; border-radius: 8px; border-left: 3px solid #27ae60;">
                <strong>Fruits:</strong><br>Apple, Blueberry, Cherry, Grape, Orange, Peach, Raspberry, Strawberry
            </div>
            <div style="padding: 1rem; background: #fffef8; border-radius: 8px; border-left: 3px solid #f39c12;">
                <strong>Vegetables:</strong><br>Bell Pepper, Potato, Squash, Tomato
            </div>
            <div style="padding: 1rem; background: #f8f9ff; border-radius: 8px; border-left: 3px solid #3498db;">
                <strong>Crops:</strong><br>Corn (Maize), Soybean
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)