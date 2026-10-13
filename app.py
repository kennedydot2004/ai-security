import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# 1. Page Configuration & Title
st.set_page_config(
    page_title="AI Operations Word Cloud Generator",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Interactive AI Operations & Security Word Cloud")
st.write("Customize keywords, adjust formatting styles, and export high-res images ready for social media.")

# 2. Hardcoded Domain Keywords
preset_security = """
Security Guardrails Prompt-Injection Data-Leakage PII-Masking Model-Poisoning 
Adversarial-Attacks Jailbreaking RBAC Encryption Vulnerability-Scanning Anomaly-Detection
Toxicity-Scoring Bias-Detection Privacy-Compliance Model-Defense Firewall OWASP-Top-10
"""

preset_telemetry = """
Telemetry Observability OpenTelemetry Trace-ID Metric-Aggregation Structured-Logging 
Latency Throughput Saturation Context-Propagation Span Distributed-Tracing
Real-Time-Monitoring Logs Metrics Traces Performance-Bottlenecks APM Dashboard
"""

preset_full_llm_ops = """
Token-Consumption TTFT Perplexity Hallucination-Rate Semantic-Similarity ROUGE 
BLEU Cost-Tracking Vector-Database Embedding RAG Cache-Rate Model-Versioning Lineage
Prompt-Engineering Fine-Tuning Inference GPU-Utilization Context-Window Evaluation
"""

# 3. Sidebar Configuration Layout
st.sidebar.header("🎨 Styling & Configurations")

# Choose Color Schemes
color_theme = st.sidebar.selectbox(
    "Select Color Palette",
    options=["viridis", "plasma", "inferno", "magma", "cool", "spring", "copper", "ocean"]
)

# Choose Background Tone
bg_color = st.sidebar.selectbox(
    "Background Color",
    options=["black", "white", "darkblue", "charcoal"]
)
# Map conversational names to hex/string values for the generator
bg_map = {"black": "black", "white": "white", "darkblue": "#0A192F", "charcoal": "#212529"}
resolved_bg = bg_map[bg_color]

# Aspect Ratio Adjustments
dimensions = st.sidebar.radio(
    "Social Media Format Aspect Ratio",
    options=["Standard Web (1200 x 800)", "Square/Instagram (1080 x 1080)", "Banner/X (1500 x 500)"]
)

# Resolution Resolution Mapping
if "Standard" in dimensions:
    w, h = 1200, 800
elif "Square" in dimensions:
    w, h = 1080, 1080
else:
    w, h = 1500, 500

# 4. Main Workspace Interface Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Your Text Data")
    
    # Preset selection helpers
    preset_choice = st.radio(
        "Load an AI Engineering Vocabulary Preset:",
        options=["Strict AI Security & Guardrails", "Telemetry & Observability", "Combined Full LLMOps Stack", "Clear Canvas (Empty)"]
    )
    
    # Initialize session text depending on preset selection
    default_text = ""
    if preset_choice == "Strict AI Security & Guardrails":
        default_text = preset_security.strip()
    elif preset_choice == "Telemetry & Observability":
        default_text = preset_telemetry.strip()
    elif preset_choice == "Combined Full LLMOps Stack":
        default_text = preset_security.strip() + " " + preset_telemetry.strip() + " " + preset_full_llm_ops.strip()
        
    user_input = st.text_area(
        "Edit words or paste custom entries here (separate words with spaces or newlines):",
        value=default_text,
        height=320
    )

with col2:
    st.subheader("🖼️ Real-Time Preview Layout")
    
    if not user_input.strip():
        st.warning("Please type or select a preset on the left panel to populate the layout graphic.")
    else:
        # Generate word cloud logic
        wc = WordCloud(
            width=w,
            height=h,
            background_color=resolved_bg,
            colormap=color_theme,
            collocations=False,  # Stops duplication of split word patterns
            random_state=42     # Maintains layout position balance upon parameter adjustments
        )

        if not wc.process_text(user_input):
            st.warning("Please enter at least one word with two or more characters.")
            st.stop()

        wc.generate(user_input)
        
        # Render visual plotting elements natively via Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        
        # Display the visual plot element directly to Streamlit screen
        st.pyplot(fig)
        
        # Save figure internally to RAM memory buffer for download operations
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", bbox_inches='tight', pad_inches=0, dpi=300)
        img_buffer.seek(0)
        plt.close(fig)
        
        # Action button to trigger binary downloads 
        st.download_button(
            label="💾 Download High-Res PNG for Social Media",
            data=img_buffer,
            file_name="ai_ops_wordcloud.png",
            mime="image/png",
            use_container_width=True
        )

st.info("💡 Pro Tip: To make specific terms appear significantly larger on the cloud layout canvas, repeat them multiple times in the text input box panel on the left.")
