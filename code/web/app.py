# web/app.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from inference import predict_audio
from utils import interpret
from generate_report import create_report

def analyze_and_report(audio_file):
    if audio_file is None:
        return "⚠️ Please upload or record an audio sample.", None, None
    
    try:
        pred, conf, raw = predict_audio(audio_file)
        md = interpret(pred, conf)
        pdf_name = create_report(pred, conf, file_in=os.path.basename(audio_file))
        
        # Use the interpret function from utils.py
        result_md = md
        
        return result_md, pdf_name, gr.update(visible=True)
    except Exception as e:
        return f"❌ Error during analysis: {e}", None, gr.update(visible=False)

with gr.Blocks() as ui:
    gr.HTML("""
        <div class="main-header">
            <h1>👶 CrySense AI</h1>
            <p style="font-size: 1.1em; opacity: 0.95;">Baby Cry Analyzer powered by Deep Learning</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Upload or Record Audio")
            audio_input = gr.Audio(
                type="filepath",
                label="Baby Cry Audio",
                elem_classes="audio-input"
            )
            gr.Markdown("""
            **Tips:**
            - 2-6 seconds recommended
            - Clear audio works best
            - Minimize background noise
            """)
            
            analyze_btn = gr.Button("🔍 Analyze Cry", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Analysis", value="Results will appear here...")
            
            with gr.Group(visible=False) as pdf_group:
                gr.Markdown("### 📄 Download Report")
                pdf_output = gr.File(label="PDF Report")
    
    gr.Markdown("""
    ---
    <div style="text-align: center; opacity: 0.7; font-size: 0.9em;">
        <p>⚠️ Research prototype • Not a substitute for professional medical advice</p>
    </div>
    """)
    
    analyze_btn.click(
        fn=analyze_and_report,
        inputs=[audio_input],
        outputs=[result_output, pdf_output, pdf_group]
    )

if __name__ == "__main__":
    ui.launch(share=False)
