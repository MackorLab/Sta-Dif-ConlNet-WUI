import gradio as gr

from diffusion_webui import (
    StableDiffusionControlNetGenerator,
    
)


def diffusion_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab(label="Text2Image"):
                    StableDiffusionText2ImageGenerator.app()
                with gr.Tab(label="Image2Image"):
                    StableDiffusionImage2ImageGenerator.app()
                with gr.Tab(label="Inpaint"):
                    StableDiffusionInpaintGenerator.app()
                with gr.Tab(label="Controlnet"):
                    StableDiffusionControlNetGenerator.app()
                with gr.Tab(label="Controlnet Inpaint"):
                    StableDiffusionControlNetInpaintGenerator.app()

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True)


diffusion_app(debug=True, max_threads=True, share=True, inbrowser=True)
