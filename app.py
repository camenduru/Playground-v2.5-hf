import gradio as gr
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import os

# Constants
#SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", "0") == "1"

# Initialize the model
pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Safety Checker (if necessary)
#if SAFETY_CHECKER:
    # Implement or import the safety checker code here

def generate_image(prompt, num_inference_steps=50, guidance_scale=7):
    # Generate image
    results = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    # Safety check (if necessary)
    if SAFETY_CHECKER:
        # Implement the safety check logic here
        pass

    return results.images[0]

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Playground-V2.5 Demo")
    with gr.Row():
        prompt = gr.Textbox(label='Enter your image prompt')
        num_inference_steps = gr.Slider(minimum=1, maximum=75, step=1, label='Number of Inference Steps', value=50)
        guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, label='Guidance Scale', value=7)
        submit = gr.Button('Generate Image')
    img = gr.Image(label='Generated Image')

    submit.click(
        fn=generate_image,
        inputs=[prompt, num_inference_steps, guidance_scale],
        outputs=img,
    )


demo.queue().launch()