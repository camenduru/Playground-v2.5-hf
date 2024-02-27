import gradio as gr
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import os
import spaces

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

@spaces.GPU(enable_queue=True)
def generate_image(prompt, num_inference_steps=50, guidance_scale=7):
    # Generate image
    results = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    # Safety check (if necessary)
    #if SAFETY_CHECKER:
        # Implement the safety check logic here
        #pass

    return results.images[0]

import gradio as gr

# Gradio Interface
description = """
This demo utilizes the playgroundai/playground-v2.5-1024px-aesthetic by Playground, which is a text-to-image generative model capable of producing high-quality images.
As a community effort, this demo was put together by AngryPenguin. Link to model: https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic
"""

with gr.Blocks() as demo:
    gr.Markdown(description)  # Display the description at the top of the interface
    gr.Markdown("## Playground-V2.5 Demo")

    # Prompt on its own row
    with gr.Row():
        prompt = gr.Textbox(label='Enter your image prompt')

    # Sliders for inference steps and guidance scale on another row
    with gr.Row():
        num_inference_steps = gr.Slider(minimum=1, maximum=75, step=1, label='Number of Inference Steps', value=50)
        guidance_scale = gr.Slider(minimum=1, maximum=10, step=0.1, label='Guidance Scale', value=5)

    # Submit button
    submit = gr.Button('Generate Image')

    # Image output at the bottom
    img = gr.Image(label='Generated Image')

    submit.click(
        fn=generate_image,  # This function needs to be defined to generate the image based on the inputs
        inputs=[prompt, num_inference_steps, guidance_scale],
        outputs=img,
    )

demo.queue().launch()