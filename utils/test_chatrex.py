from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize

# Initialize model
model = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers"  # or "vllm"
)

# Load image
image = Image.open("/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/SAM/images/images/sa_223752.jpg")

# Object Detection
results = model.inference(
    images=image,
    task="detection",
    categories=["Two square skylight panels embedded in the overhanging white architectural structure at the bottom left"]
)

result = results[0]

# 4) Visualize
vis = RexOmniVisualize(
    image=image,
    predictions=result["extracted_predictions"],
    font_size=20,
    draw_width=5,
    show_labels=True,
)
vis.save("visualize.jpg")