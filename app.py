import os, spaces
from typing import TypeVar

from tqdm import tqdm
import gradio as gr
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.detr import RFDETR
from rfdetr.util.coco_classes import COCO_CLASSES

from utils.image import calculate_resolution_wh
from utils.video import create_directory, generate_unique_name

ImageType = TypeVar("ImageType", Image.Image, np.ndarray)

MARKDOWN = """
# RF-DETR 🔥

[`[code]`](https://github.com/roboflow/rf-detr) 
[`[blog]`](https://blog.roboflow.com/rf-detr) 
[`[notebook]`](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)

RF-DETR is a real-time, transformer-based object detection model architecture developed 
by [Roboflow](https://roboflow.com/) and released under the Apache 2.0 license.
"""

IMAGE_PROCESSING_EXAMPLES = [
    ['https://media.roboflow.com/supervision/image-examples/people-walking.png', 0.3, 728, "large"],
    ['https://media.roboflow.com/supervision/image-examples/vehicles.png', 0.3, 728, "large"],
    ['https://media.roboflow.com/notebooks/examples/dog-2.jpeg', 0.5, 560, "base"],
]
VIDEO_PROCESSING_EXAMPLES = [
    ["https://huggingface.co/spaces/SkalskiP/RF-DETR/resolve/main/videos/people-walking.mp4", 0.3, 728, "large"], 
    ["https://huggingface.co/spaces/SkalskiP/RF-DETR/resolve/main/videos/vehicles.mp4", 0.3, 728, "large"]
    ]


COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

MAX_VIDEO_LENGTH_SECONDS = 5
VIDEO_SCALE_FACTOR = 0.5
VIDEO_TARGET_DIRECTORY = "tmp"

create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

def detect_and_annotate(
        model: RFDETR,
        image: ImageType,
        confidence: float,
) -> ImageType:
    detections = model.predict(image, threshold=confidence)

    resolution_wh = calculate_resolution_wh(image)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh) - 0.2
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)

    bbox_annotator = sv.BoxAnnotator(color=COLOR, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        text_color=sv.Color.BLACK,
        text_scale=text_scale
    )

    labels = [
         f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    detection_results = []
    for i in range(len(detections.class_id)):
        detection_results.append({
         "class_id": detections.class_id[i],
         "classname": COCO_CLASSES[detections.class_id[i]],
         "confidence": detections.confidence[i],
         "bounding_box": detections.xyxy[i].tolist()
        })
    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)
    return {'annotated_image': annotated_image, 
        'results': detection_results}
        
@spaces.GPU
def load_model(resolution: int, checkpoint: str) -> RFDETR:
    if checkpoint == "base":
        return RFDETRBase(resolution=resolution)
    elif checkpoint == "large":
        return RFDETRLarge(resolution=resolution)
    raise TypeError("Checkpoint must be a base or large.")


def image_processing_inference(
        input_image: Image.Image,
        confidence: float,
        resolution: int,
        checkpoint: str
) -> Image.Image:
    model = load_model(resolution=resolution, checkpoint=checkpoint)
    return detect_and_annotate(model=model, image=input_image, confidence=confidence)['annotated_image']


def video_processing_inference(
        input_video: str,
        confidence: float,
        resolution: int,
        checkpoint: str,
        progress=gr.Progress(track_tqdm=True)
):
    model = load_model(resolution=resolution, checkpoint=checkpoint)

    name = generate_unique_name()
    output_video = os.path.join(VIDEO_TARGET_DIRECTORY, f"{name}.mp4")

    video_info = sv.VideoInfo.from_video_path(input_video)
    video_info.width = int(video_info.width * VIDEO_SCALE_FACTOR)
    video_info.height = int(video_info.height * VIDEO_SCALE_FACTOR)

    total = min(video_info.total_frames, video_info.fps * MAX_VIDEO_LENGTH_SECONDS)
    frames_generator = sv.get_video_frames_generator(input_video, end=total)

    with sv.VideoSink(output_video, video_info=video_info) as sink:
        for frame in tqdm(frames_generator, total=total):
            annotated_frame = detect_and_annotate(
                model=model,
                image=frame,
                confidence=confidence
            )['annotated_image']
            annotated_frame = sv.scale_image(annotated_frame, VIDEO_SCALE_FACTOR)
            sink.write_frame(annotated_frame)

    return output_video

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Tab("Image"):
        with gr.Row():
            image_processing_input_image = gr.Image(
                label="Upload image",
                image_mode='RGB',
                type='pil',
                height=600
            )
            image_processing_output_image = gr.Image(
                label="Output image",
                image_mode='RGB',
                type='pil',
                height=600
            )
        with gr.Row():
            with gr.Column():
                image_processing_confidence_slider = gr.Slider(
                    label="Confidence",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                )
                image_processing_resolution_slider = gr.Slider(
                    label="Inference resolution",
                    minimum=560,
                    maximum=1120,
                    step=56,
                    value=728,
                )
                image_processing_checkpoint_dropdown = gr.Dropdown(
                    label="Checkpoint",
                    choices=["base", "large"],
                    value="base"
                )
            with gr.Column():
                image_processing_submit_button = gr.Button("Submit", value="primary")

        gr.Examples(
            fn=image_processing_inference,
            examples=IMAGE_PROCESSING_EXAMPLES,
            inputs=[
                image_processing_input_image,
                image_processing_confidence_slider,
                image_processing_resolution_slider,
                image_processing_checkpoint_dropdown
            ],
            outputs=image_processing_output_image,
            cache_examples=True,
            run_on_click=True
        )

        image_processing_submit_button.click(
            image_processing_inference,
            inputs=[
                image_processing_input_image,
                image_processing_confidence_slider,
                image_processing_resolution_slider,
                image_processing_checkpoint_dropdown
            ],
            outputs=image_processing_output_image,
        )
    # with gr.Tab("Video"):
    #     with gr.Row():
    #         video_processing_input_video = gr.Video(
    #             label='Upload video',
    #             height=600
    #         )
    #         video_processing_output_video = gr.Video(
    #             label='Output video',
    #             height=600
    #         )
    #     with gr.Row():
    #         with gr.Column():
    #             video_processing_confidence_slider = gr.Slider(
    #                 label="Confidence",
    #                 minimum=0.0,
    #                 maximum=1.0,
    #                 step=0.05,
    #                 value=0.5,
    #             )
    #             video_processing_resolution_slider = gr.Slider(
    #                 label="Inference resolution",
    #                 minimum=560,
    #                 maximum=1120,
    #                 step=56,
    #                 value=728,
    #             )
    #             video_processing_checkpoint_dropdown = gr.Dropdown(
    #                 label="Checkpoint",
    #                 choices=["base", "large"],
    #                 value="base"
    #             )
    #         with gr.Column():
    #             video_processing_submit_button = gr.Button("Submit", value="primary")

    #     gr.Examples(
    #         fn=video_processing_inference,
    #         examples=VIDEO_PROCESSING_EXAMPLES,
    #         inputs=[
    #             video_processing_input_video,
    #             video_processing_confidence_slider,
    #             video_processing_resolution_slider,
    #             video_processing_checkpoint_dropdown
    #         ],
    #         outputs=video_processing_output_video,
    #         run_on_click=True
    #     )

    #     video_processing_submit_button.click(
    #         video_processing_inference,
    #         inputs=[
    #             video_processing_input_video,
    #             video_processing_confidence_slider,
    #             video_processing_resolution_slider,
    #             video_processing_checkpoint_dropdown
    #         ],
    #         outputs=video_processing_output_video
    #     )

demo.launch(debug=False, show_error=True)
