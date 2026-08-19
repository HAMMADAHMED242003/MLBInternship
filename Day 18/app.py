import cv2
import gradio as gr
import tempfile
import os


def process_video(video):

    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        return None


    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    output = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name


    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    writer = cv2.VideoWriter(
        output,
        fourcc,
        fps,
        (width, height)
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        edges = cv2.Canny(
            gray,
            100,
            200
        )


        edges_color = cv2.cvtColor(
            edges,
            cv2.COLOR_GRAY2BGR
        )


        writer.write(edges_color)


    cap.release()
    writer.release()


    print("Saved:", output)
    print("File size:", os.path.getsize(output))


    return output



demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(
        label="Upload Video"
    ),
    outputs=gr.Video(
        label="Processed Video"
    ),
    title="OpenCV Day 18 - Canny Edge Detection"
)


demo.launch()