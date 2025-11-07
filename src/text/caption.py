from typing import Optional, Union

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

from src.util.text_utils import clean_markdown_code_blocks


class CaptionGenerator:
    def __init__(self, model_path: str):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model, self.output_loading_info = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", output_loading_info=True
        )

    def generate_caption(self, video: Union[str, list, tuple]) -> str:
        prompt = "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. Provide the result in json format with 'mm:ss.ff' format for time depiction."
        return self._inference(video, prompt)

    def _inference(
        self,
        video: Union[str, list, tuple],
        prompt: str,
        max_new_tokens: Optional[int] = 2048,
        total_pixels: Optional[int] = 20480 * 32 * 32,
        min_pixels: Optional[int] = 64 * 32 * 32,
        max_frames: Optional[int] = 2048,
        sample_fps: Optional[int] = 2,
    ):
        """
        Perform multimodal inference on input video and text prompt to generate model response.

        Args:
            video (str or list/tuple): Video input, supports two formats:
                - str: Path or URL to a video file. The function will automatically read and sample frames.
                - list/tuple: Pre-sampled list of video frames (PIL.Image or url).
                  In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
            prompt (str): User text prompt to guide the model's generation.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
            total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
            min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
            sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
                Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
                Used for temporal alignment or normalization in the model. Default is 2.

        Returns:
            str: Generated text response from the model.

        Notes:
            - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
            - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": video,
                        "total_pixels": total_pixels,
                        "min_pixels": min_pixels,
                        "max_frames": max_frames,
                        "sample_fps": sample_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        if video_kwargs is None:
            video_kwargs = {}
        elif not isinstance(video_kwargs, dict):
            try:
                video_kwargs = dict(video_kwargs)
            except Exception:
                video_kwargs = {}
        else:
            video_kwargs = {str(k): v for k, v in video_kwargs.items()}

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # Clean up markdown code blocks if present
        result = clean_markdown_code_blocks(output_text[0])
        
        return result
