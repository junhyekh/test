import os
import numpy as np
import grpc
import requests
import tensorflow as tf
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class TransNetV2:

    def __init__(self, model_dir=None):

        self._input_size = (27, 48, 3)
        self.stub=prediction_service_pb2_grpc.PredictionServiceStub(
            grpc.insecure_channel('localhost:8500', options=(('grpc.enable_http_proxy', 0),)))
        

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = frames.astype(np.float32)
        grpc_request = predict_pb2.PredictRequest()
        grpc_request.model_spec.name='transnet'
        grpc_request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(frames))
        t0=time.time()
        result=self.stub.Predict(grpc_request)
        print(f'it takes {time.time()-t0}sec')
        logits=np.asarray(result.outputs['output_1'].float_val)
        dict_=np.asarray(result.outputs['output_2'].float_val)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_)

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )
            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []
        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[25:75],
                                all_frames_pred.numpy()[25:75]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        
        print("")
        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames
    
    def predict_video(self, video_fn: str):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError("For `predict_video` function `ffmpeg` needs to be installed in order to extract "
                                      "individual frames from video file. Install `ffmpeg` command line tool and then "
                                      "install python wrapper by `pip install ffmpeg-python`.")

        print("[TransNetV2] Extracting frames from {}".format(video_fn))
        try:
            video_stream, err = ffmpeg.input(video_fn).output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
            ).run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return (video, *self.predict_frames(video))

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

def main():
    import sys
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+", help="path to video files to process")
    parser.add_argument("--weights", type=str, default=None,
                        help="path to TransNet V2 weights, tries to infer the location if not specified")
    parser.add_argument('--visualize', action="store_true",
                        help="save a png file with prediction visualization for each extracted video")
    args = parser.parse_args()
    t0=time.time()
    model = TransNetV2()
    for file in args.files:
        if os.path.exists(file + ".predictions.txt") or os.path.exists(file + ".scenes.txt"):
            print(f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
                  f"Skipping video {file}.", file=sys.stderr)
            continue
        t1=time.time()
        video_frames, single_frame_predictions, all_frame_predictions = \
            model.predict_video(file)
        
        predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
        np.savetxt(file + ".predictions.txt", predictions, fmt="%.6f")

        scenes = model.predictions_to_scenes(single_frame_predictions)
        np.savetxt(file + ".scenes.txt", scenes, fmt="%d")
        
        print(f'it take {time.time()-t0}seconds from loading and {time.time()-t1}seconds from start')

        """
        if args.visualize:
            if os.path.exists(file + ".vis.png"):
                print(f"[TransNetV2] {file}.vis.png already exists. "
                      f"Skipping visualization of video {file}.", file=sys.stderr)
                continue

            pil_image = model.visualize_predictions(
                video_frames, predictions=(single_frame_predictions, all_frame_predictions))
            pil_image.save(file + ".vis.png")
        """

if __name__ == "__main__":
    main()