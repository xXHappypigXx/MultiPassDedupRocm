import os
import subprocess

import cv2
import torch
from tqdm import tqdm
import warnings
import _thread
import argparse
import time
import math
import numpy as np
from queue import Queue
from models.IFNet_HDv3 import IFNet
from models.gimm.src.utils.setup import single_setup
from models.gimm.src.models import create_model
from models.model_pg104.GMFSS import Model as GMFSS
from Utils_scdet.scdet import SvfiTransitionDetection

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='Interpolation a video with AFI-ForwardDeduplicate')
parser.add_argument('-i', '--video', dest='video', type=str, required=True, help='absolute path of input video')
parser.add_argument('-o', '--video_output', dest='video_output', required=True, type=str, default='output',
                    help='absolute path of output video')
parser.add_argument('-nf', '--n_forward', dest='n_forward', type=int, default=2,
                    help='the value of parameter n_forward')
parser.add_argument('-t', '--times', dest='times', type=int, default=2, help='the interpolation ratio')
parser.add_argument('-m', '--model_type', dest='model_type', type=str, default='gmfss',
                    help='the interpolation model to use (gmfss/rife/gimm)')
parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=False,
                    help='enable scene change detection')
parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=int, default=14,
                    help='scene detection threshold, same setting as SVFI')
parser.add_argument('-stf', '--shrink_transition_frames', dest='shrink', action='store_true', default=True,
                    help='shrink the copy frames in transition to improve the smoothness')
parser.add_argument('-c', '--enable_correct_inputs', dest='correct', action='store_true', default=True,
                    help='correct scene start and scene end processing, (will reduce stuttering, but will slow down the speed, and may introduce blur at beginning and ending of the scenes)')
parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                    help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=True,
                    help='enable hardware acceleration encode(require nvidia graph card)')
args = parser.parse_args()

model_type = args.model_type
n_forward = args.n_forward  # max_consistent_deduplication_counts - 1
times = args.times  # interpolation ratio >= 2
enable_scdet = args.enable_scdet  # enable scene change detection
scdet_threshold = args.scdet_threshold  # scene change detection threshold
shrink_transition_frames = args.shrink  # shrink the frames of transition
enable_correct_inputs = args.correct  # correct scene start and scene end processing
video = args.video  # input video path
video_output = args.video_output  # output img dir
scale = args.scale  # flow scale
hwaccel = args.hwaccel  # Use hardware acceleration video encoder

assert model_type in ['gmfss', 'rife', 'gimm'], f"not implement the model {model_type}"
# assert n_forward > 0, "the parameter n_forward must larger then zero"
assert times >= 2, "at least interpolate two times"

if not os.path.exists(video):
    raise FileNotFoundError(f"can't find the file {video}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# scene detection from SVFI
scene_detection = SvfiTransitionDetection("", 4,
                                          scdet_threshold=scdet_threshold,
                                          pure_scene_threshold=10,
                                          no_scdet=not enable_scdet,
                                          use_fixed_scdet=False,
                                          fixed_max_scdet=80,
                                          scdet_output=False)


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }


if model_type == 'rife':
    model = IFNet()
    model.load_state_dict(convert(torch.load('weights/rife48.pkl')))
elif model_type == 'gmfss':
    model = GMFSS()
    model.load_model('weights/train_log_pg104', -1)
else:
    args = argparse.Namespace(
        model_config=r"models/gimm/configs/gimmvfi/gimmvfi_r_arb.yaml",
        load_path=r"weights/gimmvfi_r_arb_lpips.pt",
        ds_factor=scale,
        eval=True,
        seed=0
    )
    config = single_setup(args)
    model, _ = create_model(config.arch)

    # Checkpoint loading
    if "ours" in args.load_path:
        ckpt = torch.load(args.load_path, map_location="cpu")


        def convert(param):
            return {
                k.replace("module.feature_bone", "frame_encoder"): v
                for k, v in param.items()
                if "feature_bone" in k
            }


        ckpt = convert(ckpt)
        model.load_state_dict(ckpt, strict=False)
    else:
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()
if model_type == 'gmfss':
    model.device()
else:
    model.to(device)

print("Loaded model")


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.


def to_numpy(tensor):
    return (tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.).astype(np.uint8)


def put(things):  # put frame to write_buffer
    write_buffer.put(things)


def get():  # get frame from read_buffer
    return read_buffer.get()


video_capture = cv2.VideoCapture(video)
width, height = map(int, map(video_capture.get, [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]))
pad_size = (64 / scale)
global_size = (int(math.ceil(width / pad_size) * pad_size), int(math.ceil(height / pad_size) * pad_size))
ori_fps = video_capture.get(cv2.CAP_PROP_FPS)

def build_read_buffer(r_buffer, v):
    ret, __x = v.read()
    while ret:
        r_buffer.put(cv2.resize(__x, global_size))
        ret, __x = v.read()
    r_buffer.put(None)


def generate_frame_renderer(input_path, output_path):
    encoder = 'libx264'
    preset = 'medium'
    if hwaccel:
        encoder = 'h264_nvenc'
        preset = 'p7'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-r', f'{ori_fps * times}',
        '-s', f'{width}x{height}',
        '-i', 'pipe:0', '-i', input_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', encoder, "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-qp", "16", '-preset', preset,
        '-c:a', 'aac', '-b:a', '320k', f'{output_path}'
    ]

    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


ffmpeg_writer = generate_frame_renderer(video, video_output)


def clear_write_buffer(w_buffer):
    global ffmpeg_writer
    while True:
        item = w_buffer.get()
        if item is None:
            break
        result = cv2.resize(item, (width, height))
        ffmpeg_writer.stdin.write(np.ascontiguousarray(result[:, :, ::-1]))
    ffmpeg_writer.stdin.close()
    ffmpeg_writer.wait()


total_frames_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))
pbar = tqdm(total=total_frames_count)

if n_forward == 0:
    n_forward = math.ceil(ori_fps / 24000 * 1001) * 2


@torch.inference_mode()
@torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
def make_inf(x, y, _scale, timestep):
    if model_type == 'rife':
        return model(torch.cat((x, y), dim=1), timestep)
    elif model_type == 'gmfss':
        return model.inference(x, y, model.reuse(x, y, _scale), timestep)
    else:
        xs = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), dim=2).to(
            device, non_blocking=True
        )
        model.zero_grad()
        with torch.no_grad():
            coord_inputs = [
                (
                    model.sample_coord_input(
                        xs.shape[0],
                        xs.shape[-2:],
                        [timestep],
                        device=xs.device,
                        upsample_ratio=_scale,
                    ),
                    None,
                )
            ]
            timesteps = [
                timestep * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
            ]
            all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=_scale)
            return [im for im in all_outputs["imgt_pred"]][0]


def decrease_inference(_inputs: list, layers=0, counter=0):
    while len(_inputs) != 1:
        layers += 1
        tmp_queue = []
        for i in range(len(_inputs) - 1):
            if saved_result.get(f'{layers}{i + 1}') is not None:
                saved_result[f'{layers}{i}'] = saved_result[
                    f'{layers}{i + 1}']
                tmp_queue.append(
                    saved_result[f'{layers}{i}']
                )
            else:
                inp0, inp1 = map(to_tensor, [_inputs[i], _inputs[i + 1]])
                tmp_queue.append(
                    to_numpy(make_inf(inp0, inp1, scale, 0.5))
                )
                saved_result[f'{layers}{i}'] = tmp_queue[-1]
                counter += 1
        _inputs = tmp_queue
    return _inputs[0], counter


# Modified from https://github.com/megvii-research/ECCV2022-RIFE/blob/main/inference_video.py
def correct_inputs(_inputs, n):
    def tmp_decrease_inference(_inputs: list, layers=0, counter=0):
        _save_dict = {}
        while len(_inputs) != 1:
            layers += 1
            tmp_queue = []
            for i in range(len(_inputs) - 1):
                if _save_dict.get(f'{layers}{i + 1}') is not None:
                    _save_dict[f'{layers}{i}'] = _save_dict[
                        f'{layers}{i + 1}']
                    tmp_queue.append(
                        _save_dict[f'{layers}{i}']
                    )
                else:
                    inp0, inp1 = map(to_tensor, [_inputs[i], _inputs[i + 1]])
                    tmp_queue.append(
                        to_numpy(make_inf(inp0, inp1, scale, 0.5))
                    )
                    _save_dict[f'{layers}{i}'] = tmp_queue[-1]
                    counter += 1
            _inputs = tmp_queue
        return _inputs[0], _save_dict, counter

    global model
    middle, save_dict, _ = tmp_decrease_inference(_inputs)
    if n == 1:
        return [middle]

    depth = int(max(save_dict.keys())[0])

    first_half_list = [_inputs[0]] + [save_dict[f'{layer}0'] for layer in range(1, depth, 1)]
    second_half_list = [save_dict[f'{layer}{depth - layer}'] for layer in range(depth, 0, -1)] + [_inputs[-1]]

    first_half = correct_inputs(first_half_list, n=n // 2)
    second_half = correct_inputs(second_half_list, n=n // 2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


@torch.inference_mode()
@torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
def gen_ts_frame(x, y, _scale, ts):
    _outputs = list()
    _reuse_things = model.reuse(x, y, _scale) if model_type == 'gmfss' else None
    for t in ts:
        if model_type == 'rife':
            _out = make_inf(x, y, _scale, t)
            _outputs.append(to_numpy(_out))
        elif model_type == 'gmfss':
            _out = model.inference(x, y, _reuse_things, t)
            _outputs.append(to_numpy(_out))
    if model_type == 'gimm':
        xs = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), dim=2).to(
            device, non_blocking=True
        )
        model.zero_grad()
        with torch.no_grad():
            coord_inputs = [
                (
                    model.sample_coord_input(
                        xs.shape[0],
                        xs.shape[-2:],
                        [t],
                        device=xs.device,
                        upsample_ratio=_scale,
                    ),
                    None,
                )
                for t in ts
            ]
            timesteps = [
                t * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                for t in ts
            ]
            all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=_scale)
            return [to_numpy(im) for im in all_outputs["imgt_pred"]]

    return _outputs


queue_input = [get()]

if queue_input[-1] is None:
    raise Exception("The input video does not have enough frames (< 1frame).")

queue_output = []
saved_result = {}
output0 = None
# if times = 5, n_forward=3, right=6, left=7
right_infill = (times * n_forward) // 2 - 1
left_infill = right_infill + (times * n_forward) % 2

if shrink_transition_frames:
    right_infill += times - 1

times_ts = [i / times for i in range(1, times)]

flag_exit = False
while True:
    if flag_exit:
        break

    if output0 is None:
        for _ in range(n_forward):
            queue_input.append(get())
            if queue_input[-1] is None:
                queue_input.pop(-1)
                flag_exit = True
        if len(queue_input) < 2:
            break

        output0, count = decrease_inference(queue_input.copy())

        queue_output.append(queue_input[0])
        inputs = [queue_input[0]]

        depth = int(max(saved_result.keys())[0])
        inputs.extend(saved_result[f'{layer}0'] for layer in range(1, depth + 1))

        if enable_correct_inputs and len(inputs) > 2:
            inputs = [inputs[0]] + correct_inputs(inputs, len(inputs) - 2) + [inputs[-1]]

        timestamp = [0.5 * layer for layer in range(0, n_forward + 1)]
        t_step = timestamp[-1] / (left_infill + 1)
        require_timestamp = [t_step * i for i in range(1, left_infill + 1)]

        for i in range(len(timestamp) - 1):
            t0, t1 = timestamp[i], timestamp[i + 1]

            if t0 in require_timestamp:
                queue_output.append(inputs[i])
                require_timestamp.remove(t0)

            condition_middle = [rt for rt in require_timestamp if t0 < rt < t1]
            if len(condition_middle) != 0:
                inp0, inp1 = map(to_tensor, [inputs[i], inputs[i + 1]])
                outputs = gen_ts_frame(inp0, inp1, scale, [(t - t0) * 2 for t in condition_middle])
                queue_output.extend(outputs)

            if t1 in require_timestamp:
                queue_output.append(inputs[i + 1])
                require_timestamp.remove(t1)

            if len(require_timestamp) == 0:
                break

    if not flag_exit:
        _ = queue_input.pop(0)
        queue_input.append(get())
        if queue_input[-1] is None:
            flag_exit = True

    if flag_exit or scene_detection.check_scene(queue_input[-2], queue_input[-1]):

        queue_output.append(output0)

        depth = int(max(saved_result.keys())[0])
        inputs = list(saved_result[f'{layer}{depth - layer}'] for layer in range(depth, 0, -1))
        inputs.append(queue_input[-2])

        timestamp = [0.5 * layer for layer in range(0, n_forward + 1)]
        t_step = timestamp[-1] / (right_infill + 1)
        require_timestamp = [t_step * i for i in range(1, right_infill + 1)]

        if enable_correct_inputs and len(inputs) > 2:
            inputs = [inputs[0]] + correct_inputs(inputs, len(inputs) - 2) + [inputs[-1]]

        for i in range(len(timestamp) - 1):
            t0, t1 = timestamp[i], timestamp[i + 1]

            if t0 in require_timestamp:
                queue_output.append(inputs[i])
                require_timestamp.remove(t0)

            condition_middle = [rt for rt in require_timestamp if t0 < rt < t1]
            if len(condition_middle) != 0:
                inp0, inp1 = map(to_tensor, [inputs[i], inputs[i + 1]])
                outputs = gen_ts_frame(inp0, inp1, scale, [(t - t0) * 2 for t in condition_middle])
                queue_output.extend(outputs)

            if t1 in require_timestamp:
                queue_output.append(inputs[i + 1])
                require_timestamp.remove(t1)

            if len(require_timestamp) == 0:
                break

        queue_output.append(queue_input[-2])

        if not shrink_transition_frames:
            queue_output.extend([queue_input[-2]] * (times - 1))

        for out in queue_output:
            put(out)

        if queue_input[-1] is None:
            break

        queue_input = [queue_input[-1]]
        queue_output = list()
        saved_result = dict()
        output0 = None
        pbar.update(1)
        continue

    output1, count = decrease_inference(queue_input.copy())

    queue_output.append(output0)
    inp0, inp1 = map(to_tensor, [output0, output1])
    queue_output.extend(gen_ts_frame(inp0, inp1, scale, times_ts))

    for out in queue_output:
        put(out)

    queue_output.clear()
    output0 = output1
    pbar.update(1)

print('Wait for all frames to be exported...')
while not write_buffer.empty():
    time.sleep(0.1)

pbar.update(1)
print('Done!')
