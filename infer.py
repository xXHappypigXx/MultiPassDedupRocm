import os
import threading
from threading import Lock
import torch
from tqdm import tqdm
import warnings
import argparse
import time
import math
from queue import Queue
from models.vfi import VFI
from models.utils.tools import *

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation a video with MultiPassDedup')
    parser.add_argument('-i', '--video', dest='video', type=str, required=True, help='absolute path of input video')
    parser.add_argument('-o', '--video_output', dest='video_output', required=True, type=str, default='output',
                        help='absolute path of output video')
    parser.add_argument('-np', '--n_pass', dest='n_pass', type=int, default=3,
                        help='the value of parameter n_pass')
    parser.add_argument('-fps', '--target_fps', dest='target_fps', type=float, default=60, help='interpolate to ? fps')
    parser.add_argument('-t', '--times', dest='times', type=int, default=-1, help='interpolate to ?x fps')
    parser.add_argument('-m', '--model_type', dest='model_type', type=str, default='gmfss',
                        help='the interpolation model to use (gmfss/rife/gimm)')
    parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=False,
                        help='enable scene change detection')
    parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=float, default=0.3,
                        help='ssim scene detection threshold')
    parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                        help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
    parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=False,
                        help='enable hardware acceleration encode(require nvidia graph card)')
    return parser.parse_args()


def load_model():
    model = VFI(
        model_type=model_type,
        weights='weights',
        scale=scale,
        device=device
    )

    return model


def pass_infer(queue_idx: int):
    head = queue_idx == 0
    tail = queue_idx == len(queues)

    if head:
        inp_queue = video_io.read_buffer
    else:
        inp_queue = queues[queue_idx - 1]

    if tail:
        out_queue = video_io.write_buffer
    else:
        out_queue = queues[queue_idx]

    idx = 0
    i0 = inp_queue.get()
    if i0 is None:
        raise ValueError(f"video doesn't contains enough frames for infer with n_pass={n_pass}")

    size = get_valid_net_inp_size(i0, model.scale, div=model.pad_size)
    src_size, dst_size = size['src_size'], size['dst_size']

    I0 = to_inp(i0, dst_size)
    out_queue.put(i0)
    while True:
        i1 = inp_queue.get()
        if i1 is None:
            out_queue.put(i0)
            out_queue.put(None)
            event.set()
            break

        I1 = to_inp(i1, dst_size)

        scene_change = check_scene(I0, I1, scdet_threshold=scdet_threshold) if enable_scdet else False

        ts = [0.5]
        if tail:
            ts = mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=False, normalize=True)

        if scene_change:
            output = [I0 for _ in ts]
        else:
            with lock:  # avoid vram boom
                output = model.gen_ts_frame(I0, I1, ts)

        for out in output:
            out_queue.put(to_out(out, src_size))

        I0 = I1
        idx += 1
        pbar.update(1 / (len(queues) + 1))


if __name__ == '__main__':
    args = parse_args()
    model_type = args.model_type
    n_pass = args.n_pass  # max_consistent_deduplication_counts - 1
    target_fps = args.target_fps
    times = args.times  # interpolation ratio >= 2
    enable_scdet = args.enable_scdet  # enable scene change detection
    scdet_threshold = args.scdet_threshold  # scene change detection threshold
    video = args.video  # input video path
    video_output = args.video_output  # output img dir
    scale = args.scale  # flow scale
    hwaccel = args.hwaccel  # Use hardware acceleration video encoder

    assert model_type in ['gmfss', 'rife', 'gimm'], f"not implement the model {model_type}"

    model = load_model()

    if not os.path.exists(video):
        raise FileNotFoundError(f"can't find the file {video}")

    video_io = VideoFI_IO(video, video_output, dst_fps=target_fps, times=times, hwaccel=hwaccel)

    src_fps = video_io.src_fps
    if target_fps <= src_fps:
        raise ValueError(f'dst fps should be greater than src fps, but got tar_fps={target_fps} and src_fps={src_fps}')

    if n_pass == 0:
        n_pass = math.ceil(src_fps / 24000 * 1001) * 2

    pbar = tqdm(total=video_io.total_frames_count)
    mapper = TMapper(src_fps, target_fps, times)
    queues = [Queue(maxsize=100) for _ in range(n_pass - 1)]
    lock = Lock()  # global lock
    event = threading.Event()

    threads = []
    for _idx in range(len(queues) + 1):
        thread = threading.Thread(target=pass_infer, args=(_idx,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        event.wait()
        event.clear()

    print('Wait for all frames to be exported...')
    while not video_io.finish_writing():
        time.sleep(0.1)

    pbar.update(1)
    print('Done!')
