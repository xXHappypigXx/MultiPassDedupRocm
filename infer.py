import os
import torch
from tqdm import tqdm
import warnings
import argparse
import time
import math
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
                        help='enable hardware acceleration encode')
    return parser.parse_args()


def load_model():
    model = VFI(
        model_type=model_type,
        weights='weights',
        scale=scale,
        device=device
    )

    return model


def infer(cache_idx):
    global cache, head_end, tail_end, frame_idx  # Variables modified in this function are declared as global

    head = cache_idx == 0
    tail = cache_idx == len(cache.keys()) - 1

    # Only the head cache reads frames from video_io
    if head and len(cache[cache_idx]) != 2:
        I1 = video_io.read_frame()
        if I1 is None:
            head_end = True
            # When no frames available, repeat last frame in cache to prevent missing end frames
            cache[cache_idx].append(cache[cache_idx][0])
        else:
            cache[cache_idx].append(I1)

    # When both I0 and I1 are available, calculate intermediate frame Imid for next cache layer or output Imids
    if len(cache[cache_idx]) == 2:
        inp0 = cache[cache_idx][0]
        inp1 = cache[cache_idx][1]

        ts = [0.5]
        if tail:
            ts = mapper.get_range_timestamps(frame_idx, frame_idx + 1, lclose=True, rclose=head_end, normalize=True)
        if enable_scdet and check_scene(inp0, inp1, scdet_threshold):
            ts = [0 for _ in ts]

        if not tail:
            cache[cache_idx + 1].append(
                model.gen_ts_frame(inp0, inp1, ts)[0]
            )
        else:
            outputs = model.gen_ts_frame(inp0, inp1, ts)
            event = torch.cuda.Event()
            event.record()
            video_io.enqueue_frames(outputs, event)
            frame_idx += 1
            if head_end:
                # print('tail end')
                tail_end = True

        cache[cache_idx].pop(0)  # if-elif-else are checked sequentially, subsequent branches skipped after match
    elif len(cache[cache_idx]) == 1:
        # When frames are missing, fetch from previous layer. Repeat last frame when end flag encountered
        if head_end:
            cache[cache_idx].append(cache[cache_idx][0])
            infer(cache_idx)
        else:
            infer(cache_idx - 1)
    else:
        # This branch should never be executed
        raise ValueError(f"cache[{cache_idx}] should have 1 or 2 elements, but got {len(cache[cache_idx])}")


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

    mapper = TMapper(src_fps, target_fps, times)

    video_io.get_valid_net_inp_size(model.scale, model.pad_size)
    I0 = video_io.read_frame()

    # The cache is structured as {cache_idx: [I0, I1]},
    # with each layer initialized as {cache_idx: [I0]} to prevent missing initial frame.
    cache = {
        cache_idx: [I0] for cache_idx in range(n_pass)
    }

    head_end = False  # end sign for frameReader
    tail_end = False  # end sign for frameWriter
    frame_idx = 0  # frame index for only the last pass

    with tqdm(total=video_io.total_frames_count) as pbar:
        while not tail_end:
            for i in range(n_pass):
                infer(i)
            pbar.update(1)
    
    time.sleep(0.1)
    print('Wait for all frames to be exported...')
    while not video_io.finish_writing():
        time.sleep(0.1)

    print('Done!')