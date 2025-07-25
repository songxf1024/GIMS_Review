import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import requests
import torch
import math
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import os
import glob
import re
import torch.distributed as torch_dist
import torch.backends.cudnn as cudnn
import random
from copy import deepcopy
import torchvision
from tqdm import tqdm
from .library import buildGaussianPyramid, ComputePatches
from .preprocess_utils import scale_homography, torch_find_matches


superpoint_url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superpoint_v1.pth"
superglue_indoor_url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_indoor.pth"
superglue_outdoor_url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_outdoor.pth"
superglue_cocohomo_url = "https://github.com/gouthamk1998/files/releases/download/1.0/release_model.pt"
coco_test_images_url = "https://github.com/gouthamk1998/files/releases/download/1.0/coco_test_images.zip"
indoor_test_images_url = "https://github.com/gouthamk1998/files/releases/download/1.0/indoor_test_images.zip"
outdoor_test_imags_url = "https://github.com/gouthamk1998/files/releases/download/1.0/outdoor_test_images.zip"
weights_mapping = {
        'superpoint': Path(__file__).parent.parent / 'models/weights/superpoint_v1.pth',
        'indoor': Path(__file__).parent.parent / 'models/weights/superglue_indoor.pth',
        'outdoor': Path(__file__).parent.parent / 'models/weights/superglue_outdoor.pth',
        'coco_homo': Path(__file__).parent.parent / 'models/weights/superglue_cocohomo.pt'
    }

test_images_mapping = {
    'coco_test_images': coco_test_images_url,
    'indoor_test_images': indoor_test_images_url,
    'outdoor_test_images': outdoor_test_imags_url
}

def download_test_images():
    for i,k in test_images_mapping.items():
        zip_path = Path(__file__).parent.parent / ('assets/' + i + '.zip')
        directory_path = Path(__file__).parent.parent / ('assets/' + i)
        if not directory_path.exists():
            print("Downloading and unzipping {}...".format(i))
            os.system("curl -L {} -o {}".format(k, str(zip_path)))
            os.system("unzip {} -d {}".format(str(zip_path), str(directory_path.parent)))
            os.remove(str(zip_path))

def download_base_files():
    directory = Path(__file__).parent.parent / 'models/weights'
    if not directory.exists(): os.makedirs(str(directory))
    superpoint_path = weights_mapping['superpoint']
    indoor_path = weights_mapping['indoor']
    outdoor_path = weights_mapping['outdoor']
    coco_homo_path = weights_mapping['coco_homo']
    command = "curl -L {} -o {}"
    if not superpoint_path.exists():
        print("Downloading superpoint model...")
        os.system(command.format(superpoint_url, str(superpoint_path)))
    if not indoor_path.exists():
        print("Downloading superglue indoor model...")
        os.system(command.format(superglue_indoor_url, str(indoor_path)))
    if not outdoor_path.exists():
        print("Downloading superglue outdoor model...")
        os.system(command.format(superglue_outdoor_url, str(outdoor_path)))
    if not coco_homo_path.exists():
        print("Downloading coco homography model...")
        os.system(command.format(superglue_cocohomo_url, str(coco_homo_path)))

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)

def clean_checkpoint(ckpt):
    new_ckpt = {}
    for i,k in ckpt.items():
        if i[0:6] == "module":
            new_ckpt[i[7:]] = k
        else:
            new_ckpt[i] = k
    return new_ckpt

def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()

def reduce_tensor(inp, avg=True):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    if not avg: return reduced_inp
    return reduced_inp / world_size

class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device, color):
    if color:
        return np.expand_dims(frame, axis=0)  # .transpose(2, 0, 1)
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales

def read_image_with_homography(path, homo_matrix, device, resize, rotation, resize_float, color=False):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if image is None: return None, None, None
    w, h = image.shape[1], image.shape[0]
    warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (w, h))
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image if color else image.astype('float32'), (w_new, h_new))
        warped_image = cv2.resize(warped_image if color else warped_image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)) if color else cv2.resize(image, (w_new, h_new)).astype('float32')
        warped_image = cv2.resize(warped_image, (w_new, h_new)) if color else cv2.resize(warped_image, (w_new, h_new)).astype('float32')
    if rotation != 0:
        image = np.rot90(image, k=rotation)
        warped_image = np.rot90(warped_image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    inp = frame2tensor(image, device, color)
    warped_inp = frame2tensor(warped_image, device, color)
    scaled_homo = scale_homography(homo_matrix, h, w, h_new, w_new).astype(np.float32)
    return image, warped_image, inp, warped_inp, scales, scaled_homo



# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)

def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)

def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d

def compute_pixel_error(pred_points, gt_points):
    diff = gt_points - pred_points
    diff = (diff ** 2).sum(-1)
    sqrt = np.sqrt(diff)
    return sqrt.mean()

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()

def debug_image_plot(debug_path, keypoints0, keypoints1, match_list0, match_list1, image0, image1, epoch, it):
    np_image0, np_image1 = (image0*255).astype(np.uint8)[0], (image1*255).astype(np.uint8)[0]
    np_image0, np_image1 = cv2.cvtColor(np_image0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(np_image1, cv2.COLOR_GRAY2BGR)
    kp0, kp1 = keypoints0.detach().cpu().numpy(), keypoints1.detach().cpu().numpy()
    ma0, ma1 = match_list0.detach().cpu().numpy()[:25], match_list1.detach().cpu().numpy()[:25]
    for i,k in zip(kp0, kp1):
        cv2.circle(np_image0, (int(i[0]), int(i[1])), 2, (255, 0, 0), 1)
        cv2.circle(np_image1, (int(k[0]), int(k[1])), 2, (0, 0, 255), 1)
    write_image = np.concatenate([np_image0, np_image1], axis=1)
    for key1, key2 in zip(ma0, ma1):
        k1, k2 = kp0[key1], kp1[key2]
        cv2.line(write_image, (int(k1[0]), int(k1[1])), (int(k2[0]) + 640, int(k2[1])),
                 color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    write_path = os.path.join(debug_path, "{}_{}.jpg".format(epoch, it))
    cv2.imwrite(write_path, write_image)
    return write_image

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)

def weighted_score(results):
    weight = [0.0, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1]
    values = [results['dlt_auc'][0], results['dlt_auc'][1], results['dlt_auc'][2], results['ransac_auc'][0],
              results['ransac_auc'][1], results['ransac_auc'][2], results['precision'], results['recall']]
    weight_score = (np.array(weight) * np.array(values)).sum()
    return weight_score

def compute_sift_at_locations(sift, image, coordinates):
    keypoints = [cv2.KeyPoint(x, y, 1) for x, y in coordinates]
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors

def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp = desc
    if not torch.is_tensor(desc):
        temp = torch.Tensor(desc).to(device)
    temp = temp / temp.pow(2).sum(dim=1, keepdim=True).add(1e-10).pow(0.5)
    if torch.is_tensor(desc):
        return temp
    return temp.cpu().numpy()

def rootSIFT(descs, eps=1e-7, l2norm=False):
    # Use the Hellinger kernel for l1 normalization
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    # Find the square root for each element
    descs = np.sqrt(descs)
    # Whether to perform l2 normalization is somewhat inconsistent. In the RootSIFT paper, it is not pointed out that l2 normalization is needed, but in presentation, l2 normalization is present.
    # It is also believed that explicitly performing L2 normalization is not necessary. By adopting the L1 specification, followed by the square root, there are already L2 standardized eigenvectors, and no further standards are needed.
    if l2norm:
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        descs = desc_l2norm(descs)
    return descs

def filterMaxNumDesc(kp, MaxNum):
    if 0 < MaxNum < len(kp):
        responses = [k.response for k in kp]
        idxs = np.fliplr(np.reshape(np.argsort(responses), (1, -1))).reshape(-1)
        kpF = []
        for n in range(MaxNum): kpF.append(kp[idxs[n]])
        return kpF
    else:
        return kp

def diou_nms(dets, scores, iou_thresh=None, beta=1.0):
    """DIOU non-maximum suppression.
  diou = iou - Square of Euclidean distance in the center of the box / Square of the smallest surrounding the diagonal of the box
  Reference: https://arxiv.org/pdf/1911.08287.pdf
  Args:
    dets: detection with shape (num, 4) and format [x1, y1, x2, y2].
    iou_thresh: IOU threshold,
    Parameter β is used to control the degree of punishment for distance. 
            When β tends to infinity, DIoU degenerates into IoU, and the DIoU-NMS at this time is comparable to the standard NMS. 
            When β tends to 0, almost all boxes that do not coincide with the center point of the box with the largest score are retained.
  Returns:
    numpy.array: Retained boxes.
  """
    iou_thresh = iou_thresh or 0.5
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        smallest_enclosing_box_x1 = np.minimum(x1[i], x1[order[1:]])
        smallest_enclosing_box_x2 = np.maximum(x2[i], x2[order[1:]])
        smallest_enclosing_box_y1 = np.minimum(y1[i], y1[order[1:]])
        smallest_enclosing_box_y2 = np.maximum(y2[i], y2[order[1:]])

        square_of_the_diagonal = (
                (smallest_enclosing_box_x2 - smallest_enclosing_box_x1) ** 2 +
                (smallest_enclosing_box_y2 - smallest_enclosing_box_y1) ** 2)

        square_of_center_distance = ((center_x[i] - center_x[order[1:]]) ** 2 +
                                     (center_y[i] - center_y[order[1:]]) ** 2)

        # Add 1e-10 for numerical stability.
        diou = iou - np.power(square_of_center_distance / (square_of_the_diagonal + 1e-10), beta)
        inds = np.where(diou <= iou_thresh)[0]
        order = order[inds + 1]
    return dets[keep]

def process_diou_nms(keypoints, radius=None, iou_thresh=0.3):
    if radius == 0: return keypoints
    scores = np.array([k.response for k in keypoints]).astype(np.float32)
    if radius:
        x1 = lambda center: center.pt[0] - radius / 2
        y1 = lambda center: center.pt[1] - radius / 2
        x2 = lambda center: center.pt[0] + radius / 2
        y2 = lambda center: center.pt[1] + radius / 2
    else:
        x1 = lambda center: center.pt[0] - center.size / 2
        y1 = lambda center: center.pt[1] - center.size / 2
        x2 = lambda center: center.pt[0] + center.size / 2
        y2 = lambda center: center.pt[1] + center.size / 2
    # Extract x,y coordinates
    dets = np.array([[x1(k), y1(k), x2(k), y2(k)] for k in keypoints]).astype(np.float32)
    # The key points after filtering, note that the order may change
    try:
        res = diou_nms(dets, scores, iou_thresh)  # , beta=1e5)
    except Exception as e:
        raise Exception(f"NMS has problems: {e}, {dets.shape}, {scores.shape}")
    indexes = []
    # Match the point after searching for filtering in the original number
    for item in res:
        i = np.argwhere((dets[:, 0] == item[0]) & (dets[:, 1] == item[1]) & (dets[:, 2] == item[2]) & (dets[:, 3] == item[3]))
        if i.size: indexes.append(i[0][0])
    if type(keypoints) != np.ndarray:
        kpt_new = np.array(keypoints)[indexes].tolist()
    else:
        kpt_new = keypoints[indexes]
    return np.array(kpt_new)

def gpu_warmup(cuda, num_warmup_steps=50):
    """Preheat the GPU"""
    model = torchvision.models.resnet18()
    input_tensor = torch.randn(1, 3, 224, 224)
    model = model.to(cuda)
    input_tensor = input_tensor.to(cuda)
    model.train()
    for _ in tqdm(range(num_warmup_steps), desc='GPU Warmup...', ncols=80):
        output = model(input_tensor)
        output.backward(torch.ones_like(output))
        torch.cuda.synchronize()
    del model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def send_notify(msg):
    try:
        requests.get('http://wxbot.xfxuezhang.cn/send/friend?target=wxid_0438974384722&key=lihua&msg='+msg)
    except:
        pass

def sift_forward(data, device, copy=False):
    gconfig = {
        "nfeatures": None,
        "nOctaveLayers": 3,
        "maxoctaves": 4,
        "contrastThreshold": 0.001,
        "edgeThreshold": 80,
        "sigma": 1.6,
        "max_keypoints": data.get("max_keypoints", -1),
        "iou_borders": 4,
        "iou_thresh": 0.3,
    }
    is_train = data.get("is_train", False)

    sift = cv2.SIFT_create(
        nfeatures=gconfig["nfeatures"],
        nOctaveLayers=gconfig["nOctaveLayers"],
        contrastThreshold=gconfig["contrastThreshold"],
        edgeThreshold=gconfig["edgeThreshold"],
        sigma=gconfig["sigma"]
    )
    kpts, ori_kpts, descs, scores, patches = [], [], [], [], []
    for img in data['image']:
        t1 = time.time()
        k = sift.detect(img, None)
        print('>> Keypoint Detection:', time.time() - t1)
        # k = process_diou_nms(k, gconfig["iou_borders"], gconfig["iou_thresh"])
        k = filterMaxNumDesc(k, gconfig["max_keypoints"])
        # For less than a certain number of key points, add some pixel positions randomly.
        if is_train and len(k) < gconfig["max_keypoints"]:
            print(f"Rare condition executed [{len(k)}/{gconfig['max_keypoints']}]")
            to_add_points = gconfig["max_keypoints"] - len(k)
            # Randomly select some from existing key points and descriptors to copy
            if copy:
                selected_indices = np.random.choice(len(k), size=to_add_points, replace=True)
                additional_k, additional_d = k[selected_indices], d[selected_indices]
            # Randomly select some coordinates to calculate SIFT features
            else:
                coordinates = np.random.random((to_add_points, 2)) * img.shape[1]
                coordinates[:, 1] = np.random.random(to_add_points) * img.shape[0]
                additional_k, additional_d = compute_sift_at_locations(sift, img, coordinates)
            # Add selected key points and descriptors to the original list
            k = np.concatenate([k, additional_k]) if len(k)>0 else additional_k
        print('>> Number of Keypoints:', len(k))
        t1 = time.time()
        pyramid = buildGaussianPyramid(img, gconfig["maxoctaves"] + 2, graydesc=False)
        pts = ComputePatches(k, pyramid, radius_size=64)
        pts = np.array([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA) for p in pts]) / 255.0
        print('>> Patches Generation:', time.time() - t1)
        t1 = time.time()
        k, d = data['carhynet'].compute_sift(pts, k, True)
        print('>> Descriptor Generation:', time.time() - t1)
        ori_kpts.append(k)
        kpts.append(torch.tensor([i.pt for i in k]).to(device))
        descs.append(torch.cat([torch.tensor(d), torch.tensor(d)], dim=1).permute(1, 0).to(device))
        scores.append(torch.tensor([i.response for i in k]).to(device))
    return {'keypoints': kpts, 'scores': scores, 'descriptors': descs}

def find_pred(inp, gmodel):
    pred = {}
    if 'keypoints0' not in inp:
        pred0 = sift_forward({'image': inp['image0'], 'max_keypoints': -1, 'carhynet': inp['carhynet']}, device=inp['device'])
        pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
    if 'keypoints1' not in inp:
        pred1 = sift_forward({'image': inp['image1'], 'max_keypoints': -1, 'carhynet': inp['carhynet']}, device=inp['device'])
        pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
    data = {**inp, **pred}
    for k in data:
        if k.startswith("ori_keypoints"):
            data[k] = np.stack(data[k])
        elif isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k]).to(inp["device"])
    pred = {**pred, **gmodel(data, **{'our': inp['our']})}
    return pred

def test_model(test_loader, gmodel, val_count, device, min_matches=12, carhynet=None):
    gmodel.eval()
    all_recall, all_precision, all_error_dlt, all_error_ransac = [], [], [], []
    for i_no, (orig_warped, homography) in enumerate(test_loader):
        if i_no >= val_count: break
        homography = homography[0].to(device)
        orig_image, warped_image = orig_warped[0:1, :, :, :], orig_warped[1:2, :, :, :]
        pred = find_pred({'image0': orig_image, 'image1': warped_image, 'carhynet': carhynet, 'device': device}, gmodel)
        kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
        pred = {k: (v[0] if k in ['ori_keypoints0', 'ori_keypoints1'] else v[0].cpu().numpy()) for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        if len(mconf) < min_matches:
            all_precision.append(0)
            all_recall.append(0)
            all_error_dlt.append(500)
            all_error_ransac.append(500)
            continue
        ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, homography, dist_thresh=3, n_iters=3)
        ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
        gt_match_vec = np.ones((len(matches), ), dtype=np.int32) * -1
        gt_match_vec[ma_0] = ma_1
        corner_points = np.array([[0,0], [0, orig_image.shape[2]], [orig_image.shape[3], orig_image.shape[2]], [orig_image.shape[3], 0]]).astype(np.float32)
        sort_index = np.argsort(mconf)[::-1][0:4]
        est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
        est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)
        corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
        corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(1)
        corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), homography.cpu().numpy()).squeeze(1)
        error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
        error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
        match_flag = (matches[ma_0] == ma_1)
        precision = match_flag.sum() / valid.sum()
        fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
        if (match_flag.sum() + fn_flag.sum()) == 0:
            all_precision.append(0)
            all_recall.append(0)
            all_error_dlt.append(500)
            all_error_ransac.append(500)
            continue
        recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
        all_precision.append(precision)
        all_recall.append(recall)
        all_error_dlt.append(error_dlt)
        all_error_ransac.append(error_ransac)
    thresholds = [5, 10, 25]
    aucs_dlt = pose_auc(all_error_dlt, thresholds)
    aucs_ransac = pose_auc(all_error_ransac, thresholds)
    aucs_dlt = [float(100.*yy) for yy in aucs_dlt]
    aucs_ransac = [float(100.*yy) for yy in aucs_ransac]
    prec = float(100.*np.mean(all_precision))
    rec = float(100.*np.mean(all_recall))
    results_dict = {'dlt_auc': aucs_dlt, 'ransac_auc': aucs_ransac, 'precision': prec, 'recall': rec, 'thresholds': thresholds}
    weight_score = weighted_score(results_dict)
    results_dict['weight_score'] = float(weight_score)
    print("For DLT results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec))
    print("For homography results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec))
    return results_dict

def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ 
    #Taken from yolov5 repo
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 4000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
