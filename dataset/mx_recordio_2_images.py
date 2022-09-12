import mxnet as mx
from tqdm import tqdm
from PIL import Image
import cv2
import numbers
import os
import argparse
from pathlib import Path


def save_rec_to_img_dir(rec_path, swap_color_channel=False, save_as_png=False):
    save_path = rec_path / os.path.basename(rec_path)
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'), str(rec_path / 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        if not isinstance(header.label, numbers.Number):
            label = int(header.label[0])
        else:
            label = int(header.label)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if swap_color_channel:
            # this option saves the image in the right color.
            # but the training code uses PIL (RGB)
            # and validation code uses Cv2 (BGR)
            # so we want to turn this off to deliberately swap the color channel order.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = Image.fromarray(img)
        label_path = save_path / str(label)
        if not label_path.exists():
            label_path.mkdir()

        if save_as_png:
            img_save_path = label_path / '{}.png'.format(idx)
            img.save(img_save_path)
        else:
            img_save_path = label_path / '{}.jpg'.format(idx)
            img.save(img_save_path, quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='./faces_emore', type=str)
    parser.add_argument("--swap_color_channel", action='store_true')
    args = parser.parse_args()

    rec_path = Path(args.rec_path)
    save_rec_to_img_dir(rec_path, swap_color_channel=args.swap_color_channel)
