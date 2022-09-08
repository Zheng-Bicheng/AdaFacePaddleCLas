import os
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle

from ppcls.arch.backbone.model_zoo.adaface_ir_net import AdaFace_IR_18
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

import cv2
import numpy as np
import yaml

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--mod",
        type=str,
        default="infer",
        help="0: create,1: infer")

    # img
    parser.add_argument(
        "--infer_img",
        type=str,
        default="./temp/demo_01.jpeg",
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")

    # detection model
    parser.add_argument(
        "--det_weights",
        type=str,
        default="./pretarined/blazeface_1000e.pdparams",
        help="The weights of detection model")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")

    # recognition model
    parser.add_argument(
        "--rec_config",
        type=str,
        default="RecModel",
        help="The config of recognition model")
    parser.add_argument(
        "--rec_weights",
        type=str,
        default="./output/RecModel/best_model.pdparams",
        help="The weights of recognition model")
    parser.add_argument(
        "--database",
        type=str,
        default="./database",
        help="The database of recognition model")
    args = parser.parse_args()
    return args

class FaceRecognition(object):
    def __init__(self, 
                cfg=None,
                det_target_size=640,
                interp=1,
                det_mean=[123, 117, 104],
                det_std=[127.502231, 127.502231, 127.502231],
                is_scale=False,
                threshold=0.7,
                rec_mean=[0.5, 0.5, 0.5],
                rec_std=[0.5, 0.5, 0.5],
                rec_target_size=112,
                database_dir="./database"):
        # save config
        self.det_cfg = cfg
        with open(cfg.rec_config) as f:
            self.rec_cfg = yaml.load(f)

        # input
        self.det_target_size = det_target_size
        self.rec_target_size = rec_target_size
        self.interp = interp
        self.det_std = det_std
        self.det_mean = det_mean
        self.is_scale = is_scale
        self.rec_std = det_std
        self.rec_mean = det_mean

        # build detection model
        self.det_model = create(cfg.architecture)  # read model
        load_pretrain_weight(self.det_model,cfg.det_weights)  # load weights
        self.det_model.eval()  # model to eval

        # output
        self.clsid2catid = {0:0}
        self.catid2name = {0: 'face'}
        self.threshold = threshold

        # build recognition model
        # print(self.rec_cfg)
        self.rec_model = AdaFace_IR_18()
        weights = paddle.load(cfg.rec_weights)
        self.rec_model.set_state_dict(weights)
        print(self.rec_model)
        self.database_dir = database_dir


    def predict_det(self,
                    images,
                    draw_threshold=0.5,
                    output_dir='output',
                    save_results=False,
                    visualize=True):

        # check output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # get img
        if isinstance(images, str):
            src_image = cv2.imread(images)
        else:
            src_image = images
        assert src_image is not None,"读取图片失败,img_path:{}".format(images)
        temp_image = src_image.copy()

        # preprocess
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        temp_image,im_info = self.resize(temp_image,
                                        [self.det_target_size,self.det_target_size])
        temp_image = self.normalizeImage(temp_image,self.det_mean,self.det_std)
        temp_image = temp_image.transpose((2, 0, 1))  # chw

        # Run Infer
        # get inputs
        inputs = {}
        inputs["im_shape"] = im_info["im_shape"]
        inputs["scale_factor"] = im_info["scale_factor"]
        inputs["im_id"] = [[0]]
        inputs["image"] = paddle.to_tensor([temp_image])

        # forward
        outs = self.det_model(inputs)

        # detector
        for key in ['im_shape', 'scale_factor', 'im_id']:
            outs[key] = inputs[key]
        for key, value in outs.items():
            if hasattr(value, 'numpy'):
                outs[key] = value.numpy()
        batch_res = get_infer_results(outs, self.clsid2catid)
        bbox_num = outs['bbox_num']
        bbox_res = batch_res['bbox']

        # draw
        # draw_img = draw_bbox(src_image,0,self.catid2name,bbox_res,self.threshold)
        # cv2.imwrite("./output/infer_outputs/result.jpg",draw_img)

        result = []
        for dt in np.array(bbox_res):
            catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
            temp_box_img = src_image.copy()
            if score < self.threshold:
                continue
            if len(bbox) == 4:
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                xmin,ymax,xmax,ymin = int(xmin),int(ymax),int(xmax),int(ymin)
                temp_box_img = temp_box_img[ymin:ymax+1,xmin:xmax+1]
            result.append(temp_box_img)
        return result

    def predict_rec(self,box_imgs):
        assert len(box_imgs)!=0,"没有目标"
        for img in box_imgs:
            # prepropress
            temp_box_img,im_info = self.resize(img,
                                        [self.rec_target_size,self.rec_target_size])
            temp_box_img = self.normalizeImage(temp_box_img,
                                                self.rec_mean,
                                                self.rec_std,
                                                is_scale=True)
            temp_box_img = temp_box_img.transpose((2, 0, 1))  # chw
            temp_box_img = paddle.to_tensor([temp_box_img])
            outs = self.rec_model(temp_box_img).numpy()
            self.get_similarity(outs)
            
    def update_database(self):
        for dir_name in os.listdir(cfg.database):
            print(dir_name)
            dir_path = os.path.join(cfg.database,dir_name)
            # 遍历文件夹
            if not os.path.isdir(dir_path):
                continue
            for pic_name in os.listdir(dir_path):
                if pic_name[0]==".":
                    continue
                img_path = os.path.join(dir_path,pic_name)
                if os.path.isdir(img_path):
                    continue
                print(img_path)
                box_results = self.predict_det(img_path)
                for box_img in box_results:
                    # prepropress
                    temp_box_img,im_info = self.resize(box_img,
                                        [self.rec_target_size,self.rec_target_size])
                    temp_box_img = self.normalizeImage(temp_box_img,
                                                self.rec_mean,
                                                self.rec_std,
                                                is_scale=True)
                    temp_box_img = temp_box_img.transpose((2, 0, 1))  # chw
                    temp_box_img = paddle.to_tensor([temp_box_img])
                    outs = self.rec_model(temp_box_img).numpy()
                    if not os.path.exists(os.path.join(dir_path,"save")):
                        os.mkdir(os.path.join(dir_path,"save"))
                    np.save(os.path.join(dir_path,"save",pic_name.split(".")[0]+".npy"),outs)
    
def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    img = image.copy()
    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue
        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = [int(t_c) for t_c in tuple(catid2color[catid])]
        # draw bbox
        if len(bbox) == 4:
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h

            xmin,ymax,xmax,ymin = int(xmin),int(ymax),int(xmax),int(ymin)
            cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 2)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            x1,y1,x2,y2, x3, y3, x4, y4 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)
            cv2.rectangle(img, (x2, y2), (x1, y1), color, 2)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            print('the shape of bbox must be [M, 4] or [M, 8]!')
            exit(0)

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, text, (xmin, ymin - 40), font, 1,
                              color, 1, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)
    check_config(cfg)
    face_recognition = FaceRecognition(cfg=cfg)

    if cfg.mod == "infer":
        det_results = face_recognition.predict_det(cfg.infer_img);
        face_recognition.predict_rec(det_results);
    elif cfg.mod == "create":
        face_recognition.update_database()
