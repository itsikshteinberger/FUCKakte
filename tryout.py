import cv2
import io
import requests
from PIL import Image
import torch
import numpy
import time
'''
from segment.predict import new_main
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

cv2.imwrite('C:/Users/user/Downloads/images/hack.png', frame)
t0 = time.time()
new_img = new_main()
print((time.time() - t0) * 10**3)
cv2.imwrite('C:/Users/user/Downloads/hack.png', new_img)'''











import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

#..... Tracker modules......
import skimage
from segment.sort_count import *
import numpy as np
import pygame
#...........................


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode


class MODEL():
    def __init__(self):
        opt = self.parse_opt()
        return self.run()

    def run(self, weights='C:/Users/user/Downloads/yolov7seg/yolov7-seg.pt',  # model.pt path(s)
    source='C:/Users/user/Downloads/hack.png',  # file/dir/URL/glob, 0 for webcam
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    trk = False):

        sort_max_age = 5 
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        sort_tracker = Sort(max_age=sort_max_age,
                            min_hits=sort_min_hits,
                            iou_threshold=sort_iou_thresh) 
        #......................... 

        self.source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def run_model(self):
        visualize=False
        augment=False
        conf_thres=0.25
        iou_thres=0.45
        classes = 0
        agnostic_nms=False
        max_det=1000

        imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=1000  # maximum detections per image
        view_img=False  # show results
        save_txt=False  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=0  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        visualize=False  # visualize features
        update=False  # update all models
        project=ROOT / 'runs/predict-seg'  # save results to project/name
        name='exp'  # save results to project/name
        exist_ok=False  # existing project/name ok, do not increment
        line_thickness=3  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        half=False  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference
        trk = False

        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        save_img = False
        # Dataloader
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs


        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, out = self.model(im, augment=augment, visualize=visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Segments
                    if save_txt:
                        segments = reversed(masks2segments(masks))
                        segments = [scale_segments(im.shape[2:], x, im0.shape).round() for x in segments]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(6), True) for cls in det[:, 5]]
                    im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                    annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                    # Mask plotting ----------------------------------------------------------------------------------------

                    if trk:
                        #Tracking ----------------------------------------------------
                        dets_to_sort = np.empty((0,6))
                        for x1,y1,x2,y2,conf,detclass in det[:, :6].cpu().detach().numpy():
                            dets_to_sort = np.vstack((dets_to_sort, 
                                            np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))

                        tracked_dets = sort_tracker.update(dets_to_sort)
                        tracks =sort_tracker.getTrackers()

                        for track in tracks:
                            annotator.draw_trk(line_thickness,track)

                        if len(tracked_dets)>0:
                            bbox_xyxy = tracked_dets[:,:4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            annotator.draw_id(bbox_xyxy, identities, categories, self.names)
                
                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        if save_txt:  # Write to file
                            segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                            line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            #annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                if len(det) == 0:
                    im_masks = cv2.imread('C:/Users/user/Downloads/hack3.png')
                    print('hi')
                # Stream results
                cv2.imwrite('C:/Users/user/Downloads/hack2.png', im0)
                cv2.imwrite('C:/Users/user/Downloads/hack3.png', im_masks)
                im0 = im_masks-cv2.resize(im0, (im_masks.shape[1], im_masks.shape[0]), interpolation = cv2.INTER_LINEAR)#annotator.result()
                #im0 = im_masks-im0
                cv2.imwrite('C:/Users/user/Downloads/hack1.png', im0)
                return im0

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov7-seg.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=ROOT / 'C:/Users/user/Downloads/images/hack.png', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', default=0, nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--trk', action='store_true', help='Apply Sort Tracking')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        #print_args(vars(opt))
        return opt


    def cross(self, image, mask):
        cv2.imwrite('C:/Users/user/Downloads/hack.png', image)
        seg = self.run_model()
        if not isinstance(mask, np.ndarray):
            mask = pygame.surfarray.array3d(mask)
        gray_img = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_img, gray_mask, cv2.TM_CCOEFF_NORMED)
        return res.max()


if __name__ == "__main__":
    segModel = MODEL()
    t0 = time.time()
    #segModel.run_model()
    for i in range(1):
        t1 = time.time()
        cv2.imwrite('C:/Users/user/Downloads/hack1.png', segModel.run_model())
        #print(segModel.run_model().shape)
        print(time.time()-t1)
    #print(t1-t0)