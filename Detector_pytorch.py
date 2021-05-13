from utils.BaseDetector import baseDet
from utils.models import *
from utils.config import *
from utils.utils import *

class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        print(".........",coco_weights[-2:])
        if coco_weights[-2:] == "pt":
            print("-------------------------------------")
            self.model = torch.load(coco_weights, map_location=self.device)['model'].float().eval()  # load FP32 model
        else:
            self.model = Darknet(coco_cfg, img_size).to(self.device).eval()
            load_darknet_weights(self.model, coco_weights)

        if self.half:
            self.model.half()  # to FP16

        self.names = load_classes(coco_names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(coco_names))]


    def transforms(self,image):
        """
        预处理
        :param image:原图
        :return: 处理后的图片
        """
        img = letterbox(image,new_shape=img_size)[0]  # resize img to 416
        img = img[:,:,::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def Process(self,pred,img,image):
        """
        网络输出后处理
        :param pred:网络输出值
        :param img: 预处理后的图
        :param image: 原图
        :param names: 类别名
        :param colors: 随机颜色
        :return: 原图和裁剪部分
        """
        pred_boxes = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                # Write results
                for *xyxy, conf, cls in det:
                    label = self.names[int(cls)]
                    if not label in ['person']:
                        continue
                    x1,y1 = int(xyxy[0]),int(xyxy[1])
                    x2,y2 = int(xyxy[2]),int(xyxy[3])
                    pred_boxes.append((x1,y1,x2,y2,label,conf))

        return image,pred_boxes

    def detect(self,image,conf_thres=0.3, iou_thres=0.4):
        """
        前向推理
        :param image:图片
        :param model: 模型
        :param conf_thres: 置信度阈值
        :param iou_thres: iou阈值
        :return: 原图和裁剪部分
        """
        # Run inference
        img = self.transforms(image)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        img0, class_name= self.Process(pred,img,image)

        return img0, class_name

