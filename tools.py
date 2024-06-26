import os
import numpy as np
import cv2
from scipy.spatial.distance import cdist



# 写真の上にキーポイントを重ね合わせる
def draw_keypoints_on_picture(img_pic, img_keypoints):
    mask = ~(img_keypoints == 0).all(axis=2)
    # img1の上にimg2を重ねる
    img_result = img_pic.copy()
    img_result[mask] = img_keypoints[mask]
    return img_result




def generate_label_text(results):
	num_detected_person = results.boxes.shape[0]
	print(num_detected_person)
	label_text = ""

	for i in range(num_detected_person):
		label_text += "0" # personのclass id
		
		box_data = results.boxes.xywhn[i].cpu().numpy()
		print(box_data)
		for data in box_data:
			label_text = label_text + f" {data:.5f}"
		
		xy_data = results.keypoints.xyn[i].cpu().numpy()
		conf_data = results.keypoints.conf[i].cpu().numpy()
		
		for (x, y), conf in zip(xy_data, conf_data):
			#print(x, y, conf)
			
			if x == 0 and y == 0:
				visibility = 0 # 写真の外
			elif conf < 0.5:
				visibility = 1 # 何かに隠れている
			else:
				visibility = 2 # 見えている
			
			label_text += f" {x:.5f} {y:.5f} {visibility}"
			
		if i < (num_detected_person-1):	
			label_text += "\n"

	return label_text


class PersonKeypoints:
    def __init__(self, class_id, box_xywhn, keypoints_xyvisib, img_h, img_w):
        self.class_id = class_id
        self.box_xywhn = box_xywhn
        self.keypoints_xyvisib = keypoints_xyvisib
        self.keypoints_dict = {}
        self.keypoints_name_list = [
                        "nose",
                        "left_eye", "right_eye",
                        "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder",
                        "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist",
                        "left_hip", "right_hip",
                        "left_knee","right_knee",
                        "left_ankle", "right_ankle",
                        "box_lt", "box_rt", "box_rb", "box_lb"
                        ]
        self.img_w = img_w
        self.img_h = img_h
        self.keypoint_xy_list = []
        self.nearest_point_name = ""
        self.nearest_idx = 0
        self.nearest_point = []
        self.nearest_point_distance = 0
        
        for i, name in enumerate(self.keypoints_name_list[:17]):
            xn=self.keypoints_xyvisib[i][0]
            yn=self.keypoints_xyvisib[i][1]
            x=int(self.keypoints_xyvisib[i][0]*self.img_w)
            y=int(self.keypoints_xyvisib[i][1]*self.img_h)
            visibility=int(self.keypoints_xyvisib[i][2])
            self.keypoints_dict[name] = {
                "xn":xn,
                "yn":yn,
                "x":x,
                "y":y,
                "visibility":visibility
            }
            self.keypoint_xy_list.append([x, y])
        
        xn, yn, wn, hn = self.box_xywhn
        left = int(round((xn-(wn/2))*self.img_w))
        right = int(round((xn+(wn/2))*self.img_w))
        top = int(round((yn-(hn/2))*self.img_h))
        bottom = int(round((yn+(hn/2))*self.img_h))
        
        self.keypoints_dict["box_lt"]={"xn":left/self.img_w, "yn":top/self.img_h, "x":left, "y":top, "visibility":2}
        self.keypoint_xy_list.append([left, top])
        self.keypoints_dict["box_rt"]={"xn":right/self.img_w, "yn":top/self.img_h, "x":right, "y":top, "visibility":2}
        self.keypoint_xy_list.append([right, top])
        self.keypoints_dict["box_rb"]={"xn":right/self.img_w, "yn":bottom/self.img_h, "x":right, "y":bottom, "visibility":2}
        self.keypoint_xy_list.append([right, bottom])
        self.keypoints_dict["box_lb"]={"xn":left/self.img_w, "yn":bottom/self.img_h, "x":left, "y":bottom, "visibility":2}
        self.keypoint_xy_list.append([left, bottom])
        
    
    def generate_xy_wire(self, names):
        
        xy_wire = []
        for name in names:
            if self.keypoints_dict[name]["visibility"] != 0:
                x = self.keypoints_dict[name]["x"]
                y = self.keypoints_dict[name]["y"]
                #if (x!=0) and (y!=0):
                xy_wire.append([x, y])
        xy_wire = np.array(xy_wire)
        
        return xy_wire
    
    def find_nearest_keypoint(self, point):

        self.keypoint_xy_list = []

        for name, data in self.keypoints_dict.items():
            visibility = data["visibility"]
            if visibility != 0:
                x=data["x"]
                y=data["y"]
            else:
                x = 1E+6
                y = 1E+6
            self.keypoint_xy_list.append([x, y])

        distances = cdist([point], self.keypoint_xy_list)

        # 最小距離のインデックスを取得
        self.nearest_idx = distances.argmin()
        self.nearest_point_distance = distances.min()
        self.nearest_point = self.keypoint_xy_list[self.nearest_idx]
        self.nearest_point_name = self.keypoints_name_list[self.nearest_idx]

        # 最も近い座標データを取り出す
        return self.nearest_idx, self.nearest_point_distance, self.nearest_point_name, self.nearest_point
    
    def update_point(self, name, xy):
        x = xy[0]
        y = xy[1]
        xn = x/self.img_w
        yn = y/self.img_h
        # print(name)
        # if name == "box_lt":
        #     self.keypoints_dict["box_lt"][x] = x
        #     self.keypoints_dict["box_lt"][y] = y
        #     self.keypoints_dict["box_lt"]["xn"] = xn
        #     self.keypoints_dict["box_lt"]["yn"] = yn
        #     self.keypoints_dict["box_lb"][x] = x
        #     self.keypoints_dict["box_rt"][y] = y
        #     self.keypoints_dict["box_lb"]["xn"] = xn
        #     self.keypoints_dict["box_rt"]["yn"] = yn
        
        self.keypoints_dict[name]["x"] = x
        self.keypoints_dict[name]["y"] = y
        self.keypoints_dict[name]["xn"] = xn
        self.keypoints_dict[name]["yn"] = yn
        
        if name == "box_lt":
            self.keypoints_dict["box_lb"]["x"] = x
            self.keypoints_dict["box_lb"]["xn"] = xn
            self.keypoints_dict["box_rt"]["y"] = y
            self.keypoints_dict["box_rt"]["yn"] = yn
            
        elif name == "box_rt":
            self.keypoints_dict["box_rb"]["x"] = x
            self.keypoints_dict["box_rb"]["xn"] = xn
            self.keypoints_dict["box_lt"]["y"] = y
            self.keypoints_dict["box_lt"]["yn"] = yn

        elif name == "box_rb":
            self.keypoints_dict["box_rt"]["x"] = x
            self.keypoints_dict["box_rt"]["xn"] = xn
            self.keypoints_dict["box_lb"]["y"] = y
            self.keypoints_dict["box_lb"]["yn"] = yn

        elif name == "box_lb":
            self.keypoints_dict["box_lt"]["x"] = x
            self.keypoints_dict["box_lt"]["xn"] = xn
            self.keypoints_dict["box_rb"]["y"] = y
            self.keypoints_dict["box_rb"]["yn"] = yn
    
        

def read_annotation_data(filepath_label, img_h, img_w):
    # filepath_labelを読み込み一人ごとにPersonKeypointsクラスのデータを作成
    with open(filepath_label, "r") as file:
        data = file.read()

    lines = data.split("\n")

    detected_persons = []

    for line in lines:
        if len(line)>10:
            data = line.split(" ")
            data = np.array(data, dtype="float32")
            class_id = int(data[0])
            box_xywhn = data[1:5]
            keypoints_xyvisib = data[5:].reshape(-1, 3)
            
            print(keypoints_xyvisib)

            detected_persons.append(PersonKeypoints(class_id, box_xywhn, keypoints_xyvisib, img_h, img_w))
    
    return detected_persons





def generate_img_keypoints(img_pic, detected_persons):

    img_keypoints = np.zeros(img_pic.shape, dtype="uint8")        
    img_h, img_w, _ = img_pic.shape
    keypoints_list = []
    
    leg_color=(255, 153, 51)   #orange  
    arm_color = (0, 255, 255) # Cyan
    face_color = (0, 255, 0) # Green
    body_color = (255, 0, 255) #Magenta
    box_color = (255, 0, 0) # Red

    for i, person in enumerate(detected_persons):

        xy_dict = {}
        
        for name, data in person.keypoints_dict.items():
            print(name)
            xn = data["xn"]
            yn = data["yn"]
            visibility = data["visibility"]
            x = int(round(xn*img_w))
            y = int(round(yn*img_h))
            
            xy_dict[name] = [x, y]
            
            if name in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]:
                color = face_color # Green
            
            elif name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]:
                color = arm_color # Cyan

            elif name in ["left_hip", "right_hip", "left_knee","right_knee", "left_ankle", "right_ankle"]:
                color = leg_color # Orange
            
            elif name in ["box_lt", "box_rt", "box_rb", "box_lb"]:
                color = box_color # Red
            
            keypoints_list.append([x, y])

            if visibility == 1:
                cv2.circle(img_keypoints, center=(x, y), radius=5, color=color, thickness=1, lineType=cv2.LINE_AA)
                #cv2.circle(img_keypoints, center=(x, y), radius=6, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                #cv2.putText(img_keypoints, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=1, lineType=cv2.LINE_AA)
            elif visibility == 2:
                #cv2.circle(img_keypoints, center=(x, y), radius=5, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(img_keypoints, center=(x, y), radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
                #cv2.putText(img_keypoints, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=1, lineType=cv2.LINE_AA)
        
 

        
        left_leg = person.generate_xy_wire(["left_hip", "left_knee", "left_ankle"])
        right_leg = person.generate_xy_wire(["right_hip", "right_knee", "right_ankle"])
        left_arm = person.generate_xy_wire(["left_shoulder", "left_elbow", "left_wrist"])
        right_arm = person.generate_xy_wire(["right_shoulder", "right_elbow", "right_wrist"])
        body = person.generate_xy_wire(["right_hip", "right_shoulder", "left_shoulder", "left_hip", "right_hip"])
        face = person.generate_xy_wire(["left_ear", "left_eye", "right_eye", "right_ear"])
        box = person.generate_xy_wire(["box_lt", "box_rt", "box_rb", "box_lb", "box_lt"])

        
        cv2.polylines(img_keypoints, [face], False, face_color, 1)
        cv2.polylines(img_keypoints, [body], False, body_color, 1)
        cv2.polylines(img_keypoints, [left_leg], False, leg_color, 1)
        cv2.polylines(img_keypoints, [right_leg], False, leg_color, 1)
        cv2.polylines(img_keypoints, [left_arm], False, arm_color, 1)
        cv2.polylines(img_keypoints, [right_arm], False, arm_color, 1)
        cv2.polylines(img_keypoints, [box], False, box_color, 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(left_leg) > 0:
            cv2.putText(img_keypoints, "L", left_leg[-1]+5, font, 0.7, leg_color,2)
        if len(right_leg) > 0:
            cv2.putText(img_keypoints, "R", right_leg[-1]+5, font, 0.7, leg_color,2)
        if len(left_arm) > 0:
            cv2.putText(img_keypoints, "L", left_arm[-1]+5, font, 0.7, arm_color,2)
        if len(right_arm) > 0:
            cv2.putText(img_keypoints, "R", right_arm[-1]+5, font, 0.7, arm_color,2)
        
        x = person.keypoints_dict["box_lt"]["x"]
        y = person.keypoints_dict["box_lt"]["y"]
        cv2.rectangle(img_keypoints, (x, y), (x+25, y+28), box_color, thickness=-1)
        cv2.putText(img_keypoints, f"{i}", (x+5,y+20), font, 0.7, (255,255,255), 2)
    
    return img_keypoints, keypoints_list


