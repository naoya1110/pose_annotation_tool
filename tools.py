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


# ダミーのキーポイントデータ生成
def generate_dummy_keypoints():
    
    points_dict = {
        1: {"label":"nose", "xy":(0,0), "visibility":2},
        2: {"label":"left_eye", "xy":(0,0), "visibility":2},
        3: {"label":"right_eye", "xy":(0,0), "visibility":2},
        4: {"label":"left_ear", "xy":(0,0), "visibility":2},
        5: {"label":"right_ear", "xy":(0,0), "visibility":2},
        6: {"label":"left_shoulder","xy":(0,0), "visibility":2},
        7: {"label":"right_shoulder","xy":(0,0), "visibility":2},
        8: {"label":"left_elbow","xy":(0,0), "visibility":2},
        9: {"label":"right_elbow","xy":(0,0), "visibility":2},
        10: {"label":"left_wrist","xy":(0,0), "visibility":2},
        11: {"label":"right_wrist","xy":(0,0), "visibility":2},
        12: {"label":"left_hip","xy":(0,0), "visibility":2},
        13: {"label":"right_hip","xy":(0,0), "visibility":2},
        14: {"label":"left_knee","xy":(0,0), "visibility":2},
        15: {"label":"right_knee","xy":(0,0), "visibility":2},
        16: {"label":"left_ankle","xy":(0,0), "visibility":2},
        17: {"label":"right_ankle", "xy":(0,0), "visibility":2}
    }
    
    for i in range(17):
        x = np.random.randint(300)
        y = np.random.randint(1000)
        points_dict[i+1]["xy"] = (x,y)
        points_dict[i+1]["visibility"] = np.random.randint(3)
    
    return points_dict


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
    def __init__(self, class_id, box_xywhn, keypoints_xyvisib):
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
                        "left_ankle", "right_ankle"
                        ]
        
        for i in range(17):
            self.keypoints_dict[i] = {
                "name":self.keypoints_name_list[i],
                "xn":self.keypoints_xyvisib[i][0],
                "yn":self.keypoints_xyvisib[i][1],
                "visibility":int(self.keypoints_xyvisib[i][2])
            }

def read_annotation_data(filepath_label):
    # filepath_labelを読み込み一人ごとにPersonKeypointsクラスのデータを作成
    with open(filepath_label, "r") as file:
        data = file.read()

    lines = data.split("\n")

    detected_persons = []

    for line in lines:
        data = line.split(" ")
        data = np.array(data, dtype="float32")
        class_id = int(data[0])
        box_xywhn = data[1:5]
        keypoints_xyvisib = data[5:].reshape(-1, 3)

        detected_persons.append(PersonKeypoints(class_id, box_xywhn, keypoints_xyvisib))
    
    return detected_persons


def generate_img_keypoints(img_pic, detected_persons):

    img_keypoints = np.zeros(img_pic.shape, dtype="uint8")        
    img_h, img_w, _ = img_pic.shape
    keypoints_list = []
    
    leg_color=(255, 153, 51)   #orange  
    arm_color = (0, 255, 255) # Cyan
    face_color = (0, 255, 0) # Green
    body_color = (255, 0, 255) #Magenta

    for person in detected_persons:
        xn, yn, wn, hn = person.box_xywhn
        left = int(round((xn-(wn/2))*img_w))
        right = int(round((xn+(wn/2))*img_w))
        top = int(round((yn-(hn/2))*img_h))
        bottom = int(round((yn+(hn/2))*img_h))
        cv2.rectangle(img_keypoints, (left, top), (right, bottom), color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
        
        xy_dict = {}
        
        for i, data in person.keypoints_dict.items():
            name = data["name"]
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
            
            keypoints_list.append([x, y])

            if visibility == 1:
                #cv2.circle(img_keypoints, center=(x, y), radius=5, color=color, thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(img_keypoints, center=(x, y), radius=4, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                #cv2.putText(img_keypoints, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=1, lineType=cv2.LINE_AA)
            elif visibility == 2:
                #cv2.circle(img_keypoints, center=(x, y), radius=5, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(img_keypoints, center=(x, y), radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
                #cv2.putText(img_keypoints, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=1, lineType=cv2.LINE_AA)
        
        
        # draw wire frame
        left_leg = np.array([xy_dict[name] for name in ["left_hip", "left_knee", "left_ankle"] if xy_dict[name]!=[0,0]])
        right_leg = np.array([xy_dict[name] for name in ["right_hip", "right_knee", "right_ankle"] if xy_dict[name]!=[0,0]])
        left_arm = np.array([xy_dict[name] for name in ["left_shoulder", "left_elbow", "left_wrist"] if xy_dict[name]!=[0,0]])
        right_arm = np.array([xy_dict[name] for name in ["right_shoulder", "right_elbow", "right_wrist"] if xy_dict[name]!=[0,0]])
        body = np.array([xy_dict[name] for name in ["right_hip", "right_shoulder", "left_shoulder", "left_hip", "right_hip"] if xy_dict[name]!=[0,0]])
        face = np.array([xy_dict[name] for name in ["left_ear", "left_eye", "right_eye", "right_ear"] if xy_dict[name]!=[0,0]])
        cv2.polylines(img_keypoints, [face], False, face_color, 2)
        cv2.polylines(img_keypoints, [body], False, body_color, 2)
        cv2.polylines(img_keypoints, [left_leg], False, leg_color, 2)
        cv2.polylines(img_keypoints, [right_leg], False, leg_color, 2)
        cv2.polylines(img_keypoints, [left_arm], False, arm_color, 2)
        cv2.polylines(img_keypoints, [right_arm], False, arm_color, 2)
            
    
    return img_keypoints, keypoints_list



def find_nearest_coordinate(target_point, points):

    points = np.array(points)
    distances = cdist([target_point], points)

    # 最小距離のインデックスを取得
    nearest_idx = distances.argmin()
    nearest_point = points[nearest_idx]

    # 最も近い座標データを取り出す
    return nearest_idx, nearest_point