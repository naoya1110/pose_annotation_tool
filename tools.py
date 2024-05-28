import os
import numpy as np
import cv2

# Keypoint画像を作成
def generate_keypoint_img(img_pic, points_dict):
    img_keypoints = np.zeros(img_pic.shape, dtype="uint8")
    
    for key, value in points_dict.items():
        if value["visibility"] != 0:
            if value["visibility"] == 2:
                thickness = -1
                # 白丸
                img_keypoints = cv2.circle(
                                    img_keypoints,
                                    np.array(value["xy"]).astype("int"),
                                    radius=15,
                                    color = (255,255,255),
                                    thickness=thickness)
            else:
                thickness = 3
            img_keypoints = cv2.circle(
                                    img_keypoints,
                                    np.array(value["xy"]).astype(int),
                                    radius=10,
                                    color = (255,0,0),
                                    thickness=thickness)
    
    return img_keypoints



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