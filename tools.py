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



def generate_keypoints_dict(data):
	points_dict = {
		1: {"label":"nose"},
		2: {"label":"left_eye"},
		3: {"label":"right_eye"},
		4: {"label":"left_ear"},
		5: {"label":"right_ear"},
		6: {"label":"left_shoulder"},
		7: {"label":"right_shoulder"},
		8: {"label":"left_elbow"},
		9: {"label":"right_elbow"},
		10: {"label":"left_wrist"},
		11: {"label":"right_wrist"},
		12: {"label":"left_hip"},
		13: {"label":"right_hip"},
		14: {"label":"left_knee"},
		15: {"label":"right_knee"},
		16: {"label":"left_ankle"},
		17: {"label":"right_ankle"}
		}

	for i in range(17):
		x, y, conf = data[i]
		points_dict[i+1]["xy"]=(x, y)
		
		if x == 0 and y == 0:
			visibility = 0
		elif conf < 0.5:
			visibility = 1
		else:
			visibility = 2
		
		points_dict[i+1]["visibility"]= visibility

	return points_dict