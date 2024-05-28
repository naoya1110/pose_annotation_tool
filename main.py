# https://flet.dev/docs/controls/image
# https://flet.dev/docs/controls/gesturedetector

import flet as ft
import cv2
import base64
import numpy as np
from PIL import Image
import os

from tools import generate_keypoint_img
from tools import draw_keypoints_on_picture
from tools import generate_dummy_keypoints
from tools import generate_label_text

from ultralytics import YOLO

from tools import PersonKeypoints

modelpath = "models/yolov8n-pose.pt"
model = YOLO(modelpath) 


def main(page: ft.Page):
    
    # page(アプリ画面)の設定
    page.title = "HUMAN POSE ANNOTATION TOOL"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.colors.INDIGO_50
    page.padding = 50
    page.window_height = 500
    page.window_width = 800
    
    IMG_SIZE= 300

    # numpy画像データをbase64にエンコードする関数
    def get_base64_img(img):
        _, encoded = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(encoded).decode("ascii")
        return img_base64
    
    # ファイルが選択された時のコールバック
    def on_img_open(e: ft.FilePickerResultEvent):
        filepath_img = e.files[0].path
        dir_images, filename = os.path.split(filepath_img)
        filename_body, _ = os.path.splitext(filename)
        dataset_dir, _ = os.path.split(dir_images)
        filepath_label = os.path.join(dataset_dir, "labels", filename_body+".txt")
                

        # OpenCVで画像を読み込み
        #img_pic = np.array(Image.open(filepath))
        img_pic = cv2.imread(filepath_img)
        img_pic = cv2.resize(img_pic, dsize=None, fx=0.5, fy=0.5)
        img_h, img_w, _ = img_pic.shape
        
        # filepath_labelが存在するか確認し，なければYOLOで推論する
        if not os.path.exists(filepath_label):
            print(f"{filepath_label} does not exist.")
            print(f"Trying to detect keypoints...")
            # YOLO poseで推論
            results = model(img_pic)[0]
            label_text = generate_label_text(results)
            
            with open(filepath_label, "w") as file:
                file.write(label_text)
                print(f"{filepath_label} generated!")
        else:
            print(f"{filepath_label} already exists.")
            
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
        
        print(detected_persons)

        # Boxを描画
        img_keypoints = np.zeros(img_pic.shape, dtype="uint8")        
        for person in detected_persons:
            xn, yn, wn, hn = person.box_xywhn
            left = int(round((xn-(wn/2))*img_w))
            right = int(round((xn+(wn/2))*img_w))
            top = int(round((yn-(hn/2))*img_h))
            bottom = int(round((yn+(hn/2))*img_h))
            
            cv2.rectangle(img_keypoints, (left, top), (right, bottom), (255,0,0), 2, 1) 
        # 2025.5.28 ここまで
        
        # 写真の上にアノテーションデータを描画
        img_result = draw_keypoints_on_picture(img_pic, img_keypoints)
        
        # image_displayのプロパティを更新
        image_display.src_base64 = get_base64_img(img_result)
        image_display.height = img_h
        image_display.width = img_w
        
        # stackのプロパティを更新
        stack.height = img_h
        stack.width = img_w
        
        # pageをアップデート
        page.update()
    
    def mouse_on_hover(e: ft.HoverEvent):
        x_loc.value = int(e.local_x)
        y_loc.value = int(e.local_y)
        page.update()
    
    
    filepick_button = ft.ElevatedButton("Open Image", on_click=lambda _: file_picker.pick_files(allow_multiple=True))
    
    # 初期画像（ダミー）
    img_blank = 255*np.ones((300, 300, 3), dtype="uint8")
    img_h, img_w, _ = img_blank.shape
    image_display = ft.Image(src_base64=get_base64_img(img_blank),
                             width=img_w, height=img_h,
                             fit=ft.ImageFit.CONTAIN)
    
    gd = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        on_hover=mouse_on_hover)
    
    x_loc_label = ft.Text("X", size=20)
    x_loc = ft.Text("0", size=20)
    y_loc_label = ft.Text("Y", size=20)
    y_loc = ft.Text("0", size=20)
    
    mouse_loc = ft.Column([ft.Row([x_loc_label, x_loc]),
                           ft.Row([y_loc_label, y_loc])])
    
    stack = ft.Stack([image_display, gd], width=IMG_SIZE, height=IMG_SIZE)
        
    page.add(filepick_button)
    page.add(ft.Row([stack, mouse_loc]))
    
    file_picker = ft.FilePicker(on_result=on_img_open)
    page.overlay.append(file_picker)
    page.update()
    



# デスクトップアプリとして開く
ft.app(target=main)

# webアプリとして開く場合は任意のポート番号を指定し
# ブラウザでlocalhost:7777を開く
# ft.app(target=main, port=7777)
    