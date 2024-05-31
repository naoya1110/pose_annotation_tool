# https://flet.dev/docs/controls/image
# https://flet.dev/docs/controls/gesturedetector

import flet as ft
import cv2
import base64
import numpy as np
from PIL import Image
import os

from tools import draw_keypoints_on_picture
from tools import generate_dummy_keypoints
from tools import generate_label_text
from tools import read_annotation_data
from tools import generate_img_keypoints
from tools import find_nearest_coordinate
from tools import PersonKeypoints

from ultralytics import YOLO



modelpath = "models/yolov8n-pose.pt"
model = YOLO(modelpath) 

keypoints_list = []
nearest_idx = 0


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
        global keypoints_list
        
        filepath_img = e.files[0].path
        dir_images, filename = os.path.split(filepath_img)
        filename_body, _ = os.path.splitext(filename)
        dataset_dir, _ = os.path.split(dir_images)
        filepath_label = os.path.join(dataset_dir, "labels", filename_body+".txt")
                

        # OpenCVで画像を読み込み
        img_pic = np.array(Image.open(filepath_img))
        #img_pic = cv2.imread(filepath_img)
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
        
        # テキストファイルからアノテーションデータを読み取り
        detected_persons = read_annotation_data(filepath_label)    

        # キーポイント画像を生成
        img_keypoints, keypoints_list = generate_img_keypoints(img_pic, detected_persons)
        print(keypoints_list)
        
        # 写真とキーポイントデータを重ね合わせ
        img_result = draw_keypoints_on_picture(img_pic, img_keypoints)
        
        # image_displayのプロパティを更新
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
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
    
    def mouse_click(e: ft.TapEvent):
        global keypoints_list, nearest_idx
        x_loc.value = int(e.local_x)
        y_loc.value = int(e.local_y)        
        xy_clicked = (x_loc.value, y_loc.value)
        print("Clicked Point:", xy_clicked)
        
        nearest_idx, nearest_point = find_nearest_coordinate((xy_clicked), keypoints_list)
        print("Nearest Point:", nearest_idx, nearest_point)
    
    def mouse_drag(e: ft.DragUpdateEvent):
        global keypoints_list, nearest_idx
        x_loc.value = int(e.local_x)
        y_loc.value = int(e.local_y)        
        xy_drag = [x_loc.value, y_loc.value]
        print("Mouse Dragging:", xy_drag)
        
        keypoints_list[nearest_idx] = xy_drag
        print(keypoints_list)
        
    
    
    filepick_button = ft.ElevatedButton("Open Image", on_click=lambda _: file_picker.pick_files(allow_multiple=True))
    
    # 初期画像（ダミー）
    img_blank = 255*np.ones((300, 300, 3), dtype="uint8")
    img_h, img_w, _ = img_blank.shape
    image_display = ft.Image(src_base64=get_base64_img(img_blank),
                             width=img_w, height=img_h,
                             fit=ft.ImageFit.CONTAIN)
    
    gd = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        on_hover=mouse_on_hover,
        on_tap_down=mouse_click,
        on_horizontal_drag_update=mouse_drag)
    
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
    