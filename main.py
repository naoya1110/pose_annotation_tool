# https://flet.dev/docs/controls/image
# https://flet.dev/docs/controls/gesturedetector

import flet as ft
import cv2
import base64
import numpy as np
from PIL import Image
from tools import generate_keypoint_img
from tools import draw_keypoints_on_picture
from tools import generate_dummy_keypoints
from tools import generate_keypoints_dict

from ultralytics import YOLO


model = YOLO("models/yolov8n-pose.pt") 


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
        filepath = e.files[0].path

        # OpenCVで画像を読み込み
        #img_pic = np.array(Image.open(filepath))
        img_pic = cv2.imread(filepath)
        img_pic = cv2.resize(img_pic, dsize=None, fx=0.5, fy=0.5)
        img_h, img_w, _ = img_pic.shape
        
        
        # YOLO poseで推論
        result = model(img_pic)
        bbox_data_array = result[0].boxes.xyxy.cpu().numpy()
        keypoints_data_array = result[0].keypoints.data.cpu().numpy()
        
        # 推論結果を描画
        for i in range(bbox_data_array.shape[0]):
            # draw bbox
            left, top, right, bottom = bbox_data_array[i]
            print(left, top, right, bottom)
            cv2.rectangle(img_pic, (int(left), int(top)), (int(right), int(bottom)), (255,0,0), 2, 1)
        
        
            data = keypoints_data_array[i]
            keypoints_dict = generate_keypoints_dict(data)
        
            #points = generate_dummy_keypoints()
            img_keypoints = generate_keypoint_img(img_pic, keypoints_dict)
        
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
    