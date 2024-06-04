# https://flet.dev/docs/controls/image
# https://flet.dev/docs/controls/gesturedetector

import flet as ft
import cv2
import base64
import numpy as np
from PIL import Image
import os
import time

from tools import draw_keypoints_on_picture
from tools import generate_label_text
from tools import read_annotation_data
from tools import generate_img_keypoints
from tools import PersonKeypoints

from ultralytics import YOLO



#modelpath = "models/yolov8n-pose.pt"
modelpath = "models/yolov8l-pose.pt"
model = YOLO(modelpath) 

keypoints_list = []
detected_persons = []
nearest_idx = 0
nearest_point_name = ""
img_pic = None
selected_person_idx = 0
selected_point_name = ""
img_idx = 0
img_dir = ""
image_filenames = []
filepath_img = ""
filepath_label = ""
num_img_files = 0


def main(page: ft.Page):
    
    # page(アプリ画面)の設定
    page.title = "HUMAN POSE ANNOTATION TOOL"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.colors.INDIGO_50
    page.padding = 50
    #page.window_height = 800
    #page.window_width = 800
    page.window_maximized = True
    page.update()
    time.sleep(1)
    print("Full Screen", page.window_height, page.window_width)
    
    IMG_SIZE= int(page.window_height*0.8)
    print(IMG_SIZE)

    # numpy画像データをbase64にエンコードする関数
    def get_base64_img(img):
        _, encoded = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(encoded).decode("ascii")
        return img_base64
    
    def on_open_img_dir(e: ft.FilePickerResultEvent):
        global images_dir, image_filenames, filepath_img, num_img_files
        images_dir = e.path
        image_filenames = sorted(os.listdir(images_dir))
        image_filenames = [x for x in image_filenames if ".jpg" in x]
        num_img_files = len(image_filenames)
        print(image_filenames)
        # 最初の画像を開く
        filepath_img = os.path.join(images_dir, image_filenames[0])
        open_image(filepath_img)
    
    def on_next_img_button_clicked(e):
        global img_idx, filepath_img
        print(image_filenames)
        if img_idx < num_img_files-1:
            img_idx +=1
            filepath_img = os.path.join(images_dir, image_filenames[img_idx])
            print(filepath_img)
            open_image(filepath_img)

    def on_previous_img_button_clicked(e):
        global img_idx, filepath_img
        print(image_filenames)
        if img_idx > 0:
            img_idx -= 1
            filepath_img = os.path.join(images_dir, image_filenames[img_idx])
            print(filepath_img)
            open_image(filepath_img)
        
    
    # ファイルが選択された時のコールバック
    def open_image(filepath_img):
        global keypoints_list, detected_persons, img_pic, filepath_label
        
        print(filepath_img)
        dir_images, filename = os.path.split(filepath_img)
        filename_body, _ = os.path.splitext(filename)
        dataset_dir, _ = os.path.split(dir_images)
        filepath_label = os.path.join(dataset_dir, "labels", filename_body+".txt")
                

        # OpenCVで画像を読み込み
        img_pic = np.array(Image.open(filepath_img))
        #img_pic = cv2.imread(filepath_img)
        original_img_h, _, _ = img_pic.shape
        resize_ratio = IMG_SIZE/original_img_h
        img_pic = cv2.resize(img_pic, dsize=None, fx=resize_ratio, fy=resize_ratio)
        img_h, img_w, _ = img_pic.shape
        
        # filepath_labelが存在するか確認し，なければYOLOで推論する
        if not os.path.exists(filepath_label):
            print(f"{filepath_label} does not exist.")
            print(f"Trying to detect keypoints...")
            # YOLO poseで推論
            results = model(img_pic)[0]
            annotation_text = generate_label_text(results)
            
            with open(filepath_label, "w") as file:
                file.write(annotation_text)
                print(f"{filepath_label} generated!")
        else:
            print(f"{filepath_label} already exists.")
        
        # テキストファイルからアノテーションデータを読み取り
        detected_persons = read_annotation_data(filepath_label, img_h, img_w)    

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
    
    def on_yolo_assist_button_clicked(e):
        global keypoints_list, detected_persons, img_pic, filepath_label
        print(f"Trying to detect keypoints...")
        # YOLO poseで推論
        results = model(img_pic)[0]
        annotation_text = generate_label_text(results)
        
        with open(filepath_label, "w") as file:
            file.write(annotation_text)
            print(f"{filepath_label} generated!")

        # テキストファイルからアノテーションデータを読み取り
        img_h, img_w, _ = img_pic.shape
        detected_persons = read_annotation_data(filepath_label, img_h, img_w)    

        # キーポイント画像を生成
        img_keypoints, keypoints_list = generate_img_keypoints(img_pic, detected_persons)
        print(keypoints_list)
        
        # 写真とキーポイントデータを重ね合わせ
        img_result = draw_keypoints_on_picture(img_pic, img_keypoints)
        
        # image_displayのプロパティを更新
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        image_display.src_base64 = get_base64_img(img_result)
        
        # pageをアップデート
        page.update()
        
        
        
    
    # アノテーションを保存
    def on_save_annotation_button_clicked(e):
        print("save annotation")
        annotation_text = ""
        for person in detected_persons:
            print(person.keypoints_dict)
            class_idx = 0
            box_lt_xn = person.keypoints_dict["box_lt"]["xn"]
            box_lt_yn = person.keypoints_dict["box_lt"]["yn"]
            box_rb_xn = person.keypoints_dict["box_rb"]["xn"]
            box_rb_yn = person.keypoints_dict["box_rb"]["yn"]
            box_center_xn = (box_lt_xn + box_rb_xn)/2
            box_center_yn = (box_lt_yn + box_rb_yn)/2
            box_width = abs(box_lt_xn - box_rb_xn)
            box_height = abs(box_lt_yn - box_rb_yn)
            annotation_text += f"{class_idx:d} {box_center_xn:.6f} {box_center_yn:.6f} {box_width:.6f} {box_height:.6f}"
            i = 0
            for name, data in person.keypoints_dict.items():
                annotation_text += f' {data["xn"]:.6f} {data["yn"]:.6f} {data["visibility"]:d}'
                i += 1
                if i==17: break
            annotation_text += "\n"
        print(annotation_text)
        
        with open(filepath_label, "w") as f:
            f.write(annotation_text)
        
        
            
    
    # # ファイルが選択された時のコールバック
    # def on_img_open(e: ft.FilePickerResultEvent):
    #     global keypoints_list, detected_persons, img_pic
        
    #     filepath_img = e.files[0].path
    #     dir_images, filename = os.path.split(filepath_img)
    #     filename_body, _ = os.path.splitext(filename)
    #     dataset_dir, _ = os.path.split(dir_images)
    #     filepath_label = os.path.join(dataset_dir, "labels", filename_body+".txt")
                

    #     # OpenCVで画像を読み込み
    #     img_pic = np.array(Image.open(filepath_img))
    #     #img_pic = cv2.imread(filepath_img)
    #     original_img_h, _, _ = img_pic.shape
    #     resize_ratio = IMG_SIZE/original_img_h
    #     img_pic = cv2.resize(img_pic, dsize=None, fx=resize_ratio, fy=resize_ratio)
    #     img_h, img_w, _ = img_pic.shape
        
    #     # filepath_labelが存在するか確認し，なければYOLOで推論する
    #     if not os.path.exists(filepath_label):
    #         print(f"{filepath_label} does not exist.")
    #         print(f"Trying to detect keypoints...")
    #         # YOLO poseで推論
    #         results = model(img_pic)[0]
    #         label_text = generate_label_text(results)
            
    #         with open(filepath_label, "w") as file:
    #             file.write(label_text)
    #             print(f"{filepath_label} generated!")
    #     else:
    #         print(f"{filepath_label} already exists.")
        
    #     # テキストファイルからアノテーションデータを読み取り
    #     detected_persons = read_annotation_data(filepath_label, img_h, img_w)    

    #     # キーポイント画像を生成
    #     img_keypoints, keypoints_list = generate_img_keypoints(img_pic, detected_persons)
    #     print(keypoints_list)
        
    #     # 写真とキーポイントデータを重ね合わせ
    #     img_result = draw_keypoints_on_picture(img_pic, img_keypoints)
        
    #     # image_displayのプロパティを更新
    #     img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    #     image_display.src_base64 = get_base64_img(img_result)
    #     image_display.height = img_h
    #     image_display.width = img_w
        
    #     # stackのプロパティを更新
    #     stack.height = img_h
    #     stack.width = img_w
        
    #     # pageをアップデート
    #     page.update()
    
    def mouse_on_hover(e: ft.HoverEvent):
        x_loc.value = int(e.local_x)
        y_loc.value = int(e.local_y)
        page.update()
        
    def mouse_double_click(e: ft.TapEvent):
        print("Double click")

    def mouse_click(e: ft.TapEvent):
        global keypoints_list, nearest_idx, detected_persons, nearest_point_name, selected_person_idx, selected_point_name
        print("Single click")
        x_loc.value = int(e.local_x)
        y_loc.value = int(e.local_y)        
        xy_clicked = (x_loc.value, y_loc.value)
        print("Point:", xy_clicked)
        
        nearest_point_distance_list = []
        nearest_point_name_list = []
        for i, person in enumerate(detected_persons):
            nearest_idx, nearest_point_dictance, nearest_point_name, nearest_point = person.find_nearest_keypoint(xy_clicked)
            print("Nearest Point:", nearest_idx, nearest_point_name, nearest_point)
            nearest_point_distance_list.append(nearest_point_dictance)
            nearest_point_name_list.append(nearest_point_name)
        
        selected_person_idx = np.array(nearest_point_distance_list).argmin()
        selected_point_name = nearest_point_name_list[selected_person_idx]
        print("selected person idx", selected_person_idx)
        print("selected point name", selected_point_name)
    
    def mouse_drag(e: ft.DragUpdateEvent):
        global keypoints_list, nearest_idx, nearest_point_name, img_pic, detected_persons
        x_loc.value = int(e.local_x)
        y_loc.value = int(e.local_y)        
        xy_drag = [x_loc.value, y_loc.value]
        print("Mouse Dragging:", nearest_point_name, xy_drag)
        
        person = detected_persons[selected_person_idx]
        person.update_point(selected_point_name, xy_drag)
        detected_persons[selected_person_idx]=person
        
        
        # キーポイント画像を生成
        img_keypoints, keypoints_list = generate_img_keypoints(img_pic, detected_persons)
        print(keypoints_list)
        
        # 写真とキーポイントデータを重ね合わせ
        img_result = draw_keypoints_on_picture(img_pic, img_keypoints)
        
        # image_displayのプロパティを更新
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        image_display.src_base64 = get_base64_img(img_result)
        
        # pageをアップデート
        page.update()        
    
    
    filepick_button = ft.ElevatedButton("Open Image", on_click=lambda _: file_picker.pick_files(allow_multiple=True))
    open_img_dir_button = ft.ElevatedButton("Open Image Directory", on_click=lambda _: file_picker.get_directory_path(initial_directory="dataset"))
    next_img_button = ft.ElevatedButton("Next", on_click=on_next_img_button_clicked)
    previous_img_button = ft.ElevatedButton("Previous", on_click=on_previous_img_button_clicked)
    save_annotation_button = ft.ElevatedButton("Save", on_click=on_save_annotation_button_clicked)
    yolo_assist_button = ft.ElevatedButton("YOLO Assist", on_click=on_yolo_assist_button_clicked)
    
    # 初期画像（ダミー）
    img_blank = 255*np.ones((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    img_h, img_w, _ = img_blank.shape
    image_display = ft.Image(src_base64=get_base64_img(img_blank),
                             width=img_w, height=img_h,
                             fit=ft.ImageFit.CONTAIN)
    
    gd = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        on_hover=mouse_on_hover,
        on_tap_down=mouse_click,
        on_horizontal_drag_update=mouse_drag,
        on_double_tap=mouse_double_click)
    
    x_loc_label = ft.Text("X", size=20)
    x_loc = ft.Text("0", size=20)
    y_loc_label = ft.Text("Y", size=20)
    y_loc = ft.Text("0", size=20)
    
    mouse_loc = ft.Column([ft.Row([x_loc_label, x_loc]),
                           ft.Row([y_loc_label, y_loc])])
    
    stack = ft.Stack([image_display, gd], width=IMG_SIZE, height=IMG_SIZE)
        
    page.add(ft.Row([open_img_dir_button, previous_img_button, next_img_button, save_annotation_button, yolo_assist_button]))
    page.add(ft.Row([stack, mouse_loc]))
    
    #file_picker = ft.FilePicker(on_result=on_img_open)
    file_picker = ft.FilePicker(on_result=on_open_img_dir)
    page.overlay.append(file_picker)
    page.update()
    



# デスクトップアプリとして開く
ft.app(target=main)

# webアプリとして開く場合は任意のポート番号を指定し
# ブラウザでlocalhost:7777を開く
# ft.app(target=main, port=7777)
    