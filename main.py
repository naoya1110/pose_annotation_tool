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

keypoints_name_list =  [
                    "nose",
                    "left_eye", "right_eye",
                    "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist",
                    "left_hip", "right_hip",
                    "left_knee","right_knee",
                    "left_ankle", "right_ankle"]



#modelpath = "models/yolov8n-pose.pt"
modelpath = "models/yolov8l-pose.pt"
model = YOLO(modelpath) 

keypoints_list = []
detected_persons = []
nearest_idx = 0
nearest_point_name = ""
img_pic = None
img_pic_corrected = None
selected_person_idx = 0
selected_point_name = ""
img_idx = 0
img_dir = ""
image_filenames = []
filepath_img = ""
filepath_label = ""
num_img_files = 0
person = None


def main(page: ft.Page):
    
    # page(アプリ画面)の設定
    page.title = "HUMAN POSE ANNOTATION TOOL"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.colors.INDIGO_50
    page.padding = 25
    #page.window_height = 800
    #page.window_width = 800
    page.window_maximized = True
    page.update()
    time.sleep(1)
    print("Full Screen", page.window_height, page.window_width)
    
    IMG_SIZE= int(page.window_height*0.80)
    print(IMG_SIZE)


    def on_keypoints_table_changed(e):
        global detected_persons
        
        print("keypoints table changed manually")
        img_h, img_w, _ = img_pic.shape
        person = detected_persons[int(person_idx_dropdown.value)]
        
        for i in range(17):
            name = keypoints_table.controls[i].controls[0].value
            x = int(keypoints_table.controls[i].controls[1].value)
            y = int(keypoints_table.controls[i].controls[2].value)
            visibility = int(keypoints_table.controls[i].controls[3].value)
            
            if visibility == 0:
                x = 0
                y = 0
            
            person.keypoints_dict[name]["x"] = x
            person.keypoints_dict[name]["xn"] = x/img_w
            person.keypoints_dict[name]["y"] = y
            person.keypoints_dict[name]["yn"] = y/img_h
            person.keypoints_dict[name]["visibility"] = visibility
        
        detected_persons[int(person_idx_dropdown.value)]=person
        update_keypoints_table(detected_persons[int(person_idx_dropdown.value)])
        update_image_display(img_pic_corrected, detected_persons)
        
        
        # pageをアップデート
        page.update()

    keypoints_table = ft.Column([ft.Row([ft.Text(name, width=100),
                                        ft.TextField(label="X",
                                                    width=50, height=30,
                                                    border="underline",
                                                    text_size=16, text_align=ft.TextAlign.RIGHT,
                                                    on_submit=on_keypoints_table_changed),
                                        ft.TextField(label="Y",
                                                    width=50, height=30,
                                                    border="underline",
                                                    text_size=16, text_align=ft.TextAlign.RIGHT,
                                                    on_submit=on_keypoints_table_changed),
                                        # ft.TextField(label="Vis",
                                        #             width=50, height=30,
                                        #             border="underline",
                                        #             text_size=16, text_align=ft.TextAlign.RIGHT,
                                        #             on_submit=on_keypoints_table_changed),
                                        ft.Slider(min=0, max=2,
                                                divisions=2, label="{value}",
                                                height=20, width=100,
                                                on_change=on_keypoints_table_changed),
                                        # ft.Dropdown(label="Vis",
                                        #             width=50, height=40,
                                        #             options = [ft.dropdown.Option(i) for i in range(3)],
                                        #             text_size=8,
                                        #             #border="underline",
                                        #             #text_size=16, text_align=ft.TextAlign.RIGHT,
                                        #             on_change=on_keypoints_table_changed),
                                        ]) for name in keypoints_name_list])


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
        progress_bar.value = (img_idx+1)/num_img_files
        progress_text.value = f"{img_idx+1}/{num_img_files}"
        open_image(filepath_img)
    
    def on_next_img_button_clicked(e):
        global img_idx, filepath_img
        if auto_save_checkbox.value:
            save_annotation()
            print("saved annotation")
            
        print(image_filenames)
        if img_idx < num_img_files-1:
            img_idx +=1
            filepath_img = os.path.join(images_dir, image_filenames[img_idx])
            print(filepath_img)
            progress_bar.value = (img_idx+1)/num_img_files
            progress_text.value = f"{img_idx+1}/{num_img_files}"
            open_image(filepath_img)

    def on_previous_img_button_clicked(e):
        global img_idx, filepath_img
        if auto_save_checkbox.value:
            save_annotation()
            print("saved annotation")
            
        print(image_filenames)
        if img_idx > 0:
            img_idx -= 1
            filepath_img = os.path.join(images_dir, image_filenames[img_idx])
            print(filepath_img)
            progress_bar.value = (img_idx+1)/num_img_files
            progress_text.value = f"{img_idx+1}/{num_img_files}"
            open_image(filepath_img)
        
    
    # ファイルが選択された時のコールバック
    def open_image(filepath_img):
        global keypoints_list, detected_persons, img_pic, img_pic_corrected, filepath_label, selected_person_idx, selected_point_name
        
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

        image_display.height = img_h
        image_display.width = img_w
        stack.height = img_h
        stack.width = img_w

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
        if len(detected_persons) > 0:
            update_person_idx_dropdown(detected_persons)
            selected_person_idx = 0
            person = detected_persons[0]
        else:
            selected_person_idx = None
            selected_point_name = ""
            person = None
            
        person_idx_dropdown.value = selected_person_idx
        selected_person_idx_text.value = selected_person_idx
        selected_keypoint_name_text.value = selected_point_name
        update_keypoints_table(person)
            
            
            
        img_pic_corrected = gamma_correction(img_pic)
        update_image_display(img_pic_corrected, detected_persons)
        page.update()

    def update_image_display(img_pic, detected_persons):
        # キーポイント画像を生成
        img_keypoints, keypoints_list = generate_img_keypoints(img_pic, detected_persons)
        
        # 写真とキーポイントデータを重ね合わせ
        img_result = draw_keypoints_on_picture(img_pic, img_keypoints)
        
        # image_displayのプロパティを更新
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        image_display.src_base64 = get_base64_img(img_result)

    
    def update_person_idx_dropdown(detected_persons):
        person_idx_dropdown.options = []
        if len(detected_persons) > 0:
            for i in range(len(detected_persons)):
                person_idx_dropdown.options.append(ft.dropdown.Option(i))
            person_idx_dropdown.value = 0
    
    def on_person_idx_dropdown_changed(e):
        person = detected_persons[int(person_idx_dropdown.value)]
        update_keypoints_table(person)
        
        
    
    def on_yolo_assist_button_clicked(e):
        global keypoints_list, detected_persons, img_pic, img_pic_corrected, filepath_label
        print(f"Trying to detect keypoints...")
        
        yolo_assist_button.bgcolor = ft.colors.RED_100
        yolo_assist_button.update()
        time.sleep(0.5)

        # YOLO poseで推論
        results = model(img_pic_corrected)[0]
        annotation_text = generate_label_text(results)
        
        with open(filepath_label, "w") as file:
            file.write(annotation_text)
            print(f"{filepath_label} generated!")

        # テキストファイルからアノテーションデータを読み取り
        img_h, img_w, _ = img_pic.shape
        detected_persons = read_annotation_data(filepath_label, img_h, img_w)
        if len(detected_persons) > 0:
            update_keypoints_table(detected_persons[int(person_idx_dropdown.value)])
            update_image_display(img_pic_corrected, detected_persons)
        yolo_assist_button.bgcolor = ft.colors.BACKGROUND
        
        # pageをアップデート
        page.update()
        
        
        
    
    # アノテーションを保存
    def on_save_annotation_button_clicked(e):
        save_annotation()
    
    # アノテーションを保存
    def save_annotation():
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
        
        
        if np.array(nearest_point_distance_list).min() < 20:
            selected_person_idx = np.array(nearest_point_distance_list).argmin()
            selected_point_name = nearest_point_name_list[selected_person_idx]
            person = detected_persons[selected_person_idx]
        else:
            selected_person_idx = None
            selected_point_name = ""
            person = None
        person_idx_dropdown.value = selected_person_idx
        selected_person_idx_text.value = selected_person_idx
        selected_keypoint_name_text.value = selected_point_name
        update_keypoints_table(person)
        
        for i, name in enumerate(keypoints_name_list):
            if name == selected_point_name:
                bgcolor = ft.colors.ORANGE_100
            else:
                bgcolor = ft.colors.TRANSPARENT
            keypoints_table.controls[i].controls[0].bgcolor = bgcolor
        page.update()
        print("selected person idx", selected_person_idx)
        print("selected point name", selected_point_name)


    
    def mouse_drag(e: ft.DragUpdateEvent):
        global keypoints_list, nearest_idx, nearest_point_name, img_pic, detected_persons
        
        img_h, img_w, _ = img_pic.shape
        mouse_x = int(e.local_x)
        mouse_y = int(e.local_y)
        
        mouse_x = min([mouse_x, img_w])
        mouse_x = max([0, mouse_x])
        mouse_y = min([mouse_y, img_h])
        mouse_y = max([0, mouse_y])
        
        x_loc.value = mouse_x
        y_loc.value = mouse_y
              
        xy_drag = [mouse_x, mouse_y]
        #print("Mouse Dragging:", nearest_point_name, xy_drag)
        
        person = detected_persons[selected_person_idx]
        person.update_point(selected_point_name, xy_drag)
        detected_persons[selected_person_idx]=person
        update_keypoints_table(person)
        update_image_display(img_pic_corrected, detected_persons)
        # pageをアップデート
        page.update()
    
    def on_blightness_slider_changed(e):
        global img_pic, img_pic_corrected
        img_pic_corrected = gamma_correction(img_pic)
        update_image_display(img_pic_corrected, detected_persons)
        page.update()
        
    def gamma_correction(img_pic):
        gamma = float(blightness_slider.value)
        x = img_pic.copy()
        x = (x/255).astype("float32")
        x = x**(1/float(gamma))
        x = 255*x
        img_pic_corrected = x.astype("uint8")
        return img_pic_corrected
        

    def update_keypoints_table(person):
        for i, name in enumerate(keypoints_name_list):
            if person != None:
                x = person.keypoints_dict[name]["x"]
                y = person.keypoints_dict[name]["y"]
                visibility = person.keypoints_dict[name]["visibility"]
            else:
                x = 0
                y = 0
                visibility = 0
            keypoints_table.controls[i].controls[1].value = x
            keypoints_table.controls[i].controls[2].value = y
            keypoints_table.controls[i].controls[3].value = visibility
        keypoints_table.update()
        
        
    
    
    # コントロール
    open_img_dir_button = ft.ElevatedButton("Open Image Directory", on_click=lambda _: file_picker.get_directory_path())
    next_img_button = ft.ElevatedButton("Next", on_click=on_next_img_button_clicked)
    previous_img_button = ft.ElevatedButton("Previous", on_click=on_previous_img_button_clicked)
    save_annotation_button = ft.ElevatedButton("Save", on_click=on_save_annotation_button_clicked)
    yolo_assist_button = ft.ElevatedButton("YOLO Assist", on_click=on_yolo_assist_button_clicked)
    auto_save_checkbox = ft.Checkbox(label="Auto Save", value=True)
    progress_bar = ft.ProgressBar(width=400, height=10)
    progress_text = ft.Text()
    person_idx_dropdown = ft.Dropdown(width=100, options=[], on_change=on_person_idx_dropdown_changed)
    blightness_slider = ft.Slider(value=1, min=0.2, max=2,
                                divisions=9, label="{value}",
                                round=1,
                                on_change_end=on_blightness_slider_changed)

    


    
    # 初期画像（ダミー）
    img_blank = 255*np.ones((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    img_h, img_w, _ = img_blank.shape
    image_display = ft.Image(src_base64=get_base64_img(img_blank),
                             width=img_w, height=img_h,
                             fit=ft.ImageFit.CONTAIN)
    
    # Gesture Detector
    gd = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        on_hover=mouse_on_hover,
        on_tap_down=mouse_click,
        on_horizontal_drag_update=mouse_drag,
        on_double_tap=mouse_double_click)
    
    x_loc = ft.TextField(label="X", value=0, text_size=16, width=100, height=40)
    y_loc = ft.TextField(label="Y", value=0, text_size=16, width=100, height=40)
    
    selected_person_idx_text = ft.TextField(label="Person Idx", value=0, text_size=16, width=100, height=40)
    selected_keypoint_name_text = ft.TextField(label="Keypoint Name", value=" ", text_size=16, width=150, height=40)
    

    
    stack = ft.Stack([image_display, gd], width=IMG_SIZE, height=IMG_SIZE)
        
    page.add(ft.Row([open_img_dir_button, previous_img_button, next_img_button, save_annotation_button, yolo_assist_button, auto_save_checkbox]))
    page.add(ft.Row([stack, ft.Column([person_idx_dropdown, keypoints_table])]))
    page.add(ft.Row([progress_bar, progress_text,
                    x_loc, y_loc,
                    selected_person_idx_text, selected_keypoint_name_text,
                    blightness_slider]))
    
    #file_picker = ft.FilePicker(on_result=on_img_open)
    file_picker = ft.FilePicker(on_result=on_open_img_dir)
    page.overlay.append(file_picker)
    page.update()
    



# デスクトップアプリとして開く
ft.app(target=main)

# webアプリとして開く場合は任意のポート番号を指定し
# ブラウザでlocalhost:7777を開く
# ft.app(target=main, port=7777)
    