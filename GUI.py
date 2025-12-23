# Input: 
#   1.HEX(fetch by the color plate or RGB (modified by movable switches) and show the color real-time)
#   2.Image(recommendation the lip's mean color, Not Done Yet)
#   3.launch_video_capture(Same as Image, Not Done Yet)
# Need: 
#   1.Btn*1(OpenImage...); SmallBtn*2(Left-Rotation?, Right-Rotation?)  ***
#   2.Btn*1(OpenCapture(On/Off)?)                                       *
#   3.Btn*3(FetchColor(and show)?, ModifyColor..., Return?)             ***
#   4.MoveableBtn*3(ModifyColor - R,G,B)                                ***
#   5.Widge*1(Image/Capture)                                            *
#   6.PhotoDisplay*1(color)                                             TODO

# Recommendation:
#   Functions.
# Need:
#   1.Btn*2(SetDefault..., Recommendation?)                             **
#   1.Btn*1(Re-Inference?)                                              *
#   2.TextLabel*1(Inference Logs)                                       *

# Output: 
#   1.List(similar lipsticks)
#   2.Virtual Try-on (Video Capture covered by new-lips (click the item in List))
# Need:
#   1.ListTable*1(Recommemded List)                                     *
#   2.Btn*1(DisplayInfo?)                                               *
#   3.PhotoDisplay(Lipstick)                                            *
#   4.TextLabel(DisplayInfo)                                            *
#   5.Btn*1(Application on real-time capture(On/Off)?)                  *

# Others:
#   1.Clear canvas
#   2.Quit
#   3.Clear history
# Need:
#   1.Btn*1(ClearCanvas?)                                               *
#   2.Btn*1(Quit?)                                                      *
#   3.Btn*1(ClearHistory?)                                              TODO


import tkinter as tk
from tkinter import font, filedialog, messagebox, scrolledtext
# from tkinter.simpledialog import askinteger
from PIL import Image, ImageTk
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1" # Settings: No Welcome Message from pygame.
import pygame

import jittor as jt
import jittor.transform as transform
from utils.models import get_model
from utils.rgb2hex import rgb_to_hex
from utils.recommend import recommendation

import cv2
import dlib
import numpy as np
import pandas as pd
from collections import OrderedDict
import videocapture

class App:
    def __init__(self, master):
        self.master = master
        master.title("LipstickRecommendation")
        master.geometry("800x800")

        self.open_image_button = tk.Button(master, text="打开图片", command=self.open_image)
        self.open_image_button.place(relx=0.05, rely=0.05, relwidth=0.2, relheight=0.1)
        
        self.custom_font = font.Font(size=20, weight="bold") ## 修改字体与大小（需调用）
        
        self.right_rotate_button = tk.Button(master, text="↷", command=self.left_rotate, font=self.custom_font)
        self.right_rotate_button.place(relx=0.25, rely=0.05, relwidth=0.05, relheight=0.05)
        
        self.left_rotate_button = tk.Button(master, text="↶", command=self.right_rotate, font=self.custom_font)
        self.left_rotate_button.place(relx=0.25, rely=0.1, relwidth=0.05, relheight=0.05)

        self.display_fetched_color_button = tk.Button(master, text="提取原图色彩", command=self.display_fetched_color)
        self.display_fetched_color_button.place(relx=0.05, rely=0.2, relwidth=0.2, relheight=0.1)
        
        self.fetched_color_image = tk.Label(self.master, text="展\n示\n区", borderwidth = 3, relief="sunken")
        self.fetched_color_image.place(relx=0.25, rely=0.2, relwidth=0.05, relheight=0.1)

        self.modify_color_button = tk.Button(master, text="调整色彩", command=self.modify_color)
        self.modify_color_button.place(relx=0.05, rely=0.35, relwidth=0.2, relheight=0.1)
        
        self.modified_color_image = tk.Label(self.master, text="展\n示\n区", borderwidth = 3, relief="sunken")
        self.modified_color_image.place(relx=0.25, rely=0.35, relwidth=0.05, relheight=0.1)
        
        self.setting_recommend_numbers_button = tk.Button(master, text="设置推荐数量\n默认推荐 10 支", command=self.setting_recommend_numbers)
        self.setting_recommend_numbers_button.place(relx=0.05, rely=0.5, relwidth=0.25, relheight=0.1)
        
        self.image_label = tk.Label(master, text="图片打开位置", borderwidth = 3, relief="sunken")
        self.image_label.place(relx=0.05, rely=0.675, relwidth=0.25, relheight=0.25)
        
        # self.video_label = tk.Label(master, text="摄像头打开位置", borderwidth = 3, relief="sunken")
        # self.video_label.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.6)
        self.recommemdation_list = scrolledtext.ScrolledText(master, wrap=tk.WORD)
        self.recommemdation_list.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.6)
        
        self.recommendation_button = tk.Button(master, text="识别图片\n生成推荐色号", command=self.recommendation)
        self.recommendation_button.place(relx=0.375, rely=0.675, relwidth=0.25, relheight=0.1)

        self.virtual_try_on_button = tk.Button(master, text="虚拟试装间", command=self.virtual_try_on)
        self.virtual_try_on_button.place(relx=0.375, rely=0.825, relwidth=0.25, relheight=0.1)
        
        # self.inference_logs = tk.Label(master, text="训练日志")
        # self.inference_logs.place(relx=0.05, rely=0.65, relwidth=0.3, relheight=0.3)
        
        # self.recommemdation_list = scrolledtext.ScrolledText(master, wrap=tk.WORD)
        # self.virtual_try_on_display_color_button = tk.Button(master, text="展示商品信息", command=self.virtual_try_on_display_color)
        
        self.clear_text_button = tk.Button(master, text="清空推荐列表", command=self.clear_text)
        self.clear_text_button.place(relx=0.675, rely=0.675, relwidth=0.25, relheight=0.1)
        
        self.quit_app_button = tk.Button(master, text="关闭并退出", command=self.quit_app)
        self.quit_app_button.place(relx=0.675, rely=0.825, relwidth=0.25, relheight=0.1)

        # Rotate the image
        self.rotate_angle = 0
        
        # Video Capture
        self.cap = None
        self.video_capture_landmarks = 'models/pretrained/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.criticPoints = dlib.shape_predictor(self.video_capture_landmarks)
        self.landmarks = OrderedDict([
            ('mouth',(48,68)),
            ('right_eyebrow',(17,22)),
            ('left_eye_brow',(22,27)),
            ('right_eye',(36,42)),
            ('left_eye',(42,48)),
            ('nose',(27,36)),
            ('jaw',(0,17))
        ])
        
        # Set Recommendation Numbers
        self.recommend_numbers = 10
        self.real_recommend_count = 0
        
        # Recommendation
        self.cluster_file = 'datasets/lipstick_clusters.csv'
        self.recommendation_model_path = 'models/saves/best_train_acc_model.pkl'
        num_classes = 5
        self.recommendation_model = get_model(num_classes=num_classes)
        self.recommendation_model.load_state_dict(jt.load(self.recommendation_model_path))
        # self.recommendation_model.eval()
        
        self.lipstick_recommend_list = []
        self.historically_saved_dir = 'history_save'
        self.set_recommendation_bool = False # false 未识别与推荐， true 已推荐

        # Display and Modify the color
        self.set_fetched_bool = False # false 未提取， true 已提取
        self.set_modified_bool = False # false 未修改， true 已修改
        self.fetched_R, self.fetched_G, self.fetched_B = 0, 0, 0
        self.modified_R, self.modified_G, self.modified_B = self.fetched_R, self.fetched_G, self.fetched_B
        self.last_modified_R, self.last_modified_G, self.last_modified_B = self.modified_R, self.modified_G, self.modified_B
    
    def open_image(self):
        App.__init__(self, self.master)
        self.picture_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if self.picture_path == '':
            return
        
        self.image_label = tk.Label(self.master, borderwidth = 3)
        self.image_label.place(relx=0.05, rely=0.675, relwidth=0.25, relheight=0.25)
        if self.picture_path:
            try:
                image = Image.open(self.picture_path)
                width, height = image.size
                if width > 200 or height > 200:
                    max_size = max(width, height)
                    new_width = int(width * 200 / max_size)
                    new_height = int(height * 200 / max_size)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.original_image = image
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except Exception as e:
                messagebox.showerror("Error", f"无法打开图片：{e}")
    
    def left_rotate(self):
        if hasattr(self, 'original_image'):
            self.rotate_angle -= 90
            rotated_image = self.original_image.rotate(self.rotate_angle)
            # self.original_image = rotated_image
            
            photo = ImageTk.PhotoImage(rotated_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
    def right_rotate(self):
        if hasattr(self, 'original_image'):
            self.rotate_angle += 90
            rotated_image = self.original_image.rotate(self.rotate_angle)
            # self.original_image = rotated_image
            
            photo = ImageTk.PhotoImage(rotated_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
    
    def display_fetched_color(self):
        self.fetched_R = 255
        self.fetched_G = 255
        self.fetched_B = 2
        self.fetched_color_image = tk.Label(self.master, borderwidth = 3, relief="sunken")
        self.fetched_color_image.place(relx=0.25, rely=0.2, relwidth=0.05, relheight=0.1)
        
        pure_color_image = Image.new('RGB', (100, 100), (self.fetched_R, self.fetched_G, self.fetched_B))
        
        photo = ImageTk.PhotoImage(pure_color_image)
        self.fetched_color_image.configure(image=photo)
        self.fetched_color_image.image = photo
        
        self.set_fetched_bool = True
        return
    
    def modify_color(self):
        if not self.set_fetched_bool:
            messagebox.showinfo("Info", "请先提取色彩！")
            return
        
        if not self.set_modified_bool:
            self.modified_R, self.modified_G, self.modified_B = self.fetched_R, self.fetched_G, self.fetched_B
        
        modify_color_window = tk.Toplevel(root)
        modify_color_window.title("调色盘")
        modify_color_window.geometry("400x400+400+400")  # 宽x高+水平偏移量+垂直偏移量
            
        modify_color_window.focus_set()
        
        self.fetched_color_image = tk.Label(modify_color_window)
        self.fetched_color_image.place(relx=0.1, rely=0.1)
        pure_color_image = Image.new('RGB', (80, 80), (self.fetched_R, self.fetched_G, self.fetched_B))
        photo = ImageTk.PhotoImage(pure_color_image)
        self.fetched_color_image.configure(image=photo)
        self.fetched_color_image.image = photo
        
        self.last_modified_color_image = tk.Label(modify_color_window)
        self.last_modified_color_image.place(relx=0.4, rely=0.1)
        
        self.modified_color_image = tk.Label(modify_color_window)
        self.modified_color_image.place(relx=0.7, rely=0.1)
            
        self.fetched_color_image_label = tk.Label(modify_color_window, text="原始")
        self.fetched_color_image_label.place(relx=0.175, rely=0.31)
        self.fetched_color_image_label_RGB = tk.Label(modify_color_window, text=f"({self.fetched_R}, {self.fetched_G}, {self.fetched_B})")
        self.fetched_color_image_label_RGB.place(relx=0.1, rely=0.36)
            
        self.modify_color_image_label = tk.Label(modify_color_window, text="当前")
        self.modify_color_image_label.place(relx=0.775, rely=0.31)
        self.modify_color_image_label_RGB = tk.Label(modify_color_window, text=f"({self.modified_R}, {self.modified_G}, {self.modified_B})")
        self.modify_color_image_label_RGB.place(relx=0.7, rely=0.36)
            
        self.modify_color_R_scale = tk.Scale(modify_color_window, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
        self.modify_color_R_scale.place(relx=0.05, rely=0.41, relwidth=0.9, relheight=0.1)
        self.modify_color_R_scale.set(self.modified_R)
            
        self.modify_color_G_scale = tk.Scale(modify_color_window, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
        self.modify_color_G_scale.place(relx=0.05, rely=0.51, relwidth=0.9, relheight=0.1)
        self.modify_color_G_scale.set(self.modified_G)
            
        self.modify_color_B_scale = tk.Scale(modify_color_window, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
        self.modify_color_B_scale.place(relx=0.05, rely=0.61, relwidth=0.9, relheight=0.1)
        self.modify_color_B_scale.set(self.modified_B)
            
        if self.set_modified_bool:
            self.last_modified_R, self.last_modified_G, self.last_modified_B = self.modified_R, self.modified_G, self.modified_B
            self.last_modified_color_image = tk.Label(modify_color_window)
            self.last_modified_color_image.place(relx=0.4, rely=0.1)
            pure_last_modified_color_image = Image.new('RGB', (80, 80), (self.last_modified_R, self.last_modified_G, self.last_modified_B))
            last_modified_photo = ImageTk.PhotoImage(pure_last_modified_color_image)
            self.last_modified_color_image.configure(image=last_modified_photo)
            self.last_modified_color_image.image = last_modified_photo
            
            self.last_modified_color_image_label = tk.Label(modify_color_window, text="上次")
            self.last_modified_color_image_label.place(relx=0.475, rely=0.31)
            self.last_modified_color_image_label_RGB = tk.Label(modify_color_window, text=f"({self.last_modified_R}, {self.last_modified_G}, {self.last_modified_B})")
            self.last_modified_color_image_label_RGB.place(relx=0.4, rely=0.36)
        
        def recover_color():
            self.modified_R, self.modified_G, self.modified_B = self.fetched_R, self.fetched_G, self.fetched_B
            self.modify_color_R_scale.set(self.modified_R)
            self.modify_color_G_scale.set(self.modified_G)
            self.modify_color_B_scale.set(self.modified_B)
        
        def comfirm_color():
            self.modified_R = self.modify_color_R_scale.get()
            self.modified_G = self.modify_color_G_scale.get()
            self.modified_B = self.modify_color_B_scale.get()
                
            self.set_modified_bool = True
            
            self.modified_color_image = tk.Label(self.master, borderwidth = 3, relief="sunken")
            self.modified_color_image.place(relx=0.25, rely=0.35, relwidth=0.05, relheight=0.1)
            pure_modified_color_image = Image.new('RGB', (100, 100), (self.modified_R, self.modified_G, self.modified_B))
            modified_photo = ImageTk.PhotoImage(pure_modified_color_image)
            self.modified_color_image.configure(image=modified_photo)
            self.modified_color_image.image = modified_photo
                
            messagebox.showinfo("Info", "修改成功！")
            modify_color_window.destroy()
        
        self.recover_color_button = tk.Button(modify_color_window, text="复原色彩", command=recover_color)
        self.recover_color_button.place(relx=0.05, rely=0.8, relwidth=0.35, relheight=0.1)
        
        self.comfirm_color_button = tk.Button(modify_color_window, text="确认色彩", command=comfirm_color)
        self.comfirm_color_button.place(relx=0.6, rely=0.8, relwidth=0.35, relheight=0.1)
        
        def update_modify_image():
            self.modified_R = self.modify_color_R_scale.get()
            self.modified_G = self.modify_color_G_scale.get()
            self.modified_B = self.modify_color_B_scale.get()
            
            self.modified_color_image = tk.Label(modify_color_window)
            self.modified_color_image.place(relx=0.7, rely=0.1)
            pure_modified_color_image = Image.new('RGB', (80, 80), (self.modified_R, self.modified_G, self.modified_B))
            modified_photo = ImageTk.PhotoImage(pure_modified_color_image)
            self.modified_color_image.configure(image=modified_photo)
            self.modified_color_image.image = modified_photo
                
            self.modify_color_image_label_RGB = tk.Label(modify_color_window, text=f"({self.modified_R}, {self.modified_G}, {self.modified_B})")
            self.modify_color_image_label_RGB.place(relx=0.7, rely=0.36)
            
            self.modified_color_image.after(15, update_modify_image)
            
        update_modify_image()
        modify_color_window.mainloop            
        
    def setting_recommend_numbers(self):# Create a custom TopLevel window
        set_recommend_number_window = tk.Toplevel(root)
        set_recommend_number_window.title("希望推荐的口红数目")
        set_recommend_number_window.geometry("300x125+400+400")  # 宽x高+水平偏移量+垂直偏移量
        label = tk.Label(set_recommend_number_window, text="请输入数值:（一个整数）")
        label.pack(pady=5)
        entry_var = tk.StringVar()
        entry_var.set("10")
        entry = tk.Entry(set_recommend_number_window, textvariable=entry_var)
        entry.pack(pady=10)
        
        def comfirm_recommend_numbers():
            try:
                result = int(entry_var.get())
                # print("你期望推荐 " + str(result) + " 首歌")
                self.recommend_numbers = result
                messagebox.showinfo("Info", f"已修改：你期望推荐 " + str(result) + " 支口红")
                self.setting_recommend_numbers_button = tk.Button(self.master, text="设置推荐数量\n当前推荐 " + str(result) + " 支", command=self.setting_recommend_numbers)
                self.setting_recommend_numbers_button.place(relx=0.05, rely=0.5, relwidth=0.25, relheight=0.1)
            except ValueError:
                result = self.recommend_numbers
                messagebox.showwarning("Warning", f"无效修改。推荐数目保持 " + str(result) + " 支口红")
            set_recommend_number_window.destroy()
        
        button = tk.Button(set_recommend_number_window, text="确认修改", command=comfirm_recommend_numbers)
        button.pack(pady=5)
        entry.focus_set()  # 将焦点设置在输入框上
        set_recommend_number_window.mainloop()
    
    def recommendation(self):
        if not self.set_fetched_bool:
            messagebox.showinfo("Info", "请先提取色彩！")
            return
        if self.set_modified_bool:
            r, g, b = self.modified_R, self.modified_G, self.modified_B
        else:
            r, g, b = self.fetched_R, self.fetched_G, self.fetched_B
            
        img = Image.new('RGB', (100, 100), (r, g, b))
        hex_string = rgb_to_hex(r, g, b)
        img_filename = f'{hex_string}.jpg'
        if not os.path.exists(self.historically_saved_dir):
            os.makedirs(self.historically_saved_dir, exist_ok=True)
        img.save(os.path.join(self.historically_saved_dir, img_filename))
        
        test_transform = transform.Compose([
            # transform.Resize((224, 224)),
            # transform.ImageNormalize(mean=[0.5], std=[0.5]),
            transform.ToTensor()
        ])
        
        img = test_transform(img)
        img = jt.array(img)[None, ...]  # shape: [1, 3, 224, 224]
        self.recommendation_model.eval()
        pred = self.recommendation_model(img)
        pred_label = int(jt.argmax(pred, dim=1)[0].item())
        # print(f"{r},{g},{b}")
        # print(f"{pred_label}")
        
        self.recommendation_button = tk.Button(self.master, text="重新识别推荐", command=self.recommendation)
        self.recommendation_button.place(relx=0.375, rely=0.675, relwidth=0.25, relheight=0.1)
        
        self.lipstick_recommend_list = recommendation(self.cluster_file, pred_label, self.recommend_numbers)
        # print(self.lipstick_recommend_list)
        
        # for index in range(self.recommend_numbers):
        #     if not pd.isna(self.lipstick_recommend_list[index]['names']):
        #         print(f"{self.lipstick_recommend_list[index]['brands']} - {self.lipstick_recommend_list[index]['series']} - {self.lipstick_recommend_list[index]['names']}, {self.lipstick_recommend_list[index]['id']}")
        #     else:
        #         print(f"{self.lipstick_recommend_list[index]['brands']} - {self.lipstick_recommend_list[index]['series']} - (Unnamed), {self.lipstick_recommend_list[index]['id']}")
        
        tmp_recommendation_list_file = f"{self.historically_saved_dir}/{hex_string}.csv"
        
        df = pd.DataFrame(self.lipstick_recommend_list)
        df.to_csv(tmp_recommendation_list_file, index=False, encoding="utf-8-sig")
        
        self.recommemdation_list.delete(1.0, tk.END)
        self.recommemdation_list.insert(tk.END, "已读取色号：\n")
        self.recommemdation_list.insert(tk.END, "推荐相关色号口红： \n\n")
        self.real_recommend_count = 0
        
        for index in range(self.recommend_numbers):
            # if not self.lipstick_recommend_list[index].any():
            #     print(f"Sorry the repository don't have so much. Here is all we save, total {real_recommend_count} items.")
            #     break
            try:
                if not pd.isna(self.lipstick_recommend_list[index]['names']):
                    result_str = f"{self.lipstick_recommend_list[index]['brands']} - {self.lipstick_recommend_list[index]['series']} - {self.lipstick_recommend_list[index]['names']}, {self.lipstick_recommend_list[index]['id']}"
                else:
                    result_str = f"{self.lipstick_recommend_list[index]['brands']} - {self.lipstick_recommend_list[index]['series']} - (Unnamed), {self.lipstick_recommend_list[index]['id']}"
                self.real_recommend_count = self.real_recommend_count + 1
                self.recommemdation_list.insert(tk.END, result_str + "\n")
            except:
                print(f"Sorry the repository don't have so much. Here is all we save, total {self.real_recommend_count} items.")
                break
        self.recommemdation_list.insert(tk.END, "-------- 共计 " + str(self.real_recommend_count) + " 支 --------")
        self.set_recommendation_bool = True

    def clear_text(self):
        self.recommemdation_list.delete(1.0, tk.END)
    
    def virtual_try_on(self):
        if not self.set_recommendation_bool:
            messagebox.showinfo("Info", "请先识别与推荐！")
            return
        
        virtual_try_on_window = tk.Toplevel(root)
        virtual_try_on_window.title("试衣间")
        virtual_try_on_window.geometry("600x600+100+100")  # 宽x高+水平偏移量+垂直偏移量
            
        virtual_try_on_window.focus_set()
        
        self.current_recommend_index = 0
        
        def virtual_try_on_display_color():
            if not pd.isna(self.lipstick_recommend_list[self.current_recommend_index]['names']):
                result_str = f"{self.lipstick_recommend_list[self.current_recommend_index]['brands']} - {self.lipstick_recommend_list[self.current_recommend_index]['series']} - {self.lipstick_recommend_list[self.current_recommend_index]['names']}, {self.lipstick_recommend_list[self.current_recommend_index]['id']}"
            else:
                result_str = f"{self.lipstick_recommend_list[self.current_recommend_index]['brands']} - {self.lipstick_recommend_list[self.current_recommend_index]['series']} - (Unnamed), {self.lipstick_recommend_list[self.current_recommend_index]['id']}"
            
            self.current_recommend_item_label = tk.Label(virtual_try_on_window, text=result_str, borderwidth = 3, relief="sunken")
            self.current_recommend_item_label.place(relx=0.05, rely=0.075, relwidth=0.9, relheight=0.05)
            
            virtual_try_on_display_color_R = self.lipstick_recommend_list[self.current_recommend_index]['R']
            virtual_try_on_display_color_G = self.lipstick_recommend_list[self.current_recommend_index]['G']
            virtual_try_on_display_color_B = self.lipstick_recommend_list[self.current_recommend_index]['B']
            
            self.virtual_try_on_display_color_image_label = tk.Label(virtual_try_on_window, borderwidth = 3, relief="sunken")
            self.virtual_try_on_display_color_image_label.place(relx=0.3, rely=0.175, relwidth=0.05, relheight=0.1)
            pure_color_image = Image.new('RGB', (80, 80), (virtual_try_on_display_color_R, virtual_try_on_display_color_G, virtual_try_on_display_color_B))
            photo = ImageTk.PhotoImage(pure_color_image)
            self.virtual_try_on_display_color_image_label.configure(image=photo)
            self.virtual_try_on_display_color_image_label.image = photo
        
        def last_recommend_item():
            self.current_recommend_index = (self.current_recommend_index - 1 + self.real_recommend_count) % self.real_recommend_count
            virtual_try_on_display_color()
        
        def next_recommend_item():
            self.current_recommend_index = (self.current_recommend_index + 1 + self.real_recommend_count) % self.real_recommend_count
            virtual_try_on_display_color()    
            
        def video_update(self):
            return
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            # 人脸识别
            detected = self.detector(frame)
            mouth_range = self.landmarks['mouth']
            frame = videocapture.drawRectangle(detected, frame, self.criticPoints, mouth_range)
            frame = videocapture.drawCriticPoints(detected, frame, self.criticPoints, self.landmarks, mouth_range)
            # 将摄像头画面转换为PIL图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            # 缩放图像以适应窗口
            frame = frame.resize((800, 600), Image.Resampling.LANCZOS)
            # 将PIL图像转换为PhotoImage对象
            photoimage = ImageTk.PhotoImage(frame)
            # 更新Label的图像
            self.video_label.configure(image=photoimage)
            self.video_label.image = photoimage
            # 每隔100毫秒更新一次画面
            self.after(100, self.video_update)
            
        def launch_video_capture(self):
            # self.release_video_capture_button = tk.Button(master, text="关闭摄像头", command=release_video_capture)
            # self.release_video_capture_button.place(relx=0.4, rely=0.175, relwidth=0.25, relheight=0.1)
            return  
            self.cap = cv2.VideoCapture(0)
            self.video_update()
            
            while True:
                # _, frame = self.cap.read()
                # detected = self.detector(frame)
                # frame = videocapture.drawRectangle(detected, frame, self.criticPoints, mouth_range)
                # frame = videocapture.drawCriticPoints(detected, frame, self.criticPoints, self.landmarks, mouth_range)
                # photo = ImageTk.PhotoImage(frame)
                # self.image_label.configure(image=photo)
                # self.image_label.image = photo
                # cv2.imshow('frame', frame)
                self.video_update()
                key = cv2.waitKey(1)
                if key == 27:
                    break
            self.cap.release()
            cv2.destroyAllWindows()
            # while True:
            #     _,frame=cap.read()
            #     detected = detector(frame)
            #     frame = videocapture.drawRectangle(detected, frame, criticPoints, mouth_range)
            #     frame = videocapture.drawCriticPoints(detected, frame, criticPoints, landmarks, mouth_range)
            #     cov = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #     img = Image.fromarray(cov)
            #     img = ImageTk.PhotoImage(img)
            #     canvas.create_image(0,0,image=img)
                
            #     # key=cv2.waitKey(1)
            #     # if key == 27:
            #     #     break
            # cap.release()
            # cv2.destroyAllWindows()
                
        def dress_up(self):
            return
        
        self.current_recommend_item_label = tk.Label(virtual_try_on_window, text="", borderwidth = 3, relief="sunken")
        self.current_recommend_item_label.place(relx=0.05, rely=0.075, relwidth=0.9, relheight=0.05)
        
        self.display_color_image_button = tk.Button(virtual_try_on_window, text="展示色彩", command=virtual_try_on_display_color)
        self.display_color_image_button.place(relx=0.05, rely=0.175, relwidth=0.2, relheight=0.1)
        
        self.last_recommend_item_button = tk.Button(virtual_try_on_window, text="↑", command=last_recommend_item)
        self.last_recommend_item_button.place(relx=0.25, rely=0.175, relwidth=0.05, relheight=0.05)
        
        self.next_recommend_item_button = tk.Button(virtual_try_on_window, text="↓", command=next_recommend_item)
        self.next_recommend_item_button.place(relx=0.25, rely=0.225, relwidth=0.05, relheight=0.05)
        
        self.virtual_try_on_display_color_image_label = tk.Label(virtual_try_on_window, text="展\n示\n区", borderwidth = 3, relief="sunken")
        self.virtual_try_on_display_color_image_label.place(relx=0.3, rely=0.175, relwidth=0.05, relheight=0.1)
        
        self.launch_video_capture_button = tk.Button(virtual_try_on_window, text="打开摄像头", command=launch_video_capture)
        self.launch_video_capture_button.place(relx=0.4, rely=0.175, relwidth=0.25, relheight=0.1)
        
        self.dress_up_button = tk.Button(virtual_try_on_window, text="改变口红颜色", command=dress_up)
        self.dress_up_button.place(relx=0.7, rely=0.175, relwidth=0.25, relheight=0.1)
        
        self.video_capture_label = tk.Label(virtual_try_on_window, text="摄像头位置", borderwidth = 3, relief="sunken")
        self.video_capture_label.place(relx=0.05, rely=0.35, relwidth=0.9, relheight=0.6)

    def quit_app(self):
        exit()

pygame.init()
root = tk.Tk()
app = App(root)
root.mainloop()