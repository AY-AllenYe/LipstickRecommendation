# Input: 
#   1.HEX(fetch by the color plate or RGB (adjusted by movable switches) and show the color real-time)
#   2.Image(recommendation the lip's mean color, Not Done Yet)
#   3.launch_video_capture(Same as Image, Not Done Yet)
# Need: 
#   1.Btn*1(OpenImage...); SmallBtn*2(Left-Rotation?, Right-Rotation?)  ***
#   2.Btn*1(OpenCapture(On/Off)?)                                       *
#   3.Btn*3(FetchColor(and show)?, AdjustColor..., Return?)             ***
#   4.MoveableBtn*3(AdjustColor - R,G,B)                                ***
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
#   1.Clear
#   2.Quit
# Need:
#   1.Btn*1(Clear?)                                                     *
#   2.Btn*1(Quit?)                                                      *


import tkinter as tk
from tkinter import font, filedialog, messagebox, scrolledtext
# from tkinter.simpledialog import askinteger
from PIL import Image, ImageTk
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1" # Settings: No Welcome Message from pygame.
import pygame

import cv2
import dlib
import numpy as np
from collections import OrderedDict
import videocapture

class App:
    def __init__(self, master):
        self.master = master
        master.title("image2music")
        master.geometry("1000x800")

        self.image_label = tk.Label(master, text="图片位置")
        self.image_label.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.6)

        self.open_image_button = tk.Button(master, text="打开图片", command=self.open_image)
        self.open_image_button.place(relx=0.05, rely=0.05, relwidth=0.25, relheight=0.1)
        
        self.custom_font = font.Font(size=20, weight="bold") ## 修改字体与大小（需调用）
        
        self.right_rotate_button = tk.Button(master, text="↷", command=self.left_rotate, font=self.custom_font)
        self.right_rotate_button.place(relx=0.05, rely=0.2, relwidth=0.1, relheight=0.1)
        
        self.left_rotate_button = tk.Button(master, text="↶", command=self.right_rotate, font=self.custom_font)
        self.left_rotate_button.place(relx=0.25, rely=0.2, relwidth=0.1, relheight=0.1)
        
        self.launch_video_capture_button = tk.Button(master, text="打开摄像头", command=self.launch_video_capture)
        self.launch_video_capture_button.place(relx=0.05, rely=0.35, relwidth=0.25, relheight=0.1)
        
        self.video_label = tk.Label(master, text="图片位置")
        self.video_label.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.6)
        self.video_label.pack()

        self.fetch_color_button = tk.Button(master, text="提取色彩", command=self.fetch_color)
        self.adjust_color_button = tk.Button(master, text="调整色彩", command=self.adjust_color)

        self.setting_recommend_numbers_button = tk.Button(master, text="设置推荐数量\n默认推荐 10 支", command=self.setting_recommend_numbers)
        self.setting_recommend_numbers_button.place(relx=0.4, rely=0.675, relwidth=0.25, relheight=0.1)

        self.recommendation_button = tk.Button(master, text="识别图片\n生成推荐色号", command=self.recommendation)
        self.recommendation_button.place(relx=0.05, rely=0.5, relwidth=0.25, relheight=0.1)
        
        self.inference_logs = tk.Label(master, text="图片位置")
        # self.inference_logs.place(relx=0.05, rely=0.65, relwidth=0.3, relheight=0.3)
        
        # self.recommemdation_list = scrolledtext.ScrolledText(master, wrap=tk.WORD)
        self.display_info_button = tk.Button(master, text="展示商品信息", command=self.display_info)
        # self.goods_image_label = tk.Label(self.master)
        # self.result_labels = scrolledtext.ScrolledText(master, wrap=tk.WORD)

        self.virtual_try_on_button = tk.Button(master, text="试装？", command=self.virtual_try_on)
        self.virtual_try_on_button.place(relx=0.7, rely=0.825, relwidth=0.25, relheight=0.1)
                
        self.clear_text_button = tk.Button(master, text="清空推荐列表", command=self.clear_text)
        self.clear_text_button.place(relx=0.7, rely=0.675, relwidth=0.25, relheight=0.1)
        
        self.quit_app_button = tk.Button(master, text="关闭并退出", command=self.quit_app)
        self.quit_app_button.place(relx=0.7, rely=0.675, relwidth=0.25, relheight=0.1)

        self.cap = None
        self.cluster_file = 'datasets/lipstick_clusters.csv'
        self.rotate_angle = 0
        self.video_capture_landmarks = 'models\pretrained\shape_predictor_68_face_landmarks.dat'
        self.lipstick_label_predict = -1  # -1 未识别
        self.lipstick_recommend_list = []
        self.max_recommend_numbers = 10
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
    
    def open_image(self):
        App.clear_text(self)
        self.image_label = tk.Label(self.master)
        self.image_label.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.6)
        self.picture_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if self.picture_path:
            try:
                image = Image.open(self.picture_path)
                width, height = image.size
                if width > 500 or height > 500:
                    max_size = max(width, height)
                    new_width = int(width * 500 / max_size)
                    new_height = int(height * 500 / max_size)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.original_image = image
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except Exception as e:
                messagebox.showerror("Error", f"无法打开图片：{e}")

    def right_rotate(self):
        if hasattr(self, 'original_image'):
            self.rotate_angle += 90
            rotated_image = self.original_image.rotate(self.rotate_angle)
            # self.original_image = rotated_image
            
            photo = ImageTk.PhotoImage(rotated_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
    
    def left_rotate(self):
        if hasattr(self, 'original_image'):
            self.rotate_angle -= 90
            rotated_image = self.original_image.rotate(self.rotate_angle)
            # self.original_image = rotated_image
            
            photo = ImageTk.PhotoImage(rotated_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
    
    def launch_video_capture(self):      
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
    
    def fetch_color(self):
        return
    
    def adjust_color(self):
            # self.adjust_color_R_scale = tk.Scale(master, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
            # self.adjust_color_G_scale = tk.Scale(master, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
            # self.adjust_color_B_scale = tk.Scale(master, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
        # self.recover_color_button = tk.Button(master, text="复原色彩", command=self.recover_color)
        # self.comfirm_color_button = tk.Button(master, text="确认色彩", command=self.comfirm_color)
        return
    
    def recover_color(self):
        return
    
    def comfirm_color(self):
        return
    
    def setting_max_recommend_numbers(self, input):
        self.max_recommend_numbers = input
        return
    
    def loading_max_recommend_numbers(self):
        return self.max_recommend_numbers
    
    def setting_recommend_numbers(self):# Create a custom TopLevel window
        # self.max_recommend_numbers = askinteger(title = "请输入希望推荐的歌曲数目（一个整数）", prompt = "歌曲数目:", initialvalue = 10)
        custom_window = tk.Toplevel(root)
        custom_window.title("希望推荐的歌曲数目")
        custom_window.geometry("300x125+300+200")  # 宽x高+水平偏移量+垂直偏移量
        label = tk.Label(custom_window, text="请输入歌曲数目:（一个整数）")
        label.pack(pady=5)
        entry_var = tk.StringVar()
        entry_var.set("10")
        entry = tk.Entry(custom_window, textvariable=entry_var)
        entry.pack(pady=10)
        
        def get_integer():
            try:
                result = int(entry_var.get())
                # print("你期望推荐 " + str(result) + " 首歌")
                App.setting_max_recommend_numbers(self, input = result)
                messagebox.showinfo("Info", f"已修改：你期望推荐 " + str(result) + " 首歌")
                self.setting_recommend_numbers_button = tk.Button(self.master, text="设置推荐数量\n当前推荐 " + str(result) + " 首", command=self.setting_recommend_numbers)
                self.setting_recommend_numbers_button.place(relx=0.05, rely=0.35, relwidth=0.25, relheight=0.1)
            except ValueError:
                result = App.loading_max_recommend_numbers(self)
                messagebox.showwarning("Warning", f"无效修改。推荐数目保持 " + str(result) + " 首歌")
            custom_window.destroy()
        button = tk.Button(custom_window, text="确认修改", command=get_integer)
        button.pack(pady=5)
        entry.focus_set()  # 将焦点设置在输入框上
        custom_window.mainloop()
    
    def recommendation(self):
        # self.recognize_button = tk.Button(master, text="重新识别", command=self.)
        return 
        pygame.mixer.music.stop()
        self.inference_logs.delete(1.0, tk.END)
        self.music_path = None
        self.music_statu = 0  # 0 - 播放，1 - 暂停
        self.picture_label_predict = -1
        self.music_recommend_list = []
        App.update_playing_button(self)
        
        self.current_song = 0
        if self.picture_path:
            self.picture_label_predict = bp.recognize_picture("file:///" + self.picture_path)
            if self.picture_label_predict == 1:
                # result_str = "\n".join("happy")
                self.inference_logs.insert(tk.END, "图片是 轻快 的\n" + "\n")
            if self.picture_label_predict == 0:
                # result_str = "\n".join("quiet")
                self.inference_logs.insert(tk.END, "图片是 宁静 的\n" + "\n")
            self.music_recommend_list = bp.recommend_Music(self.picture_label_predict, self.max_recommend_numbers)
            result_str = "\n".join(self.music_recommend_list)
            self.inference_logs.insert(tk.END, "        推荐歌曲: 共计 " + str(self.max_recommend_numbers) + " 首\n" + result_str + "\n")
            
            self.recognize_button = tk.Button(self.master, text="重新生成推荐清单\n（停止音乐）", command=self.recommendation)
            self.recognize_button.place(relx=0.05, rely=0.5, relwidth=0.25, relheight=0.1)

    def display_info(self):
        return

    def virtual_try_on(self):
        return

    def clear_text(self):
        return
        pygame.mixer.music.stop()
        self.image_label.place_forget()
        self.inference_logs.delete(1.0, tk.END)
        
        self.picture_path = None
        self.music_path = None
        self.rotate_angle = 0
        self.music_statu = 0  # 0 - 播放，1 - 暂停
        self.picture_label_predict = -1
        self.music_recommend_list = []
        self.current_song = 0
        
        self.recognize_button = tk.Button(self.master, text="识别图片\n生成推荐音乐", command=self.recommendation)
        self.recognize_button.place(relx=0.05, rely=0.5, relwidth=0.25, relheight=0.1)
        App.update_playing_button(self)
        
    def quit_app(self):
        exit()

pygame.init()
root = tk.Tk()
app = App(root)
root.mainloop()
