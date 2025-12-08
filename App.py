# Input: 
#   1.HEX(fetch by the color plate or RGB (adjusted by movable switches) and show the color real-time)
#   2.Image(recommendation the lip's mean color, Not Done Yet)
#   3.Video_capture(Same as Image, Not Done Yet)
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
from tkinter import filedialog, messagebox, scrolledtext
# from tkinter.simpledialog import askinteger
from PIL import Image, ImageTk
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1" # Settings: No Welcome Message from pygame.
import pygame

class App:
    def __init__(self, master):
        self.master = master
        master.title("image2music")
        master.geometry("1000x800")

        self.image_label = tk.Label(master, text="图片位置")
        self.image_label.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.6)

        self.open_image_button = tk.Button(master, text="打开图片", command=self.open_image)
        self.open_image_button.place(relx=0.05, rely=0.05, relwidth=0.25, relheight=0.1)

        self.right_rotate_button = tk.Button(master, text="↷", command=self.right_rotate)
        self.right_rotate_button.place(relx=0.05, rely=0.2, relwidth=0.1, relheight=0.1)
        
        self.left_rotate_button = tk.Button(master, text="↶", command=self.left_rotate)
        self.left_rotate_button.place(relx=0.25, rely=0.2, relwidth=0.1, relheight=0.1)
        
        self.video_capture_button = tk.Button(master, text="打开摄像头", command=self.video_capture)
        self.video_capture_button.place(relx=0.05, rely=0.35, relwidth=0.25, relheight=0.1)

        # self.video_capture_button = tk.Button(master, text="提取色彩", command=self.)
        # self.video_capture_button = tk.Button(master, text="调整色彩", command=self.)
            # self.video_capture_button = tk.Scale(master, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
            # self.video_capture_button = tk.Scale(master, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
            # self.video_capture_button = tk.Scale(master, variable = tk.DoubleVar(), from_ = 0, to = 255, orient = tk.HORIZONTAL)
        # self.video_capture_button = tk.Button(master, text="复原色彩", command=self.)

        self.setting_recommend_numbers_button = tk.Button(master, text="设置推荐数量\n默认推荐 10 支", command=self.setting_recommend_numbers)
        self.setting_recommend_numbers_button.place(relx=0.4, rely=0.675, relwidth=0.25, relheight=0.1)

        self.recognize_button = tk.Button(master, text="识别图片\n生成推荐色号", command=self.recommendation)
        self.recognize_button.place(relx=0.05, rely=0.5, relwidth=0.25, relheight=0.1)
        
        # self.recognize_button = tk.Button(master, text="重新识别", command=self.)  已存在，在recommendation函数里。
        
        self.inference_logs = tk.Label(master, text="图片位置")
        # self.inference_logs.place(relx=0.05, rely=0.65, relwidth=0.3, relheight=0.3)
        
        # self.recommemdation_list = scrolledtext.ScrolledText(master, wrap=tk.WORD)
        # self.display_info_button = tk.Button(master, text="展示商品信息", command=self.)
        # self.goods_image_label = tk.Label(self.master)
        # self.result_labels = scrolledtext.ScrolledText(master, wrap=tk.WORD)

        self.virtual_try_on_button = tk.Button(master, text="试装？", command=self.virtual_try_on)
        self.virtual_try_on_button.place(relx=0.7, rely=0.825, relwidth=0.25, relheight=0.1)
                
        self.clear_text_button = tk.Button(master, text="清空推荐列表", command=self.clear_text)
        self.clear_text_button.place(relx=0.7, rely=0.675, relwidth=0.25, relheight=0.1)
        
        self.quit_app_button = tk.Button(master, text="关闭并退出", command=self.quit_app)
        self.quit_app_button.place(relx=0.7, rely=0.675, relwidth=0.25, relheight=0.1)


        self.cluster_file = 'datasets/lipstick_clusters.csv'
        self.rotate_angle = 0
        self.lipstick_label_predict = -1  # -1 未识别
        self.lipstick_recommend_list = []
        self.max_recommend_numbers = 10
    
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
                    image = image.resize((new_width, new_height), Image.ANTIALIAS)
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

    def recommendation(self):
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
        
    def update_playing_button(self):
        if self.picture_label_predict != -1:
            if self.music_statu == 0:
                self.video_capture_button = tk.Button(self.master, text="暂停" + "\n当前 正播放：" + str(self.music_recommend_list[self.current_song]), command=self.video_capture)
                self.video_capture_button.place(relx=0.4, rely=0.675, relwidth=0.25, relheight=0.1)
            elif self.music_statu == 1:
                self.video_capture_button = tk.Button(self.master, text="播放\n当前 已暂停播放", command=self.video_capture)
                self.video_capture_button.place(relx=0.4, rely=0.675, relwidth=0.25, relheight=0.1)
            
            self.last_song_button = tk.Button(self.master, text="播放上一首" + "\n上一首：" + str(self.music_recommend_list[(self.current_song + self.max_recommend_numbers - 1) % self.max_recommend_numbers]), command=self.last_song)
            self.last_song_button.place(relx=0.4, rely=0.825, relwidth=0.25, relheight=0.1)
            
            self.virtual_try_on_button = tk.Button(self.master, text="播放下一首" + "\n下一首：" + str(self.music_recommend_list[(self.current_song + 1) % self.max_recommend_numbers]), command=self.virtual_try_on)
            self.virtual_try_on_button.place(relx=0.7, rely=0.825, relwidth=0.25, relheight=0.1)
        else:
            self.video_capture_button = tk.Button(self.master, text="播放音乐", command=self.video_capture)
            self.video_capture_button.place(relx=0.4, rely=0.675, relwidth=0.25, relheight=0.1)
            
            self.last_song_button = tk.Button(self.master, text="播放上一首", command=self.last_song)
            self.last_song_button.place(relx=0.4, rely=0.825, relwidth=0.25, relheight=0.1)
            
            self.virtual_try_on_button = tk.Button(self.master, text="播放下一首", command=self.virtual_try_on)
            self.virtual_try_on_button.place(relx=0.7, rely=0.825, relwidth=0.25, relheight=0.1)
        
    def video_capture(self):
        # file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid")])
        if not self.music_path and self.picture_label_predict != -1:
            self.music_path = self.music_library_path + self.music_recommend_list[self.current_song] + '.mid'
            if not os.path.isfile(self.music_path):
                self.music_path = self.music_library_path + self.music_recommend_list[self.current_song] + '.MID'
            if not os.path.isfile(self.music_path):
                self.music_path = self.music_library_path + self.music_recommend_list[self.current_song] + '.wav'
            if not os.path.isfile(self.music_path):
                self.music_path = self.music_library_path + self.music_recommend_list[self.current_song] + '.WAV'
            if not os.path.isfile(self.music_path):
                self.music_path = self.music_library_path + self.music_recommend_list[self.current_song] + '.mp3'
            try:
                pygame.mixer.music.load(self.music_path)
                pygame.mixer.music.play()
                App.update_playing_button(self)
            except Exception as e:
                messagebox.showerror("Error", f"无法播放音乐：{e}")
        elif self.music_path:
            App.pause_song(self)
            App.update_playing_button(self)
                
    def pause_song(self):
        self.music_statu =  (self.music_statu + 1) % 2
        if self.music_statu == 1:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()
        
    def last_song(self):
        pygame.mixer.music.stop()
        self.music_path = None
        self.music_statu = 0
        self.current_song = (self.current_song + self.max_recommend_numbers - 1) % self.max_recommend_numbers
        App.video_capture(self)
                        
    def virtual_try_on(self):
        pygame.mixer.music.stop()
        self.music_path = None
        self.music_statu = 0
        self.current_song = (self.current_song + 1) % self.max_recommend_numbers
        App.video_capture(self)
    
    def clear_text(self):
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
