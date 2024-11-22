import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageOps
import colorize_image as CI
import numpy as np
import threading
from gui_popup import PopupWindow
from util import select_coordinate_no_color
from sampling_options import SamplingOption, get_sampling_option
from datetime import datetime

GRAYSCALE = "grayscale"
AI_COLORS = "ai_colors"
ACTUAL_COLORS = "actual_colors"

MODEL = "best_model"
FRAME_SIZE = 512
DEFAULT_P = 3
DEFAULT_SAMPLING_AMOUNT = 20


OUTPUT_PATH = "data/output/"


class MainApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ML Project: Mattis & Timon")
        width = FRAME_SIZE * 3 + 200
        height = FRAME_SIZE + 200
        
        self.root.geometry(f"{width}x{height}")
        
        # Define attributes 
        self.frame_size = FRAME_SIZE
        self.canvases = {}
        self.rectangles = []
        self.loading_frames = []
        self.first_image_loaded = False
        self.popup_open = False
        self.is_sample = False
        self.loading_animation_id = None

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = OUTPUT_PATH + timestamp + ".txt"
        with open(self.output_path, "w") as file:
            file.write("Sampling Method; Sampling Amount; Selection Radius; PSNR\n")
        


        self.reset_user_input()
        self.color_model = CI.ColorizeImageTorch(Xd=self.frame_size)
        self.color_model.prep_net(path=f'models/{MODEL}.pth')

        # Initialize UI
        self.create_title()
        self.configure_grid()
        self.create_controls_frame()

        self.load_placeholder_images()
        self.create_container(GRAYSCALE, row=3, column=0, label_text="Grayscale Image")
        self.create_container(ACTUAL_COLORS, row=3, column=1, label_text="Actual Colored Image")
        self.create_container(AI_COLORS, row=3, column=2, label_text="AI Colored Image")
        
        # Start main loop
        self.root.mainloop()


# Setup

    def load_placeholder_images(self):
        placeholder_image = Image.open("assets/placeholder.jpg").resize((self.frame_size, self.frame_size))
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_image)

        loading_image = Image.open("assets/loading.gif")
        for frame in range(loading_image.n_frames):
            loading_image.seek(frame)
            frame_image = loading_image.resize((self.frame_size, self.frame_size))
            self.loading_frames.append(ImageTk.PhotoImage(frame_image))

    def animate_loading_gif(self):
        frame = self.loading_frames[self.current_loading_frame]
        self.update_image(AI_COLORS, frame, PIL_image=False)
        
        self.current_loading_frame = (self.current_loading_frame + 1) % len(self.loading_frames)
        
        self.loading_animation_id = self.root.after(70, self.animate_loading_gif)


    def create_title(self):
        title_label = tk.Label(
            self.root,
            text=self.root.title(),
            font=("TkDefaultFont", 12, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(10, 0))

    def create_controls_frame(self):
        self.selected_sampling_var = tk.StringVar(value=SamplingOption.GAUSSIAN.display_name)
        
        # Master frame
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        # Main buttons
        main_buttons_frame = tk.Frame(self.controls_frame)
        main_buttons_frame.pack(pady=5)
        
        select_image_button = ttk.Button(main_buttons_frame, text="Select Image Path", command=self.on_select_image)
        select_image_button.pack(side="left", pady=5)

        self.forward_button = ttk.Button(main_buttons_frame, text="Generate AI Image", command=self.forward, state="disabled")
        self.forward_button.pack(side="left", pady=5)

        # Configure parameters
        parameters_frame = tk.Frame(self.controls_frame)
        parameters_frame.pack(pady=5)

        self.clear_input_button = ttk.Button(parameters_frame, text="Clear Input", command=self.on_clear_input, state="disabled")
        self.clear_input_button.pack(side="left")


        # Selection radius frame
        p_option_frame = tk.Frame(parameters_frame)
        p_option_frame.pack(side="left", padx=10)

        # Label to display the current value of the slider
        self.p_slider_label = ttk.Label(p_option_frame, text=f"Selection Radius: {DEFAULT_P}")
        self.p_slider_label.pack()

        # Scale for p value
        self.p_slider = ttk.Scale(
            p_option_frame,
            from_=0,
            to=5,
            value=DEFAULT_P,
            orient=tk.HORIZONTAL,
            state="disabled",
            command=self.update_p_value
        )
        self.p_slider.pack()

        # Sampling distribution frame
        sampling_method_frame = tk.Frame(parameters_frame)
        sampling_method_frame.pack(side="left", padx=10)
        
        ttk.Label(sampling_method_frame, text="Sampling Distribution:").pack()
        sampling_options = [method.display_name for method in SamplingOption]
        self.sampling_method_option = ttk.OptionMenu(sampling_method_frame, self.selected_sampling_var, SamplingOption.GAUSSIAN.display_name, *sampling_options)
        self.sampling_method_option.config(state="disabled")
        self.sampling_method_option.pack()

        # Sampling amount frame
        sampling_amount_frame = tk.Frame(parameters_frame)
        sampling_amount_frame.pack(side="left", padx=10)

        # Label to display the current value of the slider
        self.sampling_amount_value_label = ttk.Label(sampling_amount_frame, text=f"Sampling Amount: {DEFAULT_SAMPLING_AMOUNT}")
        self.sampling_amount_value_label.pack()

        # Scale for sampling amount
        self.sampling_amount_slider = ttk.Scale(
            sampling_amount_frame,
            from_=5,
            to=400,
            value=DEFAULT_SAMPLING_AMOUNT,
            orient=tk.HORIZONTAL,
            state="disabled",
            command=self.update_sampling_amount_value
        )
        self.sampling_amount_slider.pack()


        # Sample button
        self.sample_button = ttk.Button(parameters_frame, text="Sample", command=self.on_sample, state="disabled")
        self.sample_button.pack(side="left", padx=10)



    def update_sampling_amount_value(self, value):
        self.sampling_amount_value_label.config(text=f"Sampling Amount: {int(float(value))}")

    def update_p_value(self, value):
        self.p_slider_label.config(text=f"Selection Radius: {int(float(value))}")

    def get_p_value(self):
        return int(float(self.p_slider.get()))
    
    def get_sampling_amount(self):
        return int(float(self.sampling_amount_slider.get()))


    def configure_grid(self):
        # Configure rows and columns to resize proportionally
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)

    def create_container(self, name, row, column, label_text):
        container = tk.Frame(self.root, width=self.frame_size, height=self.frame_size + 30)
        container.grid(row=row, column=column, sticky="nsew", padx=10, pady=10)

        label = tk.Label(container, text=label_text, font=("TkDefaultFont", 10, "bold"))
        label.pack(pady=(5, 0))

        if name == AI_COLORS:
            self.ai_colors_label = label

        frame = tk.Canvas(container, width=self.frame_size, height=self.frame_size)
        frame.pack() 
        frame.create_image(0, 0, image=self.placeholder_photo, anchor="nw") 

        if name == GRAYSCALE:
            frame.bind("<Button-1>", self.on_grayscale_click)

        self.canvases[name] = frame

    def reset_user_input(self):
        self.user_input = (np.zeros((2, self.frame_size, self.frame_size)), np.zeros((1, self.frame_size, self.frame_size)))

# Handling color selection

    def on_grayscale_click(self, event):
        if not self.first_image_loaded or self.popup_open:
            return
        
        self.is_sample = False
        PopupWindow(self, event.x, event.y, self.get_p_value())

    def draw_cell(self, x1, y1, x2, y2, color):
        rectangle = self.canvases[GRAYSCALE].create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        self.rectangles.append(rectangle)

# Main buttons

    def on_clear_input(self):
        self.is_sample = False
        for rectangle in self.rectangles:
            self.canvases[GRAYSCALE].delete(rectangle)
        self.reset_user_input()

    def on_select_image(self):
        img_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if img_path:
            self.first_image_loaded = True
            self.update_image(AI_COLORS, self.placeholder_photo, PIL_image=False)
            self.ai_colors_label.config(text="AI Colored Image")
            self.reset_user_input()

            self.forward_button.config(state="normal")

            self.p_slider.config(state="normal")
            self.clear_input_button.config(state="normal")
            self.sampling_method_option.config(state="normal")
            self.sampling_amount_slider.config(state="normal")
            self.sample_button.config(state="normal")

            image = Image.open(img_path).resize((self.frame_size, self.frame_size))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.update_image(GRAYSCALE, ImageOps.grayscale(image))
            self.update_image(ACTUAL_COLORS, image)

            self.np_image = np.array(image)
            self.load_new_image(img_path)

    def on_sample(self):
        self.on_clear_input()
        self.is_sample = True

        option = get_sampling_option(self.selected_sampling_var.get())
        cords = option.call_function(self.get_sampling_amount(), self.frame_size)
        for x, y in cords:
            select_coordinate_no_color(self, x, y, self.get_p_value())


# Image loading and processing

    def update_image(self, label_name, photo, PIL_image=True):
        if PIL_image:
            photo = ImageTk.PhotoImage(photo)

        canvas = self.canvases[label_name]
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image = photo

    def load_new_image(self, img_path):
        threading.Thread(target=self.color_model.load_image, args=(img_path,)).start()

    def forward(self):
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
        
        self.current_loading_frame = 0
        self.animate_loading_gif()
        threading.Thread(target=self._forward).start()

    def _forward(self):
        ai_colored_image = self.color_model.net_forward(*self.user_input)
        ai_colored_pil = Image.fromarray(ai_colored_image)

        if self.is_sample:
            with open(self.output_path, "a") as file:
                file.write(f"{self.selected_sampling_var.get()}; {self.get_sampling_amount()}; {self.get_p_value()}; {self.color_model.get_result_PSNR()}\n")
        
        self.ai_colors_label.config(text=f"AI Colored Image, PSNR: {self.color_model.get_result_PSNR()}")

        self.root.after_cancel(self.loading_animation_id)
        self.update_image(AI_COLORS, ai_colored_pil)



if __name__ == "__main__":
    app = MainApp()
