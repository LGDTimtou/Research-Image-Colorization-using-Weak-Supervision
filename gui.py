import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
from skimage.color import rgb2lab
from PIL import Image, ImageTk, ImageOps
import colorize_image as CI
import numpy as np
import threading

GRAYSCALE = "grayscale"
AI_COLORS = "ai_colors"
ACTUAL_COLORS = "actual_colors"

MODEL = "caffemodel"
FRAME_SIZE = 512
P = 3

class MainApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ML Project: Mattis & Timon")
        width = FRAME_SIZE * 3 + 200
        heigth = FRAME_SIZE + 200
        
        self.root.geometry(f"{width}x{heigth}")
        
        # Define attributes 
        self.frame_size = FRAME_SIZE
        self.canvases = {}
        self.containers = {}
        self.rectangles = []
        self.loading_frames = []
        self.first_image_loaded = False
        self.popup_open = False

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
        placeholder_image = Image.open("assets/placeholder.jpg")
        placeholder_image = placeholder_image.resize((self.frame_size, self.frame_size))
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
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        select_image_button = ttk.Button(self.controls_frame, text="Select Image Path", command=self.on_select_image)
        select_image_button.pack(side="left", pady=5)

        self.clear_input_button = ttk.Button(self.controls_frame, text="Clear User Input", command=self.on_clear_input, state="disabled")
        self.clear_input_button.pack(side="left", padx=10, pady=5)

        self.forward_button = ttk.Button(self.controls_frame, text="Generate AI Image", command=self.forward, state="disabled")
        self.forward_button.pack(side="left", pady=5)


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

        frame = tk.Canvas(container, width=self.frame_size, height=self.frame_size)
        frame.pack() 
        frame.create_image(0, 0, image=self.placeholder_photo, anchor="nw") 

        if name == GRAYSCALE:
            frame.bind("<Button-1>", self.on_grayscale_click)

        self.canvases[name] = frame
        self.containers[name] = frame

    def reset_user_input(self):
        self.user_input = (np.zeros((2,self.frame_size,self.frame_size)), np.zeros((1,self.frame_size,self.frame_size)))


# Handling color selection

    def on_grayscale_click(self, event):
        if not self.first_image_loaded or self.popup_open:
            return
        
        x, y = event.x, event.y
        true_color_hex = self.get_mean_color(x, y)

        self.popup_open = True

        popup = tk.Toplevel(self.root, )
        popup.title("Select Color")
        popup.geometry("240x100")
        popup.resizable(False, False)

        # Initialize variables
        checkbox_var = tk.BooleanVar()
        self.selected_color = true_color_hex

        # Initialize commands
        def select_color():
            color_code = colorchooser.askcolor()[1]
            if color_code:
                self.selected_color = color_code
                color_label.config(bg=self.selected_color)

        def update_button_state():
            if checkbox_var.get():
                self.selected_color = true_color_hex
                color_button.config(state="disabled")
            else:
                self.selected_color = "#ffffff"
                color_button.config(state="normal")
            color_label.config(bg=self.selected_color)

        def on_okay():
            selected_color_rgb = tuple(int(self.selected_color[i:i+2], 16) for i in (1, 3, 5))
            selected_color_lab = rgb2lab(np.array([[selected_color_rgb]], dtype=np.uint8) / 255.0)[0, 0, 1:]
            a_channel, b_channel = selected_color_lab

            x1, y1, x2, y2 = self.get_surrounding_coords(x, y)

            self.user_input[0][0, y1:y2, x1:x2] = a_channel  
            self.user_input[0][1, y1:y2, x1:x2] = b_channel
            self.user_input[1][0, y1:y2, x1:x2] = 1

            self.draw_cell(x1, y1, x2, y2, self.selected_color)

            on_destroy()

        def on_destroy():
            self.popup_open = False
            popup.destroy()
        popup.protocol("WM_DELETE_WINDOW", on_destroy)


        # Initialize fields
        checkbox = ttk.Checkbutton(popup, text="Choose Original Color", variable=checkbox_var, command=update_button_state)
        checkbox.pack(pady=5)

        color_button = tk.Button(popup, text="Choose Color", command=select_color)
        color_button.pack(pady=5)

        ok_frame = tk.Frame(popup)
        ok_frame.pack(pady=5)

        color_label = tk.Label(ok_frame, text="     ", width=5, relief="solid", borderwidth=1)
        color_label.pack(side="left")

        ok_button = tk.Button(ok_frame, text="OK", command=on_okay)
        ok_button.pack(side="left", padx=10)

        # Set the checkbox to selected
        checkbox_var.set(True)
        update_button_state()

    def get_surrounding_coords(self, x, y):
        left = max(x - P, 0)
        right = min(x + P + 1, self.frame_size)
        top = max(y - P, 0)
        bottom = min(y + P + 1, self.frame_size)

        return left, top, right, bottom

    def draw_cell(self, x1, y1, x2, y2, color):
        rectangle = self.canvases[GRAYSCALE].create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        self.rectangles.append(rectangle)


    def get_mean_color(self, x, y):
        left, top, right, bottom = self.get_surrounding_coords(x, y)
        region = self.np_image[top:bottom, left:right]
        mean_color = region.mean(axis=(0, 1)).astype(int)

        return "#{:02x}{:02x}{:02x}".format(*mean_color)



# Main buttons

    def on_clear_input(self):
        for rectangle in self.rectangles:
            self.canvases[GRAYSCALE].delete(rectangle)
        self.reset_user_input()

    def on_select_image(self):
        # Open file selection dialog
        img_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if img_path:
            self.first_image_loaded = True
            self.clear_input_button.config(state="normal")
            self.forward_button.config(state="normal")

            self.reset_user_input()

            image = Image.open(img_path)
            image = image.resize((self.frame_size, self.frame_size))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.update_image(GRAYSCALE, ImageOps.grayscale(image))
            self.update_image(ACTUAL_COLORS, image)

            self.np_image = np.array(image)

            self.load_new_image(img_path)



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
        self.current_loading_frame = 0
        self.animate_loading_gif()
        threading.Thread(target=self._forward).start()

    def _forward(self):
        ai_colored_image = self.color_model.net_forward(*self.user_input)
        ai_colored_pil = Image.fromarray(ai_colored_image)

        self.root.after_cancel(self.loading_animation_id)

        self.update_image(AI_COLORS, ai_colored_pil)




if __name__ == "__main__":
    app = MainApp()
