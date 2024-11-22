import tkinter as tk
from tkinter import ttk, colorchooser
from util import get_mean_color, select_coordinate


class PopupWindow:
    def __init__(self, app, x, y, p):
        app.popup_open = True

        self.root = app.root
        self.app = app
        self.x = x
        self.y = y
        self.p = p
        self.true_color_hex = get_mean_color(self.app, x, y, p)
        self.selected_color = self.true_color_hex
        
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Select Color")
        self.popup.geometry("240x100")
        self.popup.resizable(False, False)
        
        self.checkbox_var = tk.BooleanVar()
        
        self.create_widgets()
        
        # Set the checkbox to selected
        self.checkbox_var.set(True)
        self.update_button_state()
        
        self.popup.protocol("WM_DELETE_WINDOW", self.on_destroy)

    def create_widgets(self):
        checkbox = ttk.Checkbutton(self.popup, text="Choose Original Color", variable=self.checkbox_var, command=self.update_button_state)
        checkbox.pack(pady=5)

        self.color_button = tk.Button(self.popup, text="Choose Color", command=self.select_color)
        self.color_button.pack(pady=5)

        ok_frame = tk.Frame(self.popup)
        ok_frame.pack(pady=5)

        self.color_label = tk.Label(ok_frame, text="     ", width=5, relief="solid", borderwidth=1)
        self.color_label.pack(side="left")

        ok_button = tk.Button(ok_frame, text="OK", command=self.on_okay)
        ok_button.pack(side="left", padx=10)

    def select_color(self):
        color_code = colorchooser.askcolor()[1]
        if color_code:
            self.selected_color = color_code
            self.color_label.config(bg=self.selected_color)

    def update_button_state(self):
        if self.checkbox_var.get():
            self.selected_color = self.true_color_hex
            self.color_button.config(state="disabled")
        else:
            self.selected_color = "#ffffff"
            self.color_button.config(state="normal")
        self.color_label.config(bg=self.selected_color)

    def on_okay(self):
        select_coordinate(self.app, self.x, self.y, self.p, self.selected_color)
        self.on_destroy()

    def on_destroy(self):
        self.app.popup_open = False
        self.popup.destroy()