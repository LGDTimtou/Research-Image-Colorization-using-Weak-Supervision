import numpy as np
from skimage.color import rgb2lab


def select_coordinate_no_color(app, x, y, p):
    color = get_mean_color(app, x, y, p)
    select_coordinate(app, x, y, p, color)


def select_coordinate(app, x, y, p, selected_color):
    selected_color_rgb = tuple(int(selected_color[i:i+2], 16) for i in (1, 3, 5))
    selected_color_lab = rgb2lab(np.array([[selected_color_rgb]], dtype=np.uint8) / 255.0)[0, 0, 1:]
    a_channel, b_channel = selected_color_lab

    x1, y1, x2, y2 = get_surrounding_coords(app, x, y, p)

    app.user_input[0][0, y1:y2, x1:x2] = a_channel  
    app.user_input[0][1, y1:y2, x1:x2] = b_channel
    app.user_input[1][0, y1:y2, x1:x2] = 1

    app.draw_cell(x1, y1, x2, y2, selected_color)


def get_surrounding_coords(app, x, y, p):
    left = max(x - p, 0)
    right = min(x + p + 1, app.frame_size)
    top = max(y - p, 0)
    bottom = min(y + p + 1, app.frame_size)
    return left, top, right, bottom


def get_mean_color(app, x, y, p):
    left, top, right, bottom = get_surrounding_coords(app, x, y, p)
    region = app.np_image[top:bottom, left:right]
    mean_color = region.mean(axis=(0, 1)).astype(int)

    return "#{:02x}{:02x}{:02x}".format(*mean_color)