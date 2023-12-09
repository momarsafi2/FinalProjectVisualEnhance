import argparse
import threading
import cv2
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import imutils
import time
import pandas as pd
from scipy.ndimage import gaussian_filter
from tkinter import filedialog
import random
import math
import os

matplotlib.use('TkAgg')

resized_image = None
original_image = None
blurred_image = None

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

# Load Image
def load_image():
    global resized_image
    global original_image
    global blurred_image

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])

    if file_path:
        print(f'Loading {file_path} ... ', end='')
        original_image = cv2.imread(file_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        target_width = 640
        target_height = None
        resized_image = resize_image(original_image, target_width, target_height, True, cv2.INTER_NEAREST)
        blurred_image = resized_image  


# Crop image
def crop_image(image, left, top, right, bottom):
    left, right = min(left, right), max(left, right)
    top, bottom = min(top, bottom), max(top, bottom)

    cropped_image = image[top:bottom, left:right]
    return cropped_image


def construct_image_histogram(np_image):
    hist, _ = np.histogram(np_image.ravel(), bins=256, range=[0, 256])
    return hist


def draw_hist(canvas, figure, key='-HIST-'):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return tkcanvas

# Resize image
def resize_image(image, target_width, target_height, maintain_aspect_ratio, interpolation_method):
    if maintain_aspect_ratio:
        aspect_ratio = image.shape[1] / image.shape[0]

        target_height = int(target_width / aspect_ratio)

    max_width = target_width
    max_height = target_height

    if target_width is not None and target_height is not None:
        max_width = min(target_width, image.shape[1])
        max_height = min(target_height, image.shape[0])

    resized_image = cv2.resize(image, (max_width, max_height), interpolation=interpolation_method)

    return resized_image


#  Sephia tone
def apply_sepia_tone(image):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(image, sepia_matrix)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image


#  Vintage effect
def apply_vintage_effect(image):
    sepia_image = apply_sepia_tone(image)

    warm_filter = np.array([[1.2, 0.2, 0],
                            [0, 1, 0],
                            [0, 0, 0.8]])
    vintage_image = cv2.transform(sepia_image, warm_filter)

    alpha = 1.2
    beta = 10
    vintage_image = cv2.addWeighted(vintage_image, alpha, np.zeros_like(vintage_image), 0, beta)

    return vintage_image

#  sketch effect
def apply_sketch_effect(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    smoothed_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)

    edges = cv2.Laplacian(smoothed_image, cv2.CV_8U, ksize=5)

    _, mask = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY_INV)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    sketch_image = cv2.addWeighted(image, 0.8, mask_rgb, 0.2, 0)

    return sketch_image

#  film effect
def apply_film_effect(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.xphoto.createSimpleWB().balanceWhite(image)

    rows, cols, _ = image.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 2, (255, 255, 255), -1, cv2.LINE_AA)

    kernel_size = (cols // 3 * 2 + 1, rows // 3 * 2 + 1)
    mask = cv2.GaussianBlur(mask, kernel_size, 0)

    result = cv2.addWeighted(image, 1.2, cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), 0.5, 0)
    result = cv2.bitwise_and(result, result, mask=mask)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

#  burn effect
def apply_burn_effect(image, intensity):
    intensity = max(0, min(1, intensity))

    burned_image = (image * (1 - intensity)).astype(np.uint8)

    return burned_image

#  Invert Effect
def apply_invert_colors(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

# Color Balance Effect
def apply_color_balance(image, blue=0, green=0, red=0):
    color_matrix = np.array([[1 + blue / 100, 0, 0],
                             [0, 1 + green / 100, 0],
                             [0, 0, 1 + red / 100]], dtype=np.float32)
    balanced_image = cv2.transform(image, color_matrix)
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    return balanced_image

# Emboss Effect
def apply_emboss_effect(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    return embossed_image

# Grayscale 
def apply_black_and_white(image):
    black_and_white = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    black_and_white_color = cv2.cvtColor(black_and_white, cv2.COLOR_GRAY2BGR)

    return black_and_white_color

# Red Effect
def apply_reddish_filter(image):
    reddish_image = image.copy()
    reddish_image[:, :, 1] = 0  
    reddish_image[:, :, 2] = 0  

    return reddish_image

# Glitch Effect
def apply_glitch_effect(image, intensity):
    rows, cols, _ = image.shape
    glitch_mask = np.random.normal(0, intensity, (rows, cols, 3)).astype(np.uint8)
    glitched_image = cv2.add(image, glitch_mask)
    return np.clip(glitched_image, 0, 255).astype(np.uint8)

    

# Blur effect
def apply_blur(image, averaging_filter_size, gaussian_filter_size):
    averaged_image = cv2.blur(image, (averaging_filter_size * 2 + 1, averaging_filter_size * 2 + 1))
    gaussian_image = cv2.GaussianBlur(image, (gaussian_filter_size * 2 + 1, gaussian_filter_size * 2 + 1), 0)
    
    combined_image = cv2.addWeighted(averaged_image, 0.5, gaussian_image, 0.5, 0)
    
    return combined_image

def draw_strokes(image, stroke_width_range, stroke_length_range):
    for i in range(0, image.shape[0], 10):
        for j in range(0, image.shape[1], 10):
            color = tuple(map(int, image[i, j]))  
            stroke_width = random.randint(*stroke_width_range)
            stroke_length = random.randint(*stroke_length_range)
            angle = math.radians(45)
            end_x = int(j + stroke_length * math.cos(angle))
            end_y = int(i + stroke_length * math.sin(angle))
            cv2.line(image, (j, i), (end_x, end_y), color, stroke_width)

def draw_strokes_perpendicular_to_gradient(image, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if gradient_magnitude[i, j] > threshold:
                neighborhood_magnitude = gradient_magnitude[i-1:i+2, j-1:j+2]
                max_magnitude_index = np.unravel_index(np.argmax(neighborhood_magnitude), neighborhood_magnitude.shape)

                perpendicular_angle = gradient_direction[i + max_magnitude_index[0] - 1, j + max_magnitude_index[1] - 1] + math.radians(90)

                color = tuple(map(int, image[i, j]))
                stroke_width = random.randint(1, 1)   
                stroke_length = random.randint(2, 5)   

                angle_offset = math.radians(45)
                #angle_offset = math.radians(random.uniform(-45, 45))
                perpendicular_angle += angle_offset

                end_x = int(j + stroke_length * math.cos(perpendicular_angle))
                end_y = int(i + stroke_length * math.sin(perpendicular_angle))

                cv2.line(image, (j, i), (end_x, end_y), color, stroke_width)

def flip_image_horizontal(image):
    return np.fliplr(image)

def flip_image_vertical(image):
    return np.flipud(image)

def rotate_image(image, angle):
    rotated_image = imutils.rotate(image, angle)
    return rotated_image

def apply_adjustments(image, brightness=0, exposure=0, contrast=0, highlights=0, tint=0, saturation=0, warmth=0):
    
    
    # brightness 
    adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness * 2) 

    # exposure 
    adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=1, beta=exposure)

    # contrast 
    adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=1 + contrast / 100, beta=0)

    # highlights 
    adjusted_image = np.where(adjusted_image > 255 - highlights, 255, adjusted_image + highlights)

    # tint 
    tint_matrix = np.array([[1, 0, 0],
                            [0, 1 + tint / 100, 0],
                            [0, 0, 1]], dtype=np.float32)
    adjusted_image = cv2.transform(adjusted_image, tint_matrix)
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    # saturation
    hsv_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + saturation, 0, 255)
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # warmth
    if warmth != 0:
        kelvin = 5000 + warmth * 50
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2LAB)
        adjusted_image[:, :, 0] = np.clip(adjusted_image[:, :, 0] + (kelvin - 5000) / 50, 0, 100)
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_LAB2RGB)

    return adjusted_image

# Zoom in and out
def zoom_image(image, factor):
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)

    zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return zoomed_image


def display_image(image, filename, width, height, maintain_aspect_ratio, interpolation_method, averaging_filter_size=0, gaussian_filter_size=0, stroke_width_range=(1, 5), stroke_length_range=(5, 20), edge_threshold=30, title_element=None, logo_element=None):
    global resized_image
    global blurred_image

    blurred_image = apply_blur(image, averaging_filter_size, gaussian_filter_size)

    before_hist = construct_image_histogram(resized_image)

    original_image_data = np_im_to_data(image)
    blurred_image_data = np_im_to_data(blurred_image)

    original_width, original_height, channels = image.shape

    # Styling** 
    # buttons
    # button_design = {'size': (10, 1), 'button_color': ('#FFFFFF', '#0078D4')}
    button_design = {'size': (15, 1), 'button_color': ('#FFFFFF', '#555'), 'font': ('Helvetica', 12), 'border_width': 2 }

    # Theme
    sg.theme('DarkGrey5')
    
    # background image
    #background_image_path = r''


    layout = [
    [logo_element, title_element],
    [sg.Text('', size=(1, 1))],
    [sg.Frame('File Menu', [[sg.Button('Save Image', **button_design), sg.Button('Load Image', **button_design), sg.Button('Exit', **button_design)]]),
    sg.Frame('View Controls', [[sg.Button('Zoom In', **button_design), sg.Button('Zoom Out', **button_design), sg.Button('Reset', **button_design)]]),],
    #[sg.Text(f'File: {filename}, Size: {original_width}x{original_height}')],
    [
            sg.Column([
            [sg.Canvas(key='-HIST-', size=(100, 50))],
            [sg.Frame('Original Image', [
                [sg.Graph(
                    canvas_size=(resized_image.shape[1], resized_image.shape[0]),
                    graph_bottom_left=(0, 0),
                    graph_top_right=(resized_image.shape[1], resized_image.shape[0]),
                    key='-IMAGE-',
                    background_color='white',
                    change_submits=True,
                    drag_submits=True),
                ],
            ], border_width=15, relief=sg.RELIEF_SUNKEN),  
             sg.Frame('Edited Image', [
                [sg.Graph(
                    canvas_size=(resized_image.shape[1], resized_image.shape[0]),
                    graph_bottom_left=(0, 0),
                    graph_top_right=(resized_image.shape[1], resized_image.shape[0]),
                    key='-BLURRED_IMAGE-',
                    background_color='white',
                    change_submits=True,
                    drag_submits=True)],
            ], border_width=15, relief=sg.RELIEF_SUNKEN)],  
        ]),
        sg.Column([
            [sg.Frame('Adjustments', [
                [sg.Text('Brightness:', size=(10, 1)), sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-BRIGHTNESS-')],
                [sg.Text('Exposure:', size=(10, 1)), sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-EXPOSURE-')],
                [sg.Text('Contrast:', size=(10, 1)), sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-CONTRAST-')],
                [sg.Text('Highlights:', size=(10, 1)), sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-HIGHLIGHTS-')],
                [sg.Text('Tint:', size=(10, 1)), sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-TINT-')],
                [sg.Text('Saturation:', size=(10, 1)), sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-SATURATION-')],
                [sg.Text('Warmth:', size=(10, 1)), sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-WARMTH-')],
                [sg.Button('Apply', key='-APPLY-')],
            ])],
            [sg.Frame('Filter Effects', [
                [sg.Button('Color Balance'), sg.Button('Glitch Effect'), sg.Button('Red Effect'), sg.Button('Invert Effect'), sg.Button('Emboss Effect')],
                [sg.Button('Sepia Tone'), sg.Button('Vintage Effect'), sg.Button('Sketch Effect'), sg.Button('Film Effect'), sg.Button('Burn Effect')],
                [sg.Button('Grayscale'), sg.Button('Histogram Equalization'), sg.Button('HSV to RGB'), sg.Button('Painted Look'),]
            ])],
            [sg.Frame('Image Manipulation Controls', [
                [sg.Button('Crop'), sg.Button('Resize'), sg.Button('Rotate Clockwise'), sg.Button('Rotate Counterclockwise')],
                [sg.Button('Flip Horizontal'), sg.Button('Flip Vertical')], 
                [sg.Text('Left'), sg.Slider(range=(0, image.shape[1]), default_value=0, orientation='h', key='-LEFT-')],
                [sg.Text('Top'), sg.Slider(range=(0, image.shape[0]), default_value=0, orientation='h', key='-TOP-')],
                [sg.Text('Right'), sg.Slider(range=(0, image.shape[1]), default_value=image.shape[1], orientation='h', key='-RIGHT-')],
                [sg.Text('Bottom'), sg.Slider(range=(0, image.shape[0]), default_value=image.shape[0], orientation='h', key='-BOTTOM-')],
            ])],
            [sg.Frame('Filter Controls', [
                [sg.Text('Averaging Filter Size:'), sg.Slider(range=(0, 15), default_value=0, orientation='h', key='-AVG_SLIDER-')],
                [sg.Text('Gaussian Filter Size:'), sg.Slider(range=(0, 15), default_value=0, orientation='h', key='-GAUSS_SLIDER-')],
                [sg.Button('Refresh')]
            ])],
        ], scrollable=True, vertical_scroll_only=True, size=(width, height)),
    ],      [sg.Text(f'File: {filename}, Size: {original_width}x{original_height}')],
            [sg.Text('Copyright © 2023 - Omar Safi - 100830933 - Computational Photography', justification='center', size=(width, 1), background_color='#555555')],
            
]
    

    window = sg.Window('Project Application', layout, finalize=True, resizable=True)
    window['-IMAGE-'].draw_image(data=original_image_data, location=(0, resized_image.shape[0]))
    window['-BLURRED_IMAGE-'].draw_image(data=blurred_image_data, location=(0, resized_image.shape[0]))
 
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Grayscalee':
            blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
        elif event == 'Histogram Equalization':
            if channels == 1:  # Grayscale image
                equalized_image = cv2.equalizeHist(blurred_image)
            else:  # RGB image
                equalized_image = blurred_image.copy()
                equalized_image[:, :, 0] = cv2.equalizeHist(blurred_image[:, :, 0])
                equalized_image[:, :, 1] = cv2.equalizeHist(blurred_image[:, :, 1])
                equalized_image[:, :, 2] = cv2.equalizeHist(blurred_image[:, :, 2])

            after_hist = construct_image_histogram(equalized_image)

            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(
                equalized_image), location=(0, resized_image.shape[0]))

           # fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
           # ax.bar(np.arange(len(before_hist)), before_hist,
           #        alpha=0.5, label='Before Equalization')
           # ax.bar(np.arange(len(after_hist)), after_hist,
           #        alpha=0.5, label='After Equalization')
           # plt.title('Histogram - After')
           # draw_hist(window['-HIST-'].TKCanvas, fig, key='-HIST-')
        elif event == 'HSV to RGB':
            blurred_image = cv2.cvtColor(
                blurred_image, cv2.COLOR_HSV2RGB)  # Convert HSV to RGB
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image),
                                         location=(0, resized_image.shape[0]))
        elif event == '-AVG_SLIDER-' or event == '-GAUSS_SLIDER-' or event == 'Refresh':
            averaging_filter_size = int(values['-AVG_SLIDER-'])
            gaussian_filter_size = int(values['-GAUSS_SLIDER-'])

            aspect_ratio = original_image.shape[1] / original_image.shape[0]

            max_width = 640  
            target_width = min(max_width, original_image.shape[1])

            target_height = int(target_width / aspect_ratio)

            blurred_image = resize_image(original_image, target_width, target_height, True, cv2.INTER_NEAREST)
            # resized_image = resize_image(original_image, target_width, target_height, True, cv2.INTER_LINEAR)


            blurred_image = apply_blur(resized_image, averaging_filter_size, gaussian_filter_size)

            #window['-IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].erase()
            #window['-IMAGE-'].draw_image(data=np_im_to_data(resized_image), location=(0, resized_image.shape[0]))
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))


        elif event == 'Resize':
            
            resize_layout = [
                [sg.Text('Width:'), sg.InputText(default_text=str(original_width), key='-WIDTH-')],
                [sg.Text('Height:'), sg.InputText(default_text=str(original_height), key='-HEIGHT-')],
                [sg.Checkbox('Maintain Aspect Ratio', default=True, key='-MAINTAIN_ASPECT-')],
                [sg.Radio('Nearest Neighbor', 'RESIZE_METHOD', default=True, key='-NEAREST_NEIGHBOR-'),
                 sg.Radio('Bilinear', 'RESIZE_METHOD', key='-BILINEAR-')],
                [sg.Button('Resize'), sg.Button('Cancel')]
            ]
            resize_window = sg.Window('Resize Options', resize_layout)
            while True:
                resize_event, resize_values = resize_window.read()
                if resize_event == sg.WIN_CLOSED or resize_event == 'Cancel':
                    break
                elif resize_event == 'Resize':
                    try:
                        target_width = int(resize_values['-WIDTH-'])
                        target_height = int(resize_values['-HEIGHT-'])
                        maintain_aspect_ratio = resize_values['-MAINTAIN_ASPECT-']
                        interpolation_method = cv2.INTER_NEAREST if resize_values['-NEAREST_NEIGHBOR-'] else cv2.INTER_LINEAR

                        blurred_image = resize_image(
                            original_image, target_width, target_height, maintain_aspect_ratio, interpolation_method)

                        
                       # window['-IMAGE-'].erase()
                       # window['-IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, blurred_image.shape[0]))
                        window['-BLURRED_IMAGE-'].erase()
                        window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, blurred_image.shape[0]))
                    except ValueError:
                        sg.popup('Invalid input! Please enter a valid number.')
                    break
            resize_window.close()
        
        elif event == 'Painted Look': # Painted Look image
            draw_strokes_perpendicular_to_gradient(blurred_image, edge_threshold)
            blurred_image_data = np_im_to_data(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=blurred_image_data, location=(0, resized_image.shape[0]))
            
        elif event == 'Sepia Tone':
            blurred_image = apply_sepia_tone(blurred_image) 
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0])) 
            
        elif event == 'Vintage Effect':
            blurred_image = apply_vintage_effect(blurred_image) 
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
            
        elif event == 'Sketch Effect':
            blurred_image = apply_sketch_effect(blurred_image) 
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))    
        
        elif event == 'Film Effect':  
            blurred_image = apply_film_effect(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
    
        elif event == 'Burn Effect':
            intensity = 0.5  
            blurred_image = apply_burn_effect(blurred_image, intensity)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
         
        elif event == 'Invert Effect':
            blurred_image = apply_invert_colors(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
            
        elif event == 'Color Balance':
            blurred_image = apply_color_balance(blurred_image, blue=10, green=0, red=-10)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
            
        elif event == 'Emboss Effect':
            blurred_image = apply_emboss_effect(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))      

        elif event == 'Grayscale':
            blurred_image = apply_black_and_white(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
            
        elif event == 'Red Effect':
            blurred_image = apply_reddish_filter(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0])) 
         
        elif event == 'Glitch Effect':
            intensity = 1
            blurred_image = apply_glitch_effect(blurred_image, intensity)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))

        elif event == 'Crop':
         try:
            left = int(values['-LEFT-'])
            top = int(values['-TOP-'])
            right = int(values['-RIGHT-'])
            bottom = int(values['-BOTTOM-'])
            
        
            left, right = min(left, right), max(left, right)
            top, bottom = min(top, bottom), max(top, bottom)

            cropped_image = crop_image(blurred_image, left, top, right, bottom)

            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(cropped_image), location=(0, cropped_image.shape[0]))

            # Update sliders
            window['-LEFT-'].update(value=left)
            window['-TOP-'].update(value=top)
            window['-RIGHT-'].update(value=right)
            window['-BOTTOM-'].update(value=bottom)

         except ValueError:
            sg.popup('Invalid input! Please enter valid integers for coordinates.')

        elif event == 'Rotate Clockwise':
            blurred_image = rotate_image(blurred_image, -90)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
            blurred_image = blurred_image
        elif event == 'Rotate Counterclockwise':
            blurred_image = rotate_image(blurred_image, 90)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
            blurred_image = blurred_image 
            
        elif event == 'Flip Horizontal':
            blurred_image = flip_image_horizontal(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))

        elif event == 'Flip Vertical':
            blurred_image = flip_image_vertical(blurred_image)
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))   
       
        elif event == '-APPLY-':
         try:
            brightness = int(values['-BRIGHTNESS-'])
            exposure = int(values['-EXPOSURE-'])
            contrast = int(values['-CONTRAST-'])
            highlights = int(values['-HIGHLIGHTS-'])
            saturation = int(values['-SATURATION-'])
            tint = int(values['-TINT-'])
            warmth = int(values['-WARMTH-'])

            blurred_image = apply_adjustments(original_image, brightness=brightness, exposure=exposure,
                                               contrast=contrast, highlights=highlights, tint=tint, saturation=saturation, warmth=warmth)

            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
         except ValueError:
            sg.popup('Invalid input! Please enter a valid number.')                   

        elif event == 'Zoom In':
            blurred_image = zoom_image(blurred_image, 1.2)  
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
        elif event == 'Zoom Out':
            blurred_image = zoom_image(blurred_image, 0.9)  
            window['-BLURRED_IMAGE-'].erase()
            window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
        
        elif event == 'Reset':
    
            window['-AVG_SLIDER-'].update(0)
            window['-GAUSS_SLIDER-'].update(0)
            window['-BRIGHTNESS-'].update(0)
            window['-EXPOSURE-'].update(0)
            window['-CONTRAST-'].update(0)
            window['-HIGHLIGHTS-'].update(0)
            window['-TINT-'].update(0)
            window['-SATURATION-'].update(0)
            window['-WARMTH-'].update(0)
            window['-HIST-'].TKCanvas.delete("all")
            window['Refresh'].click()
            
        elif event == 'Save Image':
         try:
             if blurred_image is not None:
                save_filename = sg.popup_get_file('Save Edited Image As', save_as=True, file_types=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("BMP Files", "*.bmp"), ("All Files", "*.*")))
                if save_filename:
                    file_type = os.path.splitext(save_filename)[1][1:].lower()
                    if file_type == 'png':
                        cv2.imwrite(save_filename, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
                    elif file_type == 'jpg':
                        cv2.imwrite(save_filename, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    elif file_type == 'bmp':
                        cv2.imwrite(save_filename, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
                    else:
                        sg.popup_error(f"Unsupported file type: {file_type}")
                        sg.popup(f'Edited image saved as {save_filename}')
         except Exception as e:
            sg.popup_error(f'Error: {e}')
                
        elif event == 'Load Image':
                load_image()
                window['-IMAGE-'].erase()
                window['-BLURRED_IMAGE-'].erase()
                window['-IMAGE-'].draw_image(data=np_im_to_data(resized_image), location=(0, resized_image.shape[0]))
                window['-BLURRED_IMAGE-'].draw_image(data=np_im_to_data(blurred_image), location=(0, resized_image.shape[0]))
                print(f'{resized_image.shape}')


def main():
    global resized_image
    global original_image
    global blurred_image

    parser = argparse.ArgumentParser(description='A simple image viewer.')
    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')

    target_width = 640
    original_image = image
    target_height = None
    resized_image = resize_image(image, target_width, target_height, True, cv2.INTER_NEAREST)
    print(f'Resized Image Size: {resized_image.shape[1]}x{resized_image.shape[0]}')
    
    # Set to match background theme
    dark_grey_5_color = '#343434'

    # Title Element
    title_element = sg.Text('VisualEnhance', font=('Helvetica', 50), justification='center', text_color='white', background_color=dark_grey_5_color)

    # Logo Image
    logo_path = r'logo.png' 
    logo_element = sg.Image(filename=logo_path, size=(72, 72), background_color=dark_grey_5_color)

    

    display_image(resized_image, args.file, target_width, resized_image.shape[0], True, cv2.INTER_NEAREST,
                  averaging_filter_size=0, gaussian_filter_size=0, stroke_width_range=(1, 5),
                  stroke_length_range=(5, 20), edge_threshold=30, title_element=title_element, logo_element=logo_element)



if __name__ == '__main__':
    main()