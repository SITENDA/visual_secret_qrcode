import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import qrcode
import numpy as np

def generate_qr_code(data, size=10):
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=size,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    return img

def create_color_shares(input_image):
    gray_image = input_image.convert("L")
    arr = np.array(gray_image)
    height, width = arr.shape

    share1 = np.zeros((height, width, 3), dtype=np.uint8)
    share2 = np.zeros((height, width, 3), dtype=np.uint8)
    share3 = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if arr[i, j] < 128:
                share1[i, j] = [255, 255, 0]  # Yellow in share1
                share2[i, j] = [255, 105, 180]  # Pink in share2
                share3[i, j] = [255, 255, 0]  # Yellow in share3
            else:
                share1[i, j] = [255, 255, 255]  # White in share1
                share2[i, j] = [255, 105, 180]  # Pink in share2
                share3[i, j] = [255, 255, 0]  # Yellow in share3

    share1_img = Image.fromarray(share1, 'RGB')
    share2_img = Image.fromarray(share2, 'RGB')
    share3_img = Image.fromarray(share3, 'RGB')

    return share1_img, share2_img, share3_img

def overlay_shares(share1, share2, share3):
    # Proper overlay of three images requires a different approach:
    overlay = Image.blend(Image.blend(share1, share2, alpha=0.5), share3, alpha=0.5)
    return overlay

def display_images(images):
    for i, image in enumerate(images):
        window = tk.Toplevel(root)
        window.title(f"Image {i+1}")
        photo = ImageTk.PhotoImage(image=image)
        label = tk.Label(window, image=photo)
        label.image = photo
        label.pack()

def generate_and_display():
    user_input = simpledialog.askstring("Input", "Enter the data for QR Code:", parent=root)
    if user_input:
        qr_img = generate_qr_code(user_input)
        share1_img, share2_img, share3_img = create_color_shares(qr_img)
        result_img = overlay_shares(share1_img, share2_img, share3_img)
        display_images([qr_img, share1_img, share2_img, share3_img, result_img])

# Create the main window
root = tk.Tk()
root.title("QR Code Secret Sharing GUI")

# Create a button to generate QR and shares
generate_button = tk.Button(root, text="Generate QR Code", command=generate_and_display)
generate_button.pack(pady=20)

# Start the GUI
root.mainloop()
