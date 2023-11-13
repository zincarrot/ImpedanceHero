import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create initial parameter values
initial_a = 1.0
initial_b = 1.0
initial_c = 0.0

# Generate x values
x = np.linspace(-10, 10, 100)

# Create the main window
window = tk.Tk()
window.title("Math Function Explorer")
window.geometry("1280x720")
window.config(background='white')

# Create the frames: top, center, bottom
top_frame = tk.Frame(window)
top_frame.grid(row=0, column=0, sticky='nsew')
center_frame = tk.Frame(window)
center_frame.grid(row=1, column=0, sticky='nsew')
bottom_frame = tk.Frame(window)
bottom_frame.grid(row=2, column=0, sticky='nsew')

# Create sub-frames for the center frame
center_left_frame = tk.Frame(center_frame)
center_left_frame.grid(row=0, column=0, sticky='nsew')
center_right_frame = tk.Frame(center_frame)
center_right_frame.grid(row=0, column=1, sticky='nsew')

# Create the figure
fig = Figure(figsize=(6, 4), dpi=100)

# Create the plot axes
ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Create the sliders frame inside the left frame
sliders_frame = tk.Frame(center_left_frame)
sliders_frame.grid(row=0, column=0, sticky='nsew')

# Create the plot frame inside the right frame
plot_frame = tk.Frame(center_right_frame)
plot_frame.grid(row=0, column=0, sticky='nsew')

# Create the path frame inside the top frame
path_frame = tk.Frame(top_frame)
path_frame.grid(row=0, column=0, sticky='nsew')

# Create the buttons frame inside the bottom frame
buttons_frame = tk.Frame(bottom_frame)
buttons_frame.grid(row=0, column=0, sticky='nsew')

# Create the path widget
path = tk.Label(path_frame, text='Path: python/expression_explore.py')
path.pack()

# Create the load button
button_load = tk.Button(path_frame, text='Load')
button_load.pack(side=tk.RIGHT)

# Create the save button
button_save = tk.Button(path_frame, text='Save')
button_save.pack(side=tk.RIGHT)

# Create the buttons
button_quit = tk.Button(buttons_frame, text='Quit', command=window.quit)
button_quit.pack(side=tk.RIGHT)
button_add_slider = tk.Button(buttons_frame, text='Add Parameter')
button_add_slider.pack(side=tk.RIGHT)
button_delete_slider = tk.Button(buttons_frame, text='Delete Parameter')
button_delete_slider.pack(side=tk.RIGHT)

# Create the canvas
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

# Create the entry widget for the math expression
entry_expression = tk.Entry(sliders_frame, width=50, font=('Arial', 14))
entry_expression.insert(0, 'a * np.sin(b * x) + c')
entry_expression.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')
entry_expression.pack()

# Variable to store the expression
expression = entry_expression.get()

# Function to evaluate the math expression
def evaluate_expression(x, a, b, c):
    return eval(expression)

# Function to update the plot when slider values or expression change
def update():
    global expression
    expression = entry_expression.get()
    a = slider_a.get()
    b = slider_b.get()
    c = slider_c.get()
    line.set_ydata(evaluate_expression(x, a, b, c))
    ax.relim()  # Recalculate the plot limits
    ax.autoscale()  # Autoscale the plot
    canvas.draw()  # Redraw the plot

# Function to save the expression
def save_expression():
    global expression
    with open("expression.txt", "w") as file:
        file.write(expression)

# Custom Slider class
class CustomSlider(tk.Scale):
    def __init__(self, master, label, valmin, valmax, valinit, command):
        super().__init__(master, from_=valmin, to=valmax, resolution=0.1, orient=tk.HORIZONTAL, label=label)
        self.set(valinit)
        self.pack()
        self.config(command=command)

# Create the sliders
slider_a = CustomSlider(sliders_frame, 'a', valmin=0.1, valmax=10, valinit=initial_a, command=lambda x: update())
slider_b = CustomSlider(sliders_frame, 'b', valmin=0.1, valmax=10, valinit=initial_b, command=lambda x: update())
slider_c = CustomSlider(sliders_frame, 'c', valmin=-5, valmax=5, valinit=initial_c, command=lambda x: update())

# Plot initial function
line, = ax.plot(x, np.zeros_like(x))

# Create a button to save the expression
button_save = tk.Button(sliders_frame, text='Save', command=save_expression)
button_save.pack()

# Display the plot
window.mainloop()
