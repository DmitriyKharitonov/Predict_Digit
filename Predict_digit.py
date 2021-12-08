import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
from NeuralNetwork import neuralNetwork
import numpy
import scipy.special
import PIL.ImageOps

def predict_digit(img):
    img.save('Photo.jpg')
    img = img.resize((28,28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = numpy.array(img)
    img = 255.0 - img.reshape(1,784)
    img = (img / 255.0 * 0.99) + 0.01
    new_img = numpy.where(img<20,0,img)

    input_nodes = 784
    hidden_nodes = 100 
    output_nodes = 10
    learningrate = 0.005

    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learningrate)
    outputs = n.query(img)
    return numpy.argmax(outputs)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.x = self.y = 0
        
        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Распознать", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Очистить", command = self.clear_all)
        
        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky= 'W' )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        
    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND) # получаем координату холста
        new_rect = (rect[0]+50,rect[1]+50,rect[2]+100,rect[3]+100)
        im = ImageGrab.grab(new_rect)
        
        digit= predict_digit(im)
        self.label.configure(text= str(digit))
        
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
tk.mainloop()    