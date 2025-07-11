#!/usr/bin/env python3
"""
MySky - A simple desktop application
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os


class MySkyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MySky Application")
        self.root.geometry("400x300")
        
        # Center the window
        self.center_window()
        
        # Create main frame
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Title label
        title_label = tk.Label(main_frame, text="Welcome to MySky!", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Description
        desc_label = tk.Label(main_frame, 
                             text="A simple desktop application for Windows",
                             font=("Arial", 10))
        desc_label.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        hello_btn = tk.Button(button_frame, text="Say Hello", 
                             command=self.say_hello, width=15)
        hello_btn.pack(pady=5)
        
        about_btn = tk.Button(button_frame, text="About", 
                             command=self.show_about, width=15)
        about_btn.pack(pady=5)
        
        exit_btn = tk.Button(button_frame, text="Exit", 
                            command=root.quit, width=15)
        exit_btn.pack(pady=5)
        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def say_hello(self):
        """Display a hello message"""
        messagebox.showinfo("Hello!", "Hello from MySky Application!")
        
    def show_about(self):
        """Show about dialog"""
        about_text = """MySky v1.0.0
        
A simple desktop application
Created for Windows

© 2024 MySky Project"""
        messagebox.showinfo("About MySky", about_text)


def main():
    root = tk.Tk()
    app = MySkyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()