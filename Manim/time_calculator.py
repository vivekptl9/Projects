import tkinter as tk
from tkinter import messagebox

class TimeTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Course Time Tracker")
        
        self.total_seconds = 0
        
        # Labels and Entries
        tk.Label(root, text="Hours:").grid(row=0, column=0)
        tk.Label(root, text="Minutes:").grid(row=1, column=0)
        tk.Label(root, text="Seconds:").grid(row=2, column=0)
        
        self.hours_entry = tk.Entry(root)
        self.minutes_entry = tk.Entry(root)
        self.seconds_entry = tk.Entry(root)
        
        self.hours_entry.grid(row=0, column=1)
        self.minutes_entry.grid(row=1, column=1)
        self.seconds_entry.grid(row=2, column=1)
        
        # Buttons
        self.add_button = tk.Button(root, text="Add Module Time", command=self.add_time)
        self.add_button.grid(row=3, column=0, columnspan=2)
        
        self.show_button = tk.Button(root, text="Show Total Time", command=self.show_total_time)
        self.show_button.grid(row=4, column=0, columnspan=2)
        
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_time)
        self.reset_button.grid(row=5, column=0, columnspan=2)
        
        # Bind Enter key to the add_time function when focus is on entries
        self.root.bind('<Return>', self.handle_enter)
        
    def add_time(self):
        try:
            hours = int(self.hours_entry.get() or 0)
            minutes = int(self.minutes_entry.get() or 0)
            seconds = int(self.seconds_entry.get() or 0)
            
            self.total_seconds += hours * 3600 + minutes * 60 + seconds
            
            self.hours_entry.delete(0, tk.END)
            self.minutes_entry.delete(0, tk.END)
            self.seconds_entry.delete(0, tk.END)
            
            messagebox.showinfo("Success", "Time added successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")
        
    def show_total_time(self):
        hours = self.total_seconds // 3600
        minutes = (self.total_seconds % 3600) // 60
        seconds = self.total_seconds % 60
        
        messagebox.showinfo("Total Time", f"Total Course Duration: {hours}h {minutes}m {seconds}s")
    
    def reset_time(self):
        self.total_seconds = 0
        messagebox.showinfo("Reset", "Total time has been reset!")
    
    def handle_enter(self, event):
        focused_widget = self.root.focus_get()
        if focused_widget in [self.hours_entry, self.minutes_entry, self.seconds_entry, self.add_button]:
            self.add_time()
        elif focused_widget == self.show_button:
            self.show_total_time()
        elif focused_widget == self.reset_button:
            self.reset_time()

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeTracker(root)
    root.mainloop()
