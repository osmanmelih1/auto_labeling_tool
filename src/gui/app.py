"""
Module: app.py
Description: Main Desktop GUI for the Auto Labeling Tool.
             Connects GUI buttons to core scripts using threading and subprocess
             to prevent freezing and capture terminal output.
"""

import customtkinter as ctk
import threading
import subprocess

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AutoLabelingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window settings
        self.title("Auto Labeling Tool - Core GUI")
        self.geometry("1000x700")

        # Configure grid layout (1x2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Auto Labeling", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # --- Buttons ---
        self.btn_step1 = ctk.CTkButton(
            self.sidebar_frame, 
            text="1. Deduplication",
            command=lambda: self.run_script_in_thread("src/core/step1_deduplication.py")
        )
        self.btn_step1.grid(row=1, column=0, padx=20, pady=10)

        self.btn_step2 = ctk.CTkButton(
            self.sidebar_frame, 
            text="2. Embedding (VDB)",
            command=lambda: self.run_script_in_thread("src/core/step2_embedding.py")
        )
        self.btn_step2.grid(row=2, column=0, padx=20, pady=10)
        
        self.btn_step3a = ctk.CTkButton(
            self.sidebar_frame, 
            text="3a. Text Prompting",
            command=lambda: self.run_script_in_thread("src/core/step3a_text_prompting.py")
        )
        self.btn_step3a.grid(row=3, column=0, padx=20, pady=10)
        
        self.btn_step3b = ctk.CTkButton(
            self.sidebar_frame, 
            text="3b. Manual Seeding",
            command=lambda: self.run_script_in_thread("src/core/step3b_manual_seeding.py")
        )
        self.btn_step3b.grid(row=4, column=0, padx=20, pady=10)

        self.btn_step4 = ctk.CTkButton(
            self.sidebar_frame, 
            text="4. Propagation",
            command=lambda: self.run_script_in_thread("src/core/step4_propagation.py")
        )
        self.btn_step4.grid(row=5, column=0, padx=20, pady=10)

        # Store buttons in a list for easy state management
        self.all_buttons = [
            self.btn_step1, self.btn_step2, 
            self.btn_step3a, self.btn_step3b, self.btn_step4
        ]

        # --- Main Workspace ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.welcome_label = ctk.CTkLabel(
            self.main_frame, 
            text="Console Output", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.welcome_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        # Textbox for console logs
        self.log_textbox = ctk.CTkTextbox(
            self.main_frame, 
            wrap="word", 
            font=ctk.CTkFont(family="Consolas", size=13)
        )
        self.log_textbox.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.log_textbox.insert("end", "[*] System ready. Waiting for commands...\n")

    def append_log(self, text):
        """Appends text to the log textbox safely."""
        self.log_textbox.insert("end", text)
        self.log_textbox.see("end")  # Auto-scroll to bottom

    def set_buttons_state(self, state):
        """Enables or disables all sidebar buttons."""
        for btn in self.all_buttons:
            btn.configure(state=state)

    def run_script_in_thread(self, script_path):
        """Starts a new thread to run the external script without freezing the GUI."""
        self.append_log(f"\n[{'='*40}]\n")
        self.append_log(f"[*] Executing: {script_path}\n")
        
        # Disable all buttons while running to prevent conflicts
        self.set_buttons_state("disabled")
        
        thread = threading.Thread(target=self._execute_script, args=(script_path,))
        thread.start()

    def _execute_script(self, script_path):
        """Runs the script using subprocess and captures live terminal output."""
        try:
            process = subprocess.Popen(
                ["uv", "run", "python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line as it generates
            for line in process.stdout:
                self.append_log(line)
                
            process.wait()
            
            if process.returncode == 0:
                self.append_log(f"\n[+] Process finished successfully.\n")
            else:
                self.append_log(f"\n[!] Process ended with error code: {process.returncode}\n")
                
        except Exception as e:
            self.append_log(f"\n[!] Critical Error: {str(e)}\n")
            
        finally:
            # Re-enable all buttons once finished
            self.set_buttons_state("normal")


if __name__ == "__main__":
    app = AutoLabelingApp()
    app.mainloop()