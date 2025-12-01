
import json
import sys
import threading
import os
import shutil
import subprocess


from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit,
    QLabel, QComboBox, QPushButton, QFileDialog, QMessageBox, QCheckBox, QRadioButton, QTextEdit, QSplitter
)

class CalibrationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resense Calibration Analysis System")
        self.setGeometry(100, 100, 800, 600)
        
        
        with open("machines_config.json", "r") as f:
            self.config_data = json.load(f)


        main_layout = QHBoxLayout()
        splitter = QSplitter()


        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)


        # Section 1 Machine Selection
        machine_group = QGroupBox("Machine Selection")
        machine_layout = QHBoxLayout()
        self.machine_label = QLabel("Calibration Machine")
        self.machine_combo = QComboBox()
        self.machine_combo.addItems(["CM1 - CM1_Hex21", "CM2 - CM2_Hex8_10_12",  "CM3 - CM3_Hex8_10_12", "CM4 - CM4_Hex21", "CM5 - CM5_Hex32"])
        
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.machine_config)
        machine_layout.addWidget(self.machine_label)
        machine_layout.addWidget(self.machine_combo)
        machine_layout.addWidget(self.refresh_btn)
        machine_group.setLayout(machine_layout)
        
        # SECTION 2 Data File Selection
        data_group = QGroupBox("Data File Selection")
        data_layout = QHBoxLayout()
        self.data_label = QLabel("CSV Data File")
        self.file_input = QLineEdit()
        self.browse_btn = QPushButton("Browse", self)
        self.browse_btn.clicked.connect(self.load_or_create_local_config)
        self.auto_find_btn = QPushButton("Auto-Find")
        self.sync_btn = QPushButton("Sync OneDrive", self)
        self.sync_btn.clicked.connect(self.sync_onedrive)
        data_layout.addWidget(self.data_label)
        data_layout.addWidget(self.file_input)
        data_layout.addWidget(self.browse_btn)
        data_layout.addWidget(self.auto_find_btn)
        data_layout.addWidget(self.sync_btn)
        data_group.setLayout(data_layout)


        # SECTION3 ANALYSIS 
        script_group = QGroupBox("Analysis Script")
        script_layout = QHBoxLayout()
        script_label = QLabel("Script to Run")
        self.script_map = {
            "Raw Step Analysis": "Raw_Step.py",
            "General Analysis": "Analysis.py",
            "Step Analysis": "analysis_step.py",
            "Matrix Calculation": "calculateMatrix.py"
        }

        self.script_info = {"Raw Step Analysis": "Processes raw calibration data with step detection and measurement phase analysis. Main script for calibration processing.", 
                "General Analysis": "General purpose analysis script for calibration data.", "Step Analysis": "Specialized script for analyzing calibration steps and patterns.", 
                "Matrix Calculation": "Calculates calibration matrices from processed data."}
        self.script_combo = QComboBox()
        self.script_combo.addItems(list(self.script_info.keys()))

        info_btn = QPushButton("Info")
        info_btn.clicked.connect(self.show_info)
        script_layout.addWidget(script_label)
        script_layout.addWidget(self.script_combo)
        script_layout.addWidget(info_btn)
        script_group.setLayout(script_layout)
        
        
    #Section 4 Analysis Options

        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout()
        options_layout.addWidget(QCheckBox("Show Plots during Analysis"))
        options_layout.addWidget(QCheckBox("Fx Analysis"))
        options_layout.addWidget(QCheckBox("Fy Analysis"))
        options_layout.addWidget(QCheckBox("Fz Analysis"))
        options_layout.addWidget(QCheckBox("Mx Analysis"))
        options_layout.addWidget(QCheckBox("My Analysis"))
        options_layout.addWidget(QCheckBox("Mz Analysis"))
        options_layout.addWidget(QCheckBox("FT Analysis"))
        options_group.setLayout(options_layout)


        # Section 5 Analysis Type
        type_group = QGroupBox("Analysis Type")
        type_layout = QVBoxLayout()
        type_layout.addWidget(QRadioButton("Regular Step Analysis"))
        type_layout.addWidget(QRadioButton("3-Step Analysis"))
        type_layout.addWidget(QRadioButton("Test Phase Extraction"))
        type_layout.addWidget(QRadioButton("Measurement Extraction"))
        type_layout.addWidget(QRadioButton("3-Step Evaluation"))
        type_group.setLayout(type_layout)


        # Section 6 Action Buttons
        action_layout = QHBoxLayout()
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self.run_analysis)
        stop_btn = QPushButton("Stop")
        clear_btn = QPushButton("Clear Output")
        clear_btn.clicked.connect(self.clear_output)
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        action_layout.addWidget(run_btn)
        action_layout.addWidget(stop_btn)
        action_layout.addWidget(clear_btn)
        action_layout.addWidget(exit_btn)
        
       

        # Output Console
        self.output_console = QTextEdit()
        self.output_console.setPlaceholderText("Output Console")

        # Add all sections to left layout
        left_layout.addWidget(machine_group)
        left_layout.addWidget(data_group)
        left_layout.addWidget(script_group)
        left_layout.addWidget(options_group)
        left_layout.addWidget(type_group)
        left_layout.addLayout(action_layout)
        left_layout.addWidget(self.output_console)

        left_panel.setLayout(left_layout)


        # ---------------- RIGHT PANEL ----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Local Configuration Editor"))
        right_panel.setLayout(right_layout)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def machine_config(self):
        machine_id = self.machine_combo.currentText().split(" - ")[0]
        
        try:
            machine_conf = self.config_data["machines"][machine_id]["default_config"]
            self.machine_defaults=machine_conf
            self.log_output(f"Loaded config for {machine_id}")
            
        except Exception as e:
            self.log_output(f"Error loading machine config: {e}")


    def refresh_machines(self):
        new_machines = ["CM1 - CM1_Inst1", "CM2 - CM2_Inst2", "CM3 - CM3_Inst3"]
        self.machine_combo.clear()
        self.machine_combo.addItems(new_machines)

    def load_or_create_local_config(self, csv_file):
        directory = os.path.dirname(csv_file)
        self.local_config_path = os.path.join(directory, "config.json")

        # If local config exists, load it
        if os.path.exists(self.local_config_path):
            with open(self.local_config_path, "r") as f:
                self.local_config = json.load(f)
            self.log_output("Loaded existing local config.json")
            return

        # Otherwise create from machine defaults
        if hasattr(self, "machine_defaults"):
            self.local_config = self.machine_defaults
            with open(self.local_config_path, "w") as f:
                json.dump(self.local_config, f, indent=4)
            self.log_output("Created new local config.json from machine defaults")
            

    def sync_onedrive(self):
        file_path = self.file_input.text()
        if not file_path or not os.path.isfile(file_path):
            QMessageBox.warning(self, "Error", "Please select a valid file first.")
            return

        # Detect OneDrive folder
        home_dir = os.path.expanduser("~")
        onedrive_path = os.path.join(home_dir, "OneDrive")

        if not os.path.exists(onedrive_path):
            QMessageBox.critical(self, "Error", "OneDrive folder not found on this system.")
            return

        # Copy file to OneDrive
        try:
            shutil.copy(file_path, onedrive_path)
            QMessageBox.information(self, "Success", f"File synced to OneDrive:\n{onedrive_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to sync file: {str(e)}")
            
            
    
    def show_info(self):
        selected_script = self.script_combo.currentText()
        description = self.script_info.get(selected_script, "No description available.")
        QMessageBox.information(self, "Script Info", description)

    def run_analysis(self):
        # set env vars same as tkinter version
        os.environ["CALIBRATION_MACHINE_ID"] = self.machine_combo.currentText().split(" - ")[0]
        os.environ["CALIBRATION_FILE_PATH"] = self.file_input.text()
        os.environ["CALIBRATION_FX_ANALYSIS"] = str(self.findChild(QCheckBox, "Fx Analysis").isChecked())
        os.environ["CALIBRATION_FY_ANALYSIS"] = str(self.findChild(QCheckBox, "Fy Analysis").isChecked())
        os.environ["CALIBRATION_FZ_ANALYSIS"] = str(self.findChild(QCheckBox, "Fz Analysis").isChecked())

        os.environ["CALIBRATION_MX_ANALYSIS"] = str(self.findChild(QCheckBox, "Mx Analysis").isChecked())
        os.environ["CALIBRATION_MY_ANALYSIS"] = str(self.findChild(QCheckBox, "My Analysis").isChecked())
        os.environ["CALIBRATION_MZ_ANALYSIS"] = str(self.findChild(QCheckBox, "Mz Analysis").isChecked())

        os.environ["CALIBRATION_FT_ANALYSIS"] = str(self.findChild(QCheckBox, "FT Analysis").isChecked())
        selected_script = self.script_combo.currentText()
        self.log_output(f"Starting analysis: {selected_script}")
        thread = threading.Thread(target=self.execute_script, args=(selected_script,))
        thread.daemon = True
        thread.start()


    def execute_script(self, script_name):
        script_path = self.script_map.get(script_name)
        if not script_path:
            self.log_output("Error: Script not found.")
            return

        if not self.file_input.text():
            self.log_output("Error: No CSV file selected.")
            return

        # Run script with CSV file and machine config as arguments

        cmd = [
            sys.executable,
            script_path,
            self.file_input.text(),
            self.local_config_path
        ]        
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in iter(process.stdout.readline, ""):
            self.log_output(line.rstrip())

        process.wait()
        if process.returncode != 0:
            error = process.stderr.read()
            self.log_output(f"Error: {error}")

    def clear_output(self):
        self.output_console.clear()

    def log_output(self, message):
        self.output_console.append(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationUI()
    window.show()
    sys.exit(app.exec_())
