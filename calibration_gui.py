#!/usr/bin/env python3

import os
import sys
import json


from functools import partial
from PyQt5.QtCore import Qt, QProcess, QStandardPaths
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QCheckBox, QRadioButton,
    QTextEdit, QGroupBox, QPlainTextEdit, QSplitter
)

def get_resource_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def resource_dir():

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


class CalibrationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resense Calibration Analysis System")
        self.resize(1200, 800)

        self.base_dir = resource_dir()
        self.machines_config_path = os.path.join(self.base_dir, 'machines_config.json')
        self.config_data = {}
        self.machine_defaults = {}
        self.local_config = {}
        self.local_config_path = None
        self.process = None

        # Load machine config file
        if not os.path.exists(self.machines_config_path):
            QMessageBox.critical(self, "Missing file", f"machines_config.json not found at {self.machines_config_path}")
            self.config_data = {"machines": {}, "scripts": {}}
        else:
            with open(self.machines_config_path, 'r') as f:
                self.config_data = json.load(f)

        self._build_ui()
        self._populate_machines()
        self._populate_scripts()

    # ---------------- UI BUILD ----------------
    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # Machine selection
        machine_box = QGroupBox("Machine Selection")
        m_layout = QHBoxLayout()
        m_layout.setContentsMargins(6, 6, 6, 6)
        m_layout.setSpacing(6)
        m_layout.addWidget(QLabel("Calibration Machine:"))
        self.machine_combo = QComboBox()
        self.machine_combo.currentIndexChanged.connect(self.on_machine_selected)
        m_layout.addWidget(self.machine_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._populate_machines)
        m_layout.addWidget(self.refresh_btn)
        machine_box.setLayout(m_layout)
        left_layout.addWidget(machine_box)

        # Data file selection
        file_box = QGroupBox("Data File Selection")
        f_layout = QHBoxLayout()
        f_layout.setContentsMargins(6, 6, 6, 6)
        f_layout.setSpacing(6)
        f_layout.addWidget(QLabel("CSV Data File:"))
        self.file_input = QLineEdit()
        f_layout.addWidget(self.file_input)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse)
        f_layout.addWidget(self.browse_btn)
        self.auto_find_btn = QPushButton("Auto-Find")
        self.auto_find_btn.clicked.connect(self.auto_find_files)
        f_layout.addWidget(self.auto_find_btn)
        self.sync_btn = QPushButton("Sync OneDrive")
        self.sync_btn.clicked.connect(self.sync_onedrive_file)
        f_layout.addWidget(self.sync_btn)
        file_box.setLayout(f_layout)
        left_layout.addWidget(file_box)

        # Script selection
        script_box = QGroupBox("Analysis Script")
        s_layout = QHBoxLayout()
        s_layout.setContentsMargins(6, 6, 6, 6)
        s_layout.setSpacing(6)
        s_layout.addWidget(QLabel("Script to Run:"))
        self.script_combo = QComboBox()
        s_layout.addWidget(self.script_combo)
        self.script_info_btn = QPushButton("Info")
        self.script_info_btn.clicked.connect(self.show_script_info)
        s_layout.addWidget(self.script_info_btn)
        script_box.setLayout(s_layout)
        left_layout.addWidget(script_box)

        # Analysis options 
        options_box = QGroupBox("Analysis Options")
        o_layout = QHBoxLayout()
        self.checkboxes = {}
        labels = ["Show Plots during Analysis", "Fx Analysis", "Fy Analysis", "Fz Analysis",
                  "Mx Analysis", "My Analysis", "Mz Analysis", "FT Analysis"]
        for label in labels:
            cb = QCheckBox(label)
            self.checkboxes[label] = cb
            o_layout.addWidget(cb)
        options_box.setLayout(o_layout)
        left_layout.addWidget(options_box)


        # Analysis type

        # Analysis type (checkboxes instead of radio buttons)
        type_box = QGroupBox("Analysis Type")
        t_layout = QHBoxLayout()
        self.analysis_checkboxes = {}
        analysis_labels = [
            "Regular Step Analysis",
            "3-Step Analysis",
            "Test Phase Extraction",
            "Measurement Extraction",
            "3-Step Evaluation"
        ]
        for label in analysis_labels:
            cb = QCheckBox(label)
            self.analysis_checkboxes[label] = cb
            t_layout.addWidget(cb)
        type_box.setLayout(t_layout)
        left_layout.addWidget(type_box)



        # Action buttons
        action_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear Output")
        self.clear_btn.clicked.connect(self.clear_output)
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        for w in (self.run_btn, self.stop_btn, self.clear_btn, self.exit_btn):
            action_layout.addWidget(w)
        left_layout.addLayout(action_layout)

        # Output console
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setPlaceholderText("Output Console")
        left_layout.addWidget(self.output_console, 1)

        # RIGHT PANEL (config editor)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        right_layout.addWidget(QLabel("Local Configuration Editor"))
        self.config_editor = QPlainTextEdit()
        right_layout.addWidget(self.config_editor, 1)

        cfg_btn_layout = QHBoxLayout()
        self.save_cfg_btn = QPushButton("Save Config")
        self.save_cfg_btn.clicked.connect(self.save_local_config)
        self.reload_cfg_btn = QPushButton("Reload Config")
        self.reload_cfg_btn.clicked.connect(self.reload_local_config)
        self.reset_defaults_btn = QPushButton("Reset to Machine Defaults")
        self.reset_defaults_btn.clicked.connect(self.reset_to_machine_defaults)
        cfg_btn_layout.addWidget(self.save_cfg_btn)
        cfg_btn_layout.addWidget(self.reload_cfg_btn)
        cfg_btn_layout.addWidget(self.reset_defaults_btn)
        right_layout.addLayout(cfg_btn_layout)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([700, 500])
        main_layout.addWidget(splitter)

    # ---------------- Populate helpers ----------------
    def _populate_machines(self):
        self.machine_combo.clear()
        # Prefer gui_machines if present
        machines = self.config_data.get('machines', {})
        items = []
        for mid, info in machines.items():
            name = info.get('name', mid)
            items.append(f"{mid} - {name}")
        if not items:
            items = ["CM1 - CM1_Hex21"]
        self.machine_combo.addItems(items)
        # trigger initial load if possible
        if items:
            self.on_machine_selected(0)

    def _populate_scripts(self):
        self.script_combo.clear()
        scripts = self.config_data.get('scripts', {})
        # If no scripts in config, use sensible defaults
        if not scripts:
            default_map = {
                'Raw Step Analysis': 'Raw_Step.py',
                'General Analysis': 'Analysis.py',
                'Step Analysis': 'analysis_step.py',
                'Matrix Calculation': 'calculateMatrix.py'
            }
            for name in default_map:
                self.script_combo.addItem(name, default_map[name])
            self.script_descriptions = {}
        else:
            self.script_descriptions = {}
            for script_file, meta in scripts.items():
                label = meta.get('name', script_file)
                desc = meta.get('description', '')
                self.script_combo.addItem(label, script_file)
                self.script_descriptions[label] = desc

    # ---------------- Machine selection ----------------
    def on_machine_selected(self, index):
        current = self.machine_combo.currentText()
        if not current:
            return
        machine_id = current.split(' - ')[0]
        machines = self.config_data.get('machines') or {}
        self.machine_defaults = machines.get(machine_id, {}).get('default_config', {})


    # ---------------- File selection & local config ----------------
    def browse(self):
      home = os.path.expanduser("~")
      onedrive_wika = os.path.join(home, "OneDriveWIKA")

      if not os.path.exists(onedrive_wika):
          onedrive_wika = os.path.join(home, "OneDrive - WIKA")

      if not os.path.exists(onedrive_wika):
          QMessageBox.critical(self, "OneDrive Error",
              "OneDrive-WIKA folder not found on this computer.")
          return

      file_path, _ = QFileDialog.getOpenFileName(
          self,
          "Select Calibration CSV File",
          onedrive_wika, 
          "CSV Files (*.csv)" 
      )

      if file_path:
          self.file_input.setText(file_path)
          self.load_or_create_local_config(file_path)

    def load_or_create_local_config(self, csv_file):
        if not isinstance(csv_file, str):
            self.log_output("Invalid CSV file path")
            return

        data_folder = os.path.dirname(csv_file)
        local_cfg = os.path.join(data_folder, 'machine_config.json')
        self.local_config_path = local_cfg

        # If exists -> load
        if os.path.exists(local_cfg):
            with open(local_cfg, 'r') as f:
                self.local_config = json.load(f)
            self.config_editor.setPlainText(json.dumps(self.local_config, indent=4))
            return

        # Otherwise create using machine defaults
        if not self.machine_defaults:
            self.local_config = {}
            self.config_editor.setPlainText('{}')
            return

        self.local_config = self.machine_defaults.copy()
        try:
            with open(local_cfg, 'w') as f:
                json.dump(self.local_config, f, indent=4)
            self.config_editor.setPlainText(json.dumps(self.local_config, indent=4))
            self.log_output(f"Created new local config from machine defaults: {local_cfg}")
        except Exception as e:
            self.log_output(f"Failed to create local config: {e}")

    def save_local_config(self):
        if not self.local_config_path:
            QMessageBox.warning(self, "No file", "No data file selected or local config path not set")
            return
        try:
            # Validate JSON in editor
            text = self.config_editor.toPlainText()
            parsed = json.loads(text)
            with open(self.local_config_path, 'w') as f:
                json.dump(parsed, f, indent=4)
            self.local_config = parsed
            self.log_output(f"Saved local config to {self.local_config_path}")
        except json.JSONDecodeError as jde:
            QMessageBox.critical(self, "JSON Error", f"Config editor contains invalid JSON:\n{jde}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def reload_local_config(self):
        if not self.local_config_path or not os.path.exists(self.local_config_path):
            QMessageBox.warning(self, "Missing", "Local config.json not found to reload")
            return
        try:
            with open(self.local_config_path, 'r') as f:
                self.local_config = json.load(f)
            self.config_editor.setPlainText(json.dumps(self.local_config, indent=4))
            self.log_output("Reloaded local config from disk")
        except Exception as e:
            QMessageBox.critical(self, "Reload Error", str(e))

    def reset_to_machine_defaults(self):
        if not self.machine_defaults:
            QMessageBox.warning(self, "No machine", "Select a machine first")
            return
        self.local_config = self.machine_defaults.copy()
        self.config_editor.setPlainText(json.dumps(self.local_config, indent=4))
        # Save immediately to local_config_path if available
        if self.local_config_path:
            try:
                with open(self.local_config_path, 'w') as f:
                    json.dump(self.local_config, f, indent=4)
                self.log_output("Local config reset to machine defaults and saved")
            except Exception as e:
                self.log_output(f"Failed to save reset config: {e}")

    # ---------------- Script Info ----------------
    def show_script_info(self):
        label = self.script_combo.currentText()
        desc = self.script_descriptions.get(label)
        if desc:
            QMessageBox.information(self, "Script Info", desc)
        else:
            QMessageBox.information(self, "Script Info", "No description available for this script")

    # ---------------- OneDrive helpers ----------------
    def find_onedrive_path(self):
        # Look for common OneDrive locations
        home = os.path.expanduser('~')
        candidates = [
            os.path.join(home, 'OneDrive'),
            os.path.join(home, 'OneDrive - WIKA'),
            os.path.join(home, 'OneDrive - Company'),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def sync_onedrive_file(self):
        file_path = self.file_input.text().strip()
        if not file_path:
            QMessageBox.warning(self, "No file", "Please select a data CSV first")
            return
        if 'OneDrive' not in file_path and not file_path.startswith(self.find_onedrive_path() or ''):
            QMessageBox.information(self, "Not OneDrive", "Selected file is not in OneDrive folder; nothing to sync")
            return
        od = self.find_onedrive_path()
        if not od:
            QMessageBox.critical(self, "OneDrive not found", "OneDrive folder not detected on this machine")
            return
        # Attempt to open folder to trigger sync (best-effort on Windows)
        try:
            if sys.platform.startswith('win'):
                os.startfile(os.path.dirname(file_path))
            else:
                # on linux/mac open with default file manager
                import subprocess as sp
                sp.Popen(['xdg-open', os.path.dirname(file_path)])
            QMessageBox.information(self, "Sync", "Opened the folder containing the file to trigger OneDrive sync (if applicable)")
        except Exception as e:
            QMessageBox.warning(self, "Sync error", f"Could not open folder: {e}")

    def auto_find_files(self):
        # Search user's Documents and OneDrive for recent CSVs (best-effort)
        search_paths = []
        od = self.find_onedrive_path()
        if od:
            search_paths.append(od)
        docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        if docs:
            search_paths.append(docs)
        found = []
        for root in search_paths:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith('.csv') and ('cal' in fn.lower() or 'miec' in fn.lower() or 'calibration' in fn.lower()):
                        found.append(os.path.join(dirpath, fn))
                # limit search to avoid long waits
                if len(found) > 200:
                    break
        if not found:
            QMessageBox.information(self, "Auto-Find", "No likely calibration CSV files found")
            return
        # show a simple selection dialog using QFileDialog (directory selection + list) -> use native dialog: pick first
        selected = QFileDialog.getOpenFileName(self, "Select calibration CSV", os.path.dirname(found[0]), "CSV Files (*.csv)")
        if selected and selected[0]:
            self.file_input.setText(selected[0])
            self.load_or_create_local_config(selected[0])

    # ---------------- Run/Stop Script ----------------
    def run_analysis(self):
        script_label = self.script_combo.currentText()
        script_file = self.script_combo.currentData()
        # If combo was filled from script_descriptions, currentData is file name; otherwise use label mapping
        if not script_file:
            # try to map from label text using default filenames
            mapping = {
                'Raw Step Analysis': 'Raw_Step.py',
                'General Analysis': 'Analysis.py',
                'Step Analysis': 'analysis_step.py',
                'Matrix Calculation': 'calculateMatrix.py'
            }
            script_file = mapping.get(script_label, script_label)

        script_path = os.path.join(self.base_dir, script_file)
        if not os.path.exists(script_path):
            # try looking relative to working dir
            script_path = os.path.abspath(script_file)
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Script Missing", f"Cannot find script: {script_file}\nLooked at: {script_path}")
            return

        csv_file = self.file_input.text().strip()
        if not csv_file or not os.path.exists(csv_file):
            res = QMessageBox.question(self, "File Missing", "CSV not found. Continue anyway?", QMessageBox.Yes | QMessageBox.No)
            if res != QMessageBox.Yes:
                return

        # Ensure local config exists (create if necessary)
        if not self.local_config_path:
            # attempt to create if file provided
            if csv_file:
                self.load_or_create_local_config(csv_file)

        # Set environment variables (string values) to match Tkinter behavior
        os.environ['CALIBRATION_MACHINE_ID'] = self.machine_combo.currentText().split(' - ')[0]
        os.environ['CALIBRATION_FILE_PATH'] = csv_file

        os.environ['CALIBRATION_FZ_ANALYSIS'] = str(self.checkboxes["Fz Analysis"].isChecked())
        os.environ['CALIBRATION_FX_ANALYSIS'] = str(self.checkboxes["Fx Analysis"].isChecked())
        os.environ['CALIBRATION_FY_ANALYSIS'] = str(self.checkboxes["Fy Analysis"].isChecked())
        os.environ['CALIBRATION_MZ_ANALYSIS'] = str(self.checkboxes["Mz Analysis"].isChecked())
        os.environ['CALIBRATION_MX_ANALYSIS'] = str(self.checkboxes["Mx Analysis"].isChecked())
        os.environ['CALIBRATION_MY_ANALYSIS'] = str(self.checkboxes["My Analysis"].isChecked())
        os.environ['CALIBRATION_FT_ANALYSIS'] = str(self.checkboxes["FT Analysis"].isChecked())


        os.environ['CALIBRATION_USE_STEP_ANALYSIS'] = str(self.analysis_checkboxes["Regular Step Analysis"].isChecked())
        os.environ['CALIBRATION_USE_3STEP_ANALYSIS'] = str(self.analysis_checkboxes["3-Step Analysis"].isChecked())
        os.environ['CALIBRATION_USE_TEST_PHASE'] = str(self.analysis_checkboxes["Test Phase Extraction"].isChecked())
        os.environ['CALIBRATION_USE_MEASUREMENT'] = str(self.analysis_checkboxes["Measurement Extraction"].isChecked())
        os.environ['CALIBRATION_USE_3STEP_EXTRACTION'] = str(self.analysis_checkboxes["3-Step Evaluation"].isChecked())


        # At this point, launch script using QProcess so we can stream output safely in the GUI thread
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Already running", "An analysis is already running")
            return

        self.process = QProcess(self)
        # Merge channels so we can read stdout which contains both
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        # Build arguments: pass CSV path and local config path
        args = [script_path, csv_file]
        if self.local_config_path:
            args.append(self.local_config_path)

        program = sys.executable if sys.executable else 'python'

        # Connect signals
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(lambda e: self.log_output(f"QProcess Error: {e}"))

        # Start process
        self.process.start(program, args)
        started = self.process.waitForStarted(3000)
        if not started:
            QMessageBox.critical(self, "Start failed", "Failed to start the analysis process")
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _read_process_output(self):
        if not self.process:
            return
        data = self.process.readAllStandardOutput().data().decode(errors='replace')
        if data:
            # append preserving trailing newlines
            self.output_console.moveCursor(self.output_console.textCursor().End)
            self.output_console.insertPlainText(data)
            self.output_console.moveCursor(self.output_console.textCursor().End)

    def _process_finished(self, exitCode, exitStatus):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # Ensure we flush any remaining output
        self._read_process_output()

    def stop_analysis(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(2000)
            self.log_output("Process killed by user")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    # ---------------- Utilities ----------------
    def clear_output(self):
        self.output_console.clear()

    def log_output(self, msg):
        self.output_console.append(msg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CalibrationUI()
    w.show()
    sys.exit(app.exec_())
