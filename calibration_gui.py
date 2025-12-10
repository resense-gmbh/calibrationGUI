
import os
import sys
import json
import re
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib import cm 

from PyQt5.QtCore import Qt, QProcess, QStandardPaths
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QCheckBox,
    QTextEdit, QGroupBox, QPlainTextEdit, QDialog, QSplitter, QListWidget,
    QListWidgetItem, QAbstractItemView
)

#res

def get_resource_path(relative_path: str) -> str:
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def resource_dir() -> str:
    # Support PyInstaller 'frozen' apps
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

#GUI

class PlotWindow(QDialog):
    def __init__(self, parent=None, title="KPI Plot"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 700)

        self.fig = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        self.fig.clear()

    def draw(self):
        self.canvas.draw()

class CalibrationUI(QWidget):
    # KPI column indices (Excel layout)
    SENSOR_COL_IDX = 1   # Column B
    QUARTER_COL_IDX = 3  # Column D
    KPI_START_IDX   = 4  # Column E onward
    AXES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resense Calibration Analysis System")
        self.resize(1200, 800)

        # Paths and config structures
        self.base_dir = resource_dir()
        self.machines_config_path = os.path.join(self.base_dir, 'machines_config.json')
        self.config_data = {}
        self.machine_defaults = {}
        self.local_config = {}
        self.local_config_path = None

        # Process handle
        self.process = None

        # KPI data
        self.kpi_df = None
        self.kpi_excel_path = None
        self.kpi_sensors = []
        self.kpi_groups = {ax: [] for ax in self.AXES}       # axis -> list of KPI columns (param_axis)
        self.params_by_axis = {ax: set() for ax in self.AXES}# axis -> set of base params
        self.kpi_param_union = []

        # Load machine config file
        if not os.path.exists(self.machines_config_path):
            QMessageBox.critical(self, "Missing file",
                                 f"machines_config.json not found at {self.machines_config_path}")
            self.config_data = {"machines": {}, "scripts": {}}
        else:
            try:
                with open(self.machines_config_path, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Config Error", f"Failed to read machines_config.json:\n{e}")
                self.config_data = {"machines": {}, "scripts": {}}

        self._build_ui()
        self._populate_machines()
        self._populate_scripts()
        self.plot_win = None  
    # ---------------------------- UI BUILD ----------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # LEFT PANEL
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
        self.browse_btn = QPushButton("Browse")
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

        # Analysis type (checkboxes)
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

        # --- KPI Comparison (Quarter-based) ---
        kpi_box = QGroupBox("KPI Comparison")
        kpi_layout = QVBoxLayout(kpi_box)

        # Row 1: Excel picker + Load
        kpi_file_row = QHBoxLayout()
        kpi_file_row.addWidget(QLabel("KPI Excel:"))
        self.kpi_excel_input = QLineEdit()
        kpi_file_row.addWidget(self.kpi_excel_input)
        self.kpi_browse_btn = QPushButton("Browse")
        self.kpi_browse_btn.clicked.connect(self.browse_kpi_excel)
        kpi_file_row.addWidget(self.kpi_browse_btn)
        self.kpi_load_btn = QPushButton("Load data")
        self.kpi_load_btn.clicked.connect(self.load_quarters)
        kpi_file_row.addWidget(self.kpi_load_btn)
        kpi_layout.addLayout(kpi_file_row)

        # Row 2: Quarter A/B
        quarter_row = QHBoxLayout()
        quarter_row.addWidget(QLabel("Quarter A:"))
        self.quarter1_combo = QComboBox()
        quarter_row.addWidget(self.quarter1_combo)
        quarter_row.addWidget(QLabel("Quarter B:"))
        self.quarter2_combo = QComboBox()
        quarter_row.addWidget(self.quarter2_combo)
        kpi_layout.addLayout(quarter_row)

        # Row 3: Axes checkboxes
        axes_row = QHBoxLayout()
        self.axes_checkboxes = {}
        for ax in self.AXES:
            cb = QCheckBox(ax)
            cb.setChecked(True)  # default include all axes
            self.axes_checkboxes[ax] = cb
            axes_row.addWidget(cb)
            cb.stateChanged.connect(lambda _state: self._refresh_params_from_axes())
        kpi_layout.addLayout(axes_row)

        # Row 4: Parameters multi-select
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Parameters:"))
        self.param_list = QListWidget()
        self.param_list.setSelectionMode(QAbstractItemView.MultiSelection)
        params_row.addWidget(self.param_list, 1)
        self.select_all_params_btn = QPushButton("Select all")
        self.select_all_params_btn.clicked.connect(lambda: self._select_all_params(True))
        params_row.addWidget(self.select_all_params_btn)
        self.clear_params_btn = QPushButton("Clear")
        self.clear_params_btn.clicked.connect(lambda: self._select_all_params(False))
        params_row.addWidget(self.clear_params_btn)
        kpi_layout.addLayout(params_row)

        # Row 5: Plot type + Plot button
        kpi_controls_row = QHBoxLayout()
        kpi_controls_row.addWidget(QLabel("Plot:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Bar","Box", "Paired Bar", "Violin"])
        kpi_controls_row.addWidget(self.plot_type_combo)

        self.plot_btn = QPushButton("Compare quarters")
        self.plot_btn.clicked.connect(self.plot_quarter_comparison)
        kpi_controls_row.addWidget(self.plot_btn)
        kpi_layout.addLayout(kpi_controls_row)

        left_layout.addWidget(kpi_box)

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

    # ---------------------------- Populate helpers ----------------------------

    def _populate_machines(self):
        self.machine_combo.clear()
        machines = self.config_data.get('machines', {})
        items = []
        for mid, info in machines.items():
            name = info.get('name', mid)
            items.append(f"{mid} - {name}")
        if not items:
            items = ["CM1 - CM1_Hex21"]  # safe default
        self.machine_combo.addItems(items)
        if items:
            self.on_machine_selected(0)

    def _populate_scripts(self):
        self.script_combo.clear()
        scripts = self.config_data.get('scripts', {})
        if not scripts:
            default_map = {
                'Raw Step Analysis': 'Raw_Step.py',
                'General Analysis': 'Analysis.py',
                'Step Analysis': 'analysis_step.py',
                'Matrix Calculation': 'calculateMatrix.py'
            }
            for name, fname in default_map.items():
                self.script_combo.addItem(name, fname)
            self.script_descriptions = {}
        else:
            self.script_descriptions = {}
            for script_file, meta in scripts.items():
                label = meta.get('name', script_file)
                desc  = meta.get('description', '')
                self.script_combo.addItem(label, script_file)
                self.script_descriptions[label] = desc

    # ---------------------------- Machine selection ----------------------------

    def on_machine_selected(self, index: int):
        current = self.machine_combo.currentText()
        if not current:
            return
        machine_id = current.split(' - ')[0]
        machines = self.config_data.get('machines') or {}
        self.machine_defaults = machines.get(machine_id, {}).get('default_config', {})

    # ---------------------------- File selection & local config ----------------------------

    def find_onedrive_path(self):
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

    def browse(self):
        base = self.find_onedrive_path() or QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration CSV File", base, "CSV Files (*.csv)"
        )
        if file_path:
            self.file_input.setText(file_path)
            self.load_or_create_local_config(file_path)

    def load_or_create_local_config(self, csv_file: str):
        if not isinstance(csv_file, str):
            self.log_output("Invalid CSV file path")
            return
        data_folder = os.path.dirname(csv_file)
        local_cfg = os.path.join(data_folder, 'machine_config.json')
        self.local_config_path = local_cfg

        # If exists -> load
        if os.path.exists(local_cfg):
            try:
                with open(local_cfg, 'r', encoding='utf-8') as f:
                    self.local_config = json.load(f)
                self.config_editor.setPlainText(json.dumps(self.local_config, indent=4))
                return
            except Exception as e:
                self.log_output(f"Failed to read local config: {e}")

        # Otherwise create using machine defaults
        if not self.machine_defaults:
            self.local_config = {}
            self.config_editor.setPlainText('{}')
            return
        self.local_config = self.machine_defaults.copy()
        try:
            with open(local_cfg, 'w', encoding='utf-8') as f:
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
            text = self.config_editor.toPlainText()
            parsed = json.loads(text)
            with open(self.local_config_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=4)
            self.local_config = parsed
            self.log_output(f"Saved local config to {self.local_config_path}")
        except json.JSONDecodeError as jde:
            QMessageBox.critical(self, "JSON Error", f"Config editor contains invalid JSON:\n{jde}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def reload_local_config(self):
        if not self.local_config_path or not os.path.exists(self.local_config_path):
            QMessageBox.warning(self, "Missing", "Local machine_config.json not found to reload")
            return
        try:
            with open(self.local_config_path, 'r', encoding='utf-8') as f:
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
        if self.local_config_path:
            try:
                with open(self.local_config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.local_config, f, indent=4)
                self.log_output("Local config reset to machine defaults and saved")
            except Exception as e:
                self.log_output(f"Failed to save reset config: {e}")

    # ---------------------------- Script Info ----------------------------

    def show_script_info(self):
        label = self.script_combo.currentText()
        desc = getattr(self, 'script_descriptions', {}).get(label)
        if desc:
            QMessageBox.information(self, "Script Info", desc)
        else:
            QMessageBox.information(self, "Script Info", "No description available for this script")

    # ---------------------------- OneDrive helpers ----------------------------

    def sync_onedrive_file(self):
        file_path = self.file_input.text().strip()
        if not file_path:
            QMessageBox.warning(self, "No file", "Please select a data CSV first")
            return
        od = self.find_onedrive_path()
        if not od:
            QMessageBox.critical(self, "OneDrive not found", "OneDrive folder not detected on this machine")
            return
        if 'OneDrive' not in file_path and not file_path.startswith(od):
            QMessageBox.information(self, "Not OneDrive", "Selected file is not in OneDrive folder; nothing to sync")
            return
        try:
            if sys.platform.startswith('win'):
                os.startfile(os.path.dirname(file_path))
            else:
                import subprocess as sp
                sp.Popen(['xdg-open', os.path.dirname(file_path)])
            QMessageBox.information(self, "Sync",
                                    "Opened the folder containing the file to trigger OneDrive sync (if applicable)")
        except Exception as e:
            QMessageBox.warning(self, "Sync error", f"Could not open folder: {e}")

    def auto_find_files(self):
        search_paths = []
        od = self.find_onedrive_path()
        if od:
            search_paths.append(od)
        docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        if docs:
            search_paths.append(docs)

        found, limit = [], 200
        for root in search_paths:
            for dirpath, dirnames, filenames in os.walk(root):
                # Optional pruning: skip heavy folders
                skip = {'node_modules', '.git', '.venv', '__pycache__'}
                dirnames[:] = [d for d in dirnames if d not in skip]
                for fn in filenames:
                    name = fn.lower()
                    if name.endswith('.csv') and ('cal' in name or 'miec' in name or 'calibration' in name):
                        found.append(os.path.join(dirpath, fn))
                        if len(found) >= limit:
                            break
                if len(found) >= limit:
                    break
            if len(found) >= limit:
                break

        if not found:
            QMessageBox.information(self, "Auto-Find", "No likely calibration CSV files found")
            return
        selected, _ = QFileDialog.getOpenFileName(
            self, "Select calibration CSV", os.path.dirname(found[0]), "CSV Files (*.csv)"
        )
        if selected:
            self.file_input.setText(selected)
            self.load_or_create_local_config(selected)

    # ---------------------------- KPI Excel: browse & robust load ----------------------------

    def browse_kpi_excel(self):
        dir = r"C:\Users\DaryapS\WIKA\EM-CEO-ORG-Resense - Files\110_Manufacturing\Calibration\00_KPI"
        path, _ = QFileDialog.getOpenFileName(
            self, "Select KPI Excel", dir, "Excel Files (*.xlsx *.xls)"
        )
        if path:
            self.kpi_excel_path = path
            self.kpi_excel_input.setText(path)

    def _normalize_columns(self, cols):
        """Normalize column labels to safe strings."""
        norm = []
        for c in cols:
            s = str(c)
            s = s.replace("\n", " ").strip()
            # unify separators to underscore
            s = re.sub(r"[ \t\-]+", "_", s)
            # collapse multiple underscores
            s = re.sub(r"_+", "_", s)
            # drop leading/trailing underscores
            s = s.strip("_")
            norm.append(s)
        return norm

    def _flatten_multiindex(self, df: pd.DataFrame) -> pd.DataFrame:
        """If df.columns is a MultiIndex, flatten by joining non-empty levels."""
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for tup in df.columns:
                parts = [str(x) for x in tup if str(x) != "nan"]
                label = "_".join(parts)
                new_cols.append(label)
            df = df.copy()
            df.columns = self._normalize_columns(new_cols)
        else:
            df = df.copy()
            df.columns = self._normalize_columns(df.columns)
        return df

    def _read_kpi_excel_robust(self, path: str) -> pd.DataFrame:
        """Try different header rows and pick the variant with the richest KPI set."""
        candidates = []
        for hdr in (0, 1, 2):
            try:
                df_try = pd.read_excel(path, header=hdr, engine="openpyxl")
                df_try = self._flatten_multiindex(df_try)
                # drop empty rows
                df_try = df_try.dropna(how="all")
                # score: KPI-like columns after index >= 4
                kpi_like = sum(bool(re.search(r"_(Fx|Fy|Fz|Mx|My|Mz)$", c))
                               for c in df_try.columns[self.KPI_START_IDX:])
                candidates.append((kpi_like, df_try))
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        # fallback simple read
        df = pd.read_excel(path, engine="openpyxl")
        return self._flatten_multiindex(df)

    def load_quarters(self):
        """
        Read KPI Excel, infer quarters, sensors, and (axis, parameter) structure.
        """
        try:
            self.kpi_excel_path = self.kpi_excel_input.text().strip() or self.kpi_excel_path
            if not self.kpi_excel_path or not os.path.exists(self.kpi_excel_path):
                QMessageBox.warning(self, "KPI Excel", "Please select a valid KPI Excel file first.")
                return

            df = self._read_kpi_excel_robust(self.kpi_excel_path)
            self.kpi_df = df

            # Log preview of columns
            preview_cols = ", ".join(df.columns[:12])
            self.log_output(f"KPI Excel loaded. First columns: {preview_cols}")

            # Validate shape
            if df.shape[1] < self.KPI_START_IDX + 1:
                QMessageBox.critical(self, "KPI Excel",
                                     "The Excel file doesn't have enough columns for B/D/E+. Please check layout.")
                return

            # Quarters
            quarters = (df.iloc[:, self.QUARTER_COL_IDX]
                        .dropna().astype(str).str.strip().unique().tolist())
            quarters = sorted(quarters)
            self.quarter1_combo.clear()
            self.quarter2_combo.clear()
            self.quarter1_combo.addItems(quarters)
            self.quarter2_combo.addItems(quarters)

            # Sensors (context)
            self.kpi_sensors = (df.iloc[:, self.SENSOR_COL_IDX]
                                .dropna().astype(str).str.strip().tolist())

            # KPI columns (E+)
            kpi_all = list(df.columns[self.KPI_START_IDX:])

            # Group KPI columns by axis suffix and extract parameter names
            self.kpi_groups = {ax: [] for ax in self.AXES}
            self.params_by_axis = {ax: set() for ax in self.AXES}

            pat = re.compile(r"^(?P<param>.+)_(?P<axis>Fx|Fy|Fz|Mx|My|Mz)$", re.IGNORECASE)
            for col in kpi_all:
                m = pat.match(str(col))
                if m:
                    base = m.group("param")
                    axis = m.group("axis")
                    # normalize axis casing
                    axis = axis[:1].upper() + axis[1:].lower()
                    base_norm = self._normalize_columns([base])[0]
                    self.kpi_groups[axis].append(f"{base_norm}_{axis}")
                    self.params_by_axis[axis].add(base_norm)

            # Build parameter union from currently checked axes
            selected_axes_now = [ax for ax, cb in self.axes_checkboxes.items() if cb.isChecked()]
            if not selected_axes_now:
                selected_axes_now = self.AXES

            param_union = sorted({p for ax in selected_axes_now for p in self.params_by_axis[ax]})

            # Diagnostics if empty
            if not param_union:
                diag = []
                for ax in self.AXES:
                    cols = self.kpi_groups[ax]
                    if cols:
                        preview = ", ".join(cols[:8]) + (" …" if len(cols) > 8 else "")
                        diag.append(f"{ax}: {preview}")
                detail = "\n".join(diag) if diag else "No KPI columns matched the *_Fx|*_Fy|... pattern."
                QMessageBox.warning(self, "KPI Excel",
                                    "No parameters found for the currently selected axes.\n\n"
                                    "Diagnostics:\n" + detail)
                self.log_output("Parameter union is empty. Check column names and header rows.")

            # Populate parameter list
            self.param_list.clear()
            for p in param_union:
                item = QListWidgetItem(p)
                item.setSelected(True)
                self.param_list.addItem(item)

            self.kpi_param_union = param_union
            self.log_output(f"Meta: {len(quarters)} quarter(s), {len(self.kpi_sensors)} sensor(s), "
                            f"{sum(len(v) for v in self.kpi_groups.values())} KPI columns.")
        except Exception as e:
            QMessageBox.critical(self, "Load Meta Error", str(e))

    def _refresh_params_from_axes(self):
        """Refresh parameter union when axis checkboxes change."""
        if self.kpi_df is None:
            return
        axes = [ax for ax, cb in self.axes_checkboxes.items() if cb.isChecked()]
        if not axes:
            axes = self.AXES
        param_union = sorted({p for ax in axes for p in self.params_by_axis.get(ax, set())})
        self.param_list.clear()
        for p in param_union:
            item = QListWidgetItem(p)
            item.setSelected(True)
            self.param_list.addItem(item)
        self.kpi_param_union = param_union
        self.log_output(f"Axes changed → {len(param_union)} param(s) listed.")

    def _select_all_params(self, select: bool = True):
        for i in range(self.param_list.count()):
            self.param_list.item(i).setSelected(select)

    def _expand_params_to_columns(self, selected_axes, selected_params):
        """Turn (axes, params) into concrete KPI column names that exist in the sheet."""
        selected_cols, seen = [], set()
        for ax in selected_axes:
            existing = set(self.kpi_groups.get(ax, []))
            for p in selected_params:
                c = f"{p}_{ax}"
                if c in existing and c not in seen:
                    selected_cols.append(c)
                    seen.add(c)
        if not selected_cols:
            raise ValueError("No KPI columns matched the chosen axes/parameters.")
        return selected_cols

    def _build_quarter_sensor_matrix(self, quarter_value, kpi_cols):
        """Return sensors list, non-empty KPI subset, and KPI matrix for the selected quarter."""
        df = self.kpi_df.copy()
        qmask = (df.iloc[:, self.QUARTER_COL_IDX].astype(str).str.strip().str.lower()
                 == str(quarter_value).strip().lower())
        dfq = df[qmask].copy()
        if dfq.empty:
            raise ValueError(f"No rows found for Quarter='{quarter_value}'")
        sensors = dfq.iloc[:, self.SENSOR_COL_IDX].astype(str).str.strip().tolist()
        kpi_numeric = dfq[kpi_cols].apply(pd.to_numeric, errors="coerce")
        non_empty_kpis = kpi_numeric.columns[kpi_numeric.notna().any(axis=0)].tolist()
        kpi_numeric = kpi_numeric[non_empty_kpis]
        return sensors, non_empty_kpis, kpi_numeric

    # ---------------------------- Plotting ----------------------------

    def _normalize_matrix_01(self, mat: pd.DataFrame) -> pd.DataFrame:
        """Per-KPI min-max normalization to [0,1] within a quarter, for radar readability."""
        m = mat.copy()
        for c in m.columns:
            col = m[c]
            mn, mx = col.min(skipna=True), col.max(skipna=True)
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                m[c] = 0.0
            else:
                m[c] = (col - mn) / (mx - mn)
        return m


    def _bar_single_quarter(self, ax, sensors, mat, title):
        """Fallback bar chart when only one KPI is selected."""
        x = np.arange(len(sensors))
        width = 0.8
        cmap = cm.get_cmap('tab20')
        vals = mat.iloc[:, 0].values if mat.shape[1] else np.zeros(len(sensors))

        bars = ax.bar(x, vals, width=width, color=[cmap(i % cmap.N) for i in range(len(sensors))])

        ax.set_xticks(x)
        ax.set_xticklabels(sensors, rotation=45, ha='right', fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(mat.columns[0] if mat.shape[1] else "Value")
        ax.grid(axis='y', linestyle=':', alpha=0.4)

        # Add value labels
        self._annotate_bars(ax, bars, fmt="{:.2f}", dy=0.02)


    def _plot_quarter_boxplots(self, fig, mat_q1, mat_q2, kpis_common, quarter_a, quarter_b):
        """
        Grouped boxplots per KPI: for each KPI, show two boxes (Quarter A vs Quarter B).
        mat_q1/mat_q2: DataFrames (rows=sensors, cols=KPIs).
        """
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"Box plot — {quarter_a} vs {quarter_b}")
        ax.set_xlabel("KPI")
        ax.set_ylabel("Value")

        # Collect data per KPI
        data = []
        labels = []
        positions = []
        colors = []
        width = 0.35  # spacing offset

        # positions: for kpi i, place quarter A at i - width/2, quarter B at i + width/2
        for i, kpi in enumerate(kpis_common):
            vals_a = pd.to_numeric(mat_q1[kpi], errors="coerce").dropna().values
            vals_b = pd.to_numeric(mat_q2[kpi], errors="coerce").dropna().values
            # Keep empty arrays as empty (boxplot will skip if empty)
            data.extend([vals_a, vals_b])
            labels.extend([f"{kpi}\n{quarter_a}", f"{kpi}\n{quarter_b}"])
            positions.extend([i - width/2, i + width/2])
            colors.extend(["#1f77b4", "#ff7f0e"])

        if not data:
            raise ValueError("No numeric data for box plotting.")

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.3,
            patch_artist=True,  # to color boxes
            showfliers=False    # hide outliers to reduce clutter
        )

        # Color the boxes and add medians in bold
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.35)
            box.set_edgecolor(color)
            box.set_linewidth(1.2)

        for med in bp["medians"]:
            med.set_color("#333")
            med.set_linewidth(2)

        # X ticks at KPI centers
        centers = list(range(len(kpis_common)))
        ax.set_xticks(centers)
        ax.set_xticklabels(kpis_common, rotation=45, ha="right")

        # Build a legend
        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.35, label=quarter_a),
                        Patch(facecolor="#ff7f0e", edgecolor="#ff7f0e", alpha=0.35, label=quarter_b)]
        ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0, 1.15), ncol=2, fontsize=9)

        ax.grid(axis="y", linestyle=":", alpha=0.4)


    def _paired_bar_by_sensor(self, ax, mat_q1, mat_q2, sensors_q1, sensors_q2, kpi, quarter_a, quarter_b):

        # Ensure indices are sensor IDs and strings
        m1 = mat_q1.copy(); m2 = mat_q2.copy()
        m1.index = pd.Index(sensors_q1, dtype=str)
        m2.index = pd.Index(sensors_q2, dtype=str)

        # Aggregate duplicates by mean
        s1 = pd.to_numeric(m1[kpi], errors="coerce").groupby(level=0).mean()
        s2 = pd.to_numeric(m2[kpi], errors="coerce").groupby(level=0).mean()

        # Common sensor set and consistent order
        common = sorted(set(s1.index).intersection(s2.index))
        if not common:
            raise ValueError("No common sensors found for the selected KPI.")
        s1 = s1.reindex(common)
        s2 = s2.reindex(common)

        yA = s1.values
        yB = s2.values
        mask = np.isfinite(yA) & np.isfinite(yB)
        sensors = np.array(common)[mask]
        yA = yA[mask]
        yB = yB[mask]

        if len(sensors) == 0:
            raise ValueError(f"No comparable numeric data for KPI '{kpi}' across the selected quarters.")

        # X positions
        x = np.arange(len(sensors))
        width = 0.42

        # Colors (consistent with your palette)
        color_a, color_b = "#1f77b4", "#ff7f0e"

        # Bars
        barsA = ax.bar(x - width/2, yA, width=width, color=color_a, label=quarter_a)
        barsB = ax.bar(x + width/2, yB, width=width, color=color_b, label=quarter_b)

        # Labels & grid
        ax.set_xlabel("Sensor")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Sensor tick labels (reduce density if many sensors)
        step = max(1, len(sensors)//30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([sensors[i] for i in range(0, len(sensors), step)], rotation=45, ha="right", fontsize=8)

        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2, fontsize=9)

        # Annotate bar values (reuse your helper)
        self._annotate_bars(ax, barsA, fmt="{:.2f}", dy=0.02)
        self._annotate_bars(ax, barsB, fmt="{:.2f}", dy=0.02)

        # Optional: annotate Δ (QuarterB - QuarterA) above each pair
        diffs = yB - yA
        y_min, y_max = ax.get_ylim()
        dy = (y_max - y_min) * 0.03
        for xi, d in zip(x, diffs):
            ax.annotate(f"Δ={d:+.2f}", xy=(xi, max(yA[xi], yB[xi])),
                        xytext=(0, dy), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#333")



    def _plot_quarter_bars(self, ax, sensors, mat, title):
        """Grouped bars: sensors per KPI."""
        num_kpis = len(mat.columns)
        x = np.arange(num_kpis)
        width = 0.8 / max(1, len(sensors))
        cmap = cm.get_cmap('tab20')

        all_bars = []  # collect for annotation

        for i, sensor in enumerate(sensors):
            vals = mat.iloc[i].values if i < len(mat) else np.zeros(num_kpis)
            bars = ax.bar(x + i*width, vals, width=width, label=str(sensor), color=cmap(i % cmap.N))
            all_bars.extend(bars)

        ax.set_xticks(x + (width * (len(sensors)-1) / 2))
        ax.set_xticklabels(list(mat.columns), rotation=45, ha='right')
        ax.set_title(title)
        ax.set_xlabel("KPI")
        ax.set_ylabel("Value")
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2, fontsize=8)

        # Add value labels on all bars
        self._annotate_bars(ax, all_bars, fmt="{:.2f}", dy=0.02)





    def _plot_quarter_violins(self, fig, mat_q1, mat_q2, kpis_common, quarter_a, quarter_b, common_sensors=None):
        """
        Grouped violin plots per KPI: for each KPI, show distributions across sensors.
        Two panels: Quarter A and Quarter B.
        mat_q1/mat_q2: DataFrames (index=sensor_id, columns=KPI names)
        """
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Optionally align both quarters to the same sensors
        if common_sensors is not None and len(common_sensors) > 0:
            mat_q1_plot = mat_q1.loc[common_sensors, kpis_common]
            mat_q2_plot = mat_q2.loc[common_sensors, kpis_common]
        else:
            mat_q1_plot = mat_q1[kpis_common]
            mat_q2_plot = mat_q2[kpis_common]


# Draw each panel
        self._violin_panel(ax1, mat_q1_plot, kpis_common, quarter=quarter_a, title=f"Quarter: {quarter_a}")
        self._violin_panel(ax2, mat_q2_plot, kpis_common, quarter=quarter_b, title=f"Quarter: {quarter_b}")

        # Keep same Y scale for fair visual comparison (handles different sensor counts)
        try:
            vmin = min(np.nanmin(mat_q1_plot.values), np.nanmin(mat_q2_plot.values))
            vmax = max(np.nanmax(mat_q1_plot.values), np.nanmax(mat_q2_plot.values))
            if np.isfinite(vmin) and np.isfinite(vmax):
                ax1.set_ylim(vmin, vmax)
                ax2.set_ylim(vmin, vmax)
        except Exception:
            # Fallback: let Matplotlib autoscale if anything odd happens
            pass

        fig.suptitle(
            f"Violin — {', '.join(kpis_common)}\n{quarter_a} vs {quarter_b}",
            fontsize=12
        )
        fig.tight_layout()



        
        
    def _violin_panel(self, ax, mat, kpis, quarter, title):
        """
        One panel with multiple violins (one per KPI) from values across sensors.
        mat: DataFrame (index=sensor_id, columns=kpis)
        kpis: list[str] columns to plot
        quarter: str (e.g., "2024-Q1") for tooltips
        """
        # Collect values per KPI (preserve sensor ids)
        data = []
        sensor_lists = []
        for kpi in kpis:
            series = pd.to_numeric(mat[kpi], errors="coerce")
            mask = series.notna()
            vals = series[mask].values
            sids = series.index[mask].astype(str).values
            data.append(vals if len(vals) > 0 else np.array([]))
            sensor_lists.append(sids if len(sids) > 0 else np.array([]))

        # Positions for violins
        positions = np.arange(1, len(kpis) + 1)

        # Create violins; show mean/median/extrema for interpretability
        parts = ax.violinplot(
            dataset=data,
            positions=positions,
            showmeans=True,
            showmedians=True,
            showextrema=True
        )

        # Style bodies consistently with your palette
        for i, body in enumerate(parts.get('bodies', [])):
            body.set_facecolor("#1f77b4")
            body.set_edgecolor("#1f77b4")
            body.set_alpha(0.30)

        # Median/mean/extrema line styling
        if "cmeans" in parts:
            parts["cmeans"].set_color("#333"); parts["cmeans"].set_linewidth(1.5)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#333"); parts["cmedians"].set_linewidth(2.0)
        if "cmaxes" in parts:
            parts["cmaxes"].set_color("#555")
        if "cmins" in parts:
            parts["cmins"].set_color("#555")

        # Overlay jittered points (each dot = one sensor), make them clickable
        rng = np.random.default_rng(42)
        for pos, vals, sids, kpi in zip(positions, data, sensor_lists, kpis):
            if vals.size == 0:
                continue
            x = pos + 0.05 * rng.normal(size=len(vals))
            sc = ax.scatter(x, vals, s=22, alpha=0.60, color="#343a40", picker=True)
            # Attach metadata to the artist for pick_event
            sc._meta = {
                "quarter": quarter,
                "kpi": kpi,
                "sensor_ids": sids,
                "values": vals,
                "x": x
            }

        # Axis labels & grid
        ax.set_xticks(positions)
        ax.set_xticklabels(kpis, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel("KPI")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle=":", alpha=0.4)




    def _on_pick_point(self, event):
        """
        Matplotlib pick_event handler for scatter points created in _violin_panel.
        Shows a small annotation near the clicked point with sensor/KPI/value/quarter.
        """
        artist = event.artist
        if not hasattr(artist, "_meta"):
            return

        meta = artist._meta
        inds = event.ind
        if inds is None or len(inds) == 0:
            return
        idx = int(inds[0])

        # Extract info
        sensor_id = str(meta["sensor_ids"][idx]) if idx < len(meta["sensor_ids"]) else "?"
        val = float(meta["values"][idx]) if idx < len(meta["values"]) else np.nan
        x = float(meta["x"][idx]) if idx < len(meta["x"]) else None
        quarter = meta.get("quarter", "")
        kpi = meta.get("kpi", "")

        ax = event.mouseevent.inaxes
        if ax is None:
            return

        # Remove previous annotation if present
        if hasattr(self, "_hover_ann") and self._hover_ann:
            try:
                self._hover_ann.remove()
            except Exception:
                pass
            self._hover_ann = None

        # Create a new annotation near the point
        txt = f"Sensor: {sensor_id}\nQuarter: {quarter}\nKPI: {kpi}\nValue: {val:.4g}"
        self._hover_ann = ax.annotate(
            txt,
            xy=(x, val),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6),
            arrowprops=dict(arrowstyle="->", color="#333", alpha=0.6)
        )
        self.plot_win.canvas.draw_idle()

    def plot_quarter_comparison(self):
        if self.kpi_df is None:
            QMessageBox.warning(self, "KPI", "Load KPI data first.")
            return

        quarter_a = self.quarter1_combo.currentText().strip()
        quarter_b = self.quarter2_combo.currentText().strip()
        if not quarter_a or not quarter_b:
            QMessageBox.warning(self, "KPI", "Please select both quarters.")
            return

        selected_axes = [ax for ax, cb in self.axes_checkboxes.items() if cb.isChecked()]
        if not selected_axes:
            QMessageBox.warning(self, "KPI", "Please select at least one axis (Fx/Fy/Fz/Mx/My/Mz).")
            return

        selected_params = [
            self.param_list.item(i).text()
            for i in range(self.param_list.count())
            if self.param_list.item(i).isSelected()
        ]
        if not selected_params:
            QMessageBox.warning(self, "KPI", "Please select at least one parameter.")
            return

        try:
            # Expand params to concrete KPI columns
            kpi_selected = self._expand_params_to_columns(selected_axes, selected_params)

            # Build matrices
            sensors_q1, kpis_q1, mat_q1 = self._build_quarter_sensor_matrix(quarter_a, kpi_selected)
            sensors_q2, kpis_q2, mat_q2 = self._build_quarter_sensor_matrix(quarter_b, kpi_selected)

            # Align KPI columns present in both quarters
            kpis_common = [k for k in kpis_q1 if k in kpis_q2]
            if not kpis_common:
                raise ValueError("The selected KPI(s) have no numeric data in common between the two quarters.")

            mat_q1 = mat_q1[kpis_common].reset_index(drop=True)
            mat_q2 = mat_q2[kpis_common].reset_index(drop=True)



            # Set index to sensor names to facilitate alignment for facet lines
            mat_q1.index = sensors_q1
            mat_q2.index = sensors_q2
            common_sensors = sorted(set(sensors_q1).intersection(sensors_q2))

            # --- Always plot in a separate window ---
            # Create or reuse the popup window
            if self.plot_win is None or not self.plot_win.isVisible():
                self.plot_win = PlotWindow(self, title="KPI Comparison")
            else:
                self.plot_win.clear()

            fig = self.plot_win.fig
            fig.clear()
            plot_type = self.plot_type_combo.currentText()

            if plot_type.lower() == "box":
                # Grouped box plots per KPI
                self._plot_quarter_boxplots(fig, mat_q1, mat_q2, kpis_common, quarter_a, quarter_b)


            elif plot_type.lower() == "bar":
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                self._plot_quarter_bars(ax1, sensors_q1, mat_q1, f"Quarter: {quarter_a}")
                self._plot_quarter_bars(ax2, sensors_q2, mat_q2, f"Quarter: {quarter_b}")
                fig.suptitle(
                    f"Bars — {', '.join(kpis_common)}\nAll sensors in {quarter_a} vs {quarter_b}",
                    fontsize=12
                )
            
            
            elif plot_type.lower() == "paired bar":
                # Require exactly one KPI (e.g., "f_hys_Fx")
                if len(kpis_common) != 1:
                    raise ValueError("Paired Bar requires exactly one KPI selected.")
                kpi = kpis_common[0]
                ax = fig.add_subplot(1, 1, 1)
                self._paired_bar_by_sensor(ax, mat_q1, mat_q2, sensors_q1, sensors_q2, kpi, quarter_a, quarter_b)
                fig.suptitle(f"Paired Bar — {kpi}   ({quarter_a} vs {quarter_b})", fontsize=12)
                
            
            elif plot_type.lower() == "violin":
                # Grouped violin plots per KPI
                self._plot_quarter_violins(fig, mat_q1, mat_q2, kpis_common, quarter_a, quarter_b)

            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
                
            
            # Connect click-to-tooltip for scatter points
            try:
                # Disconnect old handler if any
                if hasattr(self, "_last_pick_cid"):
                    self.plot_win.canvas.mpl_disconnect(self._last_pick_cid)
                self._last_pick_cid = self.plot_win.canvas.mpl_connect('pick_event', self._on_pick_point)
            except Exception:
                pass

            self.plot_win.draw()
            self.plot_win.showMaximized()  # maximize for more room
            self.plot_win.raise_()
            self.plot_win.activateWindow()

            self.log_output(
                f"Plotted comparison for quarters {quarter_a} vs {quarter_b} "
                f"({len(kpis_common)} KPI(s), {len(sensors_q1)} + {len(sensors_q2)} sensor rows)."
            )


        except Exception as e:
            QMessageBox.critical(self, "Plot Error", str(e))


    def _annotate_bars(self, ax, rects, fmt="{:.2f}", dy=0.02):
        """
        Annotate a list of bar rectangles with their height (value).
        - fmt: formatting string for numeric labels
        - dy : vertical offset in axes-fraction (so it scales with y-limits)
        """
        # Compute offset in data units based on current y-range
        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * dy

        for r in rects:
            height = r.get_height()
            # Place text at top-center of the bar
            ax.annotate(
                fmt.format(height if height is not None else 0),
                xy=(r.get_x() + r.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="data",
                ha="center", va="bottom",
                fontsize=9, rotation=0,
                color="#333"
            )


    # ---------------------------- Run/Stop Script ----------------------------

    def run_analysis(self):
        script_label = self.script_combo.currentText()
        script_file = self.script_combo.currentData()
        if not script_file:
            mapping = {
                'Raw Step Analysis': 'Raw_Step.py',
                'General Analysis': 'Analysis.py',
                'Step Analysis': 'analysis_step.py',
                'Matrix Calculation': 'calculateMatrix.py'
            }
            script_file = mapping.get(script_label, script_label)

        script_path = os.path.join(self.base_dir, script_file)
        if not os.path.exists(script_path):
            script_path = os.path.abspath(script_file)
            if not os.path.exists(script_path):
                QMessageBox.critical(self, "Script Missing",
                                     f"Cannot find script: {script_file}\nLooked at: {script_path}")
                return

        csv_file = self.file_input.text().strip()
        if not csv_file or not os.path.exists(csv_file):
            res = QMessageBox.question(self, "File Missing", "CSV not found. Continue anyway?",
                                       QMessageBox.Yes | QMessageBox.No)
            if res != QMessageBox.Yes:
                return

        # Ensure local config exists (create if necessary)
        if not self.local_config_path:
            if csv_file:
                self.load_or_create_local_config(csv_file)

        # Environment variables for downstream scripts
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

        # Launch using QProcess; stream output
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Already running", "An analysis is already running")
            return

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        args = [script_path, csv_file]
        if self.local_config_path:
            args.append(self.local_config_path)
        program = sys.executable if sys.executable else 'python'

        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(lambda e: self.log_output(f"QProcess Error: {e}"))

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
            self.output_console.moveCursor(self.output_console.textCursor().End)
            self.output_console.insertPlainText(data)
            self.output_console.moveCursor(self.output_console.textCursor().End)

    def _process_finished(self, exitCode, exitStatus):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._read_process_output()  # flush remaining output

    def stop_analysis(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()
                self.process.waitForFinished(2000)
            self.log_output("Process stopped")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    # ---------------------------- Utilities ----------------------------

    def clear_output(self):
        self.output_console.clear()

    def log_output(self, msg: str):
        self.output_console.append(msg)

# ---------------------------- main ----------------------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CalibrationUI()
    w.show()
    sys.exit(app.exec_())
