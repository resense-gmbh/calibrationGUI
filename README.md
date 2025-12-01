# Calibration GUI

Grafical User Interface for calibration tools

## Usage

1. **Checkout repository**:

   ```bash
   git clone https://github.com/resense-gmbh/calibrationGUI.git
   cd calibrationGUI
   ```

2. **Open workspace with VSCode**:
   - Start VSCode
   - `File` → `Open Workspace from File` → Choose workspace file (.vscode/calibration_gui.code-workspace)

3. **Run workspace setup**:
   - Press: `Ctrl+Shift+B` (Run Build Task)
   - Or: `Ctrl+Shift+P` → `Tasks: Run Build Task` → `Setup Workspace`

   The setup automatically performs the following steps:
   - ✅ Checks if virtual environment (.venv) exists
   - 🔧 Creates virtual environment if not present
   - 📦 Installs all requirements from `requirements.txt`
   - ✅ Confirms successful setup

4. **Ready to Work**:
   - All new terminals automatically use the virtual environment
   - Python interpreter is correctly configured
   - Dependencies are installed and ready to use

## Development

### Available Tasks

- **`Ctrl+Shift+B`**: Setup Workspace (Default Build Task)
- **Create Virtual Environment**: Creates only the virtual environment
- **Install Requirements**: Installs dependencies (with automatic venv setup)
- **Update Requirements**: Updates `requirements.txt` with current packages

### Debugging

To debug, use F5 or the debug configurations:

- **Python Debugger: Calibration GUI** - Starts the main application
- **Python Debugger: Current File** - Debugs the currently open file

Both configurations automatically run the requirements setup before debugging.
