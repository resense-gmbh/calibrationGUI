import os
import sys
import subprocess
import json


def debug_labview_types(file_name, sample_start=None, shift=None, mvg=None, thr=None, com_port=None, machine_id=None):
    """Debug function to check LabVIEW data types - call this from LabVIEW to diagnose type issues"""
    try:
        debug_info = {
            'file_name': {'value': file_name, 'type': str(type(file_name)), 'hasattr_value': hasattr(file_name, 'value')},
            'sample_start': {'value': sample_start, 'type': str(type(sample_start)), 'hasattr_value': hasattr(sample_start, 'value') if sample_start is not None else False},
            'shift': {'value': shift, 'type': str(type(shift)), 'hasattr_value': hasattr(shift, 'value') if shift is not None else False},
            'mvg': {'value': mvg, 'type': str(type(mvg)), 'hasattr_value': hasattr(mvg, 'value') if mvg is not None else False},
            'thr': {'value': thr, 'type': str(type(thr)), 'hasattr_value': hasattr(thr, 'value') if thr is not None else False},
            'com_port': {'value': com_port, 'type': str(type(com_port)), 'hasattr_value': hasattr(com_port, 'value') if com_port is not None else False},
            'machine_id': {'value': machine_id, 'type': str(type(machine_id)), 'hasattr_value': hasattr(machine_id, 'value') if machine_id is not None else False}
        }
        
        debug_output = "=== LabVIEW TYPE DEBUG ===\n"
        for param, info in debug_info.items():
            debug_output += f"{param}: value={info['value']}, type={info['type']}, has_value_attr={info['hasattr_value']}\n"
        
        return 0, debug_output, ""
    except Exception as e:
        return 1, "", f"Debug error: {str(e)}"


def process_data(file_name, sampleStart, shift, MVG, thr, comPort, machine_id=2):
    """
    Bridge function for LabVIEW integration.
    
    Parameters:
    - file_name: full path to the CSV data file
    - sampleStart: starting sample index for analysis
    - shift: shift parameter for processing
    - MVG: moving average window size
    - thr: threshold value
    - comPort: COM port string
    - machine_id: machine identifier (numeric like 2 or string like 'CM2', default: 2 which becomes 'CM2')
    
    Returns: None (void function - LabVIEW handles errors through console output)
    """
    # Convert LabVIEW types to Python types as simply as possible
    file_path = os.path.abspath(file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Prepare environment for the subprocess
    env = os.environ.copy()
    env['CALIBRATION_FILE_PATH'] = file_path
    
    # Convert machine_id properly - LabVIEW might send 2.0, convert to 'CM2'
    if isinstance(machine_id, (int, float)):
        # If LabVIEW sends numeric value like 2.0, convert to 'CM2'
        machine_str = f'CM{int(machine_id)}'
    else:
        # If already string, use as-is
        machine_str = str(machine_id)
    
    env['CALIBRATION_MACHINE_ID'] = machine_str
    
    # Convert parameters exactly like old script did: str(sampleStart), str(shift), etc.
    # No int() conversion that might cause LabVIEW type issues
    env['LABVIEW_SAMPLE_START'] = str(sampleStart)
    env['LABVIEW_OVERRIDE_SAMPLE_START'] = 'true'
    env['LABVIEW_SHIFT'] = str(shift)
    env['LABVIEW_OVERRIDE_SHIFT'] = 'true'
    env['LABVIEW_MVG'] = str(MVG)
    env['LABVIEW_OVERRIDE_MVG'] = 'true'
    env['LABVIEW_THR'] = str(thr)
    env['LABVIEW_OVERRIDE_THR'] = 'true'
    env['LABVIEW_COMPORT'] = str(comPort)
    env['LABVIEW_OVERRIDE_COMPORT'] = 'true'

    # Step 1: Run Raw_Step.py first
    raw_script_path = os.path.join(os.path.dirname(__file__), 'Raw_Step.py')
    raw_cmd = [sys.executable, raw_script_path]
    raw_proc = subprocess.Popen(raw_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, raw_stderr = raw_proc.communicate()

    # If Raw_Step fails, raise exception (LabVIEW will catch it)
    if raw_proc.returncode != 0:
        raise RuntimeError(f"Raw_Step failed: {raw_stderr}")

    # Step 2: Run Analysis.py
    analysis_script_path = os.path.join(os.path.dirname(__file__), 'Analysis.py')
    analysis_cmd = [sys.executable, analysis_script_path]
    analysis_proc = subprocess.Popen(analysis_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, analysis_stderr = analysis_proc.communicate()

    # Check if Analysis failed and raise exception if so
    if analysis_proc.returncode != 0:
        raise RuntimeError(f"Analysis failed: {analysis_stderr}")

    # If everything succeeded, function completes without return value
    # LabVIEW will see success through the error output being empty


if __name__ == '__main__':
    # Test with the same data as the old script
    file_name = r"C:\Users\SahayaJ\OneDrive - WIKA\Desktop\2025_08_18_MIEC8E3_Q325_HEX10_195__175min_m1\2025_08_18_MIEC8E3_Q325_HEX10_195__175min_m1.csv"
    result = process_data(file_name, 85000, 80, 200, 3000, 'COM28', 'CM2')
    print('Result:', result)
FileNotFoundError()