import os as operating_system_module
import json
import numpy as np
import pandas as pd
from config_manager import ConfigManager

# VERSION IDENTIFIER - to track which version is running
SCRIPT_VERSION = "v2025.09.22.001"
print(f"*** RAW_STEP VERSION {SCRIPT_VERSION} STARTING ***")

def get_machine_and_file():
    """Get machine ID and file path from environment variables or user input"""
    machine_id = operating_system_module.environ.get('CALIBRATION_MACHINE_ID')
    file_path_master = operating_system_module.environ.get('CALIBRATION_FILE_PATH')
    
    if machine_id and file_path_master:
        print("Using parameters from launcher:")
        print(f"Machine: {machine_id}")
        print(f"File: {file_path_master}")
        return machine_id, file_path_master
    
    # Fallback to direct input if not launched from main script
    print("No launcher parameters found. Running in standalone mode.")
    print("ERROR: Raw_Step.py should be run from the main GUI or with environment variables set.")
    print("Required environment variables:")
    print("  CALIBRATION_MACHINE_ID (e.g., CM1, CM2, etc.)")
    print("  CALIBRATION_FILE_PATH (path to CSV data file)")
    print("\nTo run standalone, set these variables first:")
    print("  set CALIBRATION_MACHINE_ID=CM1")
    print("  set CALIBRATION_FILE_PATH=path\\to\\your\\data.csv")
    print("  python Raw_Step.py")
    
    # Exit gracefully instead of trying to continue
    print("\nExiting...")
    exit(1)

def main():
    """Main function to run Raw_Step analysis"""
    # === GET MACHINE AND FILE ===
    machine_id, file_path_master = get_machine_and_file()

    # Check if running from GUI (subprocess) by looking for GUI-specific environment variable
    running_from_gui = operating_system_module.environ.get('CALIBRATION_GUI_MODE') == 'true'

    # === CONFIGURATION SETUP ===
    config_manager = ConfigManager()
    folder = operating_system_module.path.dirname(file_path_master)
    file_name = operating_system_module.path.splitext(operating_system_module.path.basename(file_path_master))[0]

    # Check for session config (from GUI)
    session_config_path = operating_system_module.environ.get('CALIBRATION_SESSION_CONFIG')
    if session_config_path and operating_system_module.path.exists(session_config_path):
        print(f"Using session config: {session_config_path}")
        with open(session_config_path, 'r') as f:
            config = json.load(f)
        print("[OK] Loaded session-specific configuration from GUI")
    else:
        # Create or load standard configuration
        config = config_manager.create_or_update_local_config(machine_id, file_path_master)
        print(f"Using standard config file: {operating_system_module.path.join(folder, 'config.json')}")

    print(f"Using machine: {machine_id} - {config.get('machine_name', machine_id)}")

    # === MATPLOTLIB CONFIGURATION ===
    # Check if we should show plots from config or GUI environment variable
    showPlot = config.get("showPlot", True)
    
    # Check for GUI environment variable override
    gui_show_plots = operating_system_module.environ.get('CALIBRATION_SHOW_PLOTS')
    if gui_show_plots is not None:
        showPlot = gui_show_plots.lower() == 'true'
        print(f"GUI override: showPlot = {showPlot}")
    
    print(f"Plot display setting: {showPlot}")
    
    # === EXTRACTION TYPE CONTROLS ===
    # Check for GUI boolean controls for 3 main extraction types
    extract_test_phase = operating_system_module.environ.get('CALIBRATION_USE_TEST_PHASE', 'true').lower() == 'true'
    extract_measurement = operating_system_module.environ.get('CALIBRATION_USE_MEASUREMENT', 'true').lower() == 'true'
    extract_3step = operating_system_module.environ.get('CALIBRATION_USE_3STEP_EXTRACTION', 'true').lower() == 'true'
    
    print(f"Extraction controls - Test Phase: {extract_test_phase}, Measurement: {extract_measurement}, 3-Step: {extract_3step}")
    
    # Import and configure matplotlib based on showPlot setting
    import matplotlib
    matplotlib.use('TkAgg')
    
    import matplotlib.pyplot as plt
    
    if showPlot:
        plt.ion()  # Turn on interactive mode for plot display
        print("Interactive plotting enabled")
    else:
        plt.ioff()  # Turn off interactive mode
        print("Plotting disabled - figures will only be saved")


    # === LOAD CONFIG VALUES ===
    # Use environment variable file path if available (from GUI), otherwise use config file path
    file_path = operating_system_module.environ.get('CALIBRATION_FILE_PATH', config["file_path"])
    startValue = config["startValue"]
    shift = config["shift"]
    channel = config["channel"]
    MVG = config["MVG"]
    thr = config["thr"]
    offset = config["offset"]
    number_loads = config["number_loads"]
    number_load_steps = config["number_load_steps"]
    number_load_cycles = config["number_load_cycles"]
    step_zero = config["step_zero"]
    step_max = config["step_max"]

    # === LABVIEW OVERRIDES ===
    # Check for LabVIEW-specific parameter overrides
    if operating_system_module.environ.get('LABVIEW_OVERRIDE_SAMPLE_START') == 'true':
        labview_start = operating_system_module.environ.get('LABVIEW_SAMPLE_START')
        if labview_start:
            startValue = int(float(labview_start))  # Convert through float to handle decimals like '85000.0'
            print(f"LabVIEW override: startValue = {startValue}")

    if operating_system_module.environ.get('LABVIEW_OVERRIDE_SHIFT') == 'true':
        labview_shift = operating_system_module.environ.get('LABVIEW_SHIFT')
        if labview_shift:
            shift = int(float(labview_shift))  # Convert through float to handle decimals
            print(f"LabVIEW override: shift = {shift}")

    if operating_system_module.environ.get('LABVIEW_OVERRIDE_MVG') == 'true':
        labview_mvg = operating_system_module.environ.get('LABVIEW_MVG')
        if labview_mvg:
            MVG = int(float(labview_mvg))  # Convert through float to handle decimals
            print(f"LabVIEW override: MVG = {MVG}")

    if operating_system_module.environ.get('LABVIEW_OVERRIDE_THR') == 'true':
        labview_thr = operating_system_module.environ.get('LABVIEW_THR')
        if labview_thr:
            thr = int(float(labview_thr))  # Convert through float to handle decimals
            print(f"LabVIEW override: thr = {thr}")

    print(f"Configuration loaded for {config['machine_name']}")
    print(f"Key parameters: startValue={startValue}, MVG={MVG}, thr={thr}, shift={shift}")
    print(f"Using file path: {file_path}")

    # === CREATE PLOTS FOLDER ===
    folder = operating_system_module.path.dirname(file_path)
    plots_folder = operating_system_module.path.join(folder, "Plots_Raw_Step")
    
    # Create plots folder if it doesn't exist
    if not operating_system_module.path.exists(plots_folder):
        operating_system_module.makedirs(plots_folder)
        print(f"Created plots folder: {plots_folder}")
    else:
        print(f"Using existing plots folder: {plots_folder}")

    # regular code
    operating_system_module.chdir(operating_system_module.path.dirname(file_path))

    # Load data
    data = pd.read_csv(file_path, names=['Index', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'Vref'], header=0)
    file_name = operating_system_module.path.basename(file_path).replace('.csv', '')
    values_c = data.to_numpy()

    # === MANUAL START INDEX CONFIGURATION ===
    print("\n=== MEASUREMENT PHASE START CONFIGURATION ===")
    print(f"Current startValue from config: {startValue}")
    print(f"Machine: {config['machine_name']}")

    # Check if running from GUI (non-interactive mode)
    running_from_gui = operating_system_module.environ.get('CALIBRATION_MACHINE_ID') is not None

    if running_from_gui:
        # Running from GUI - use config values automatically
        print(f"Running from GUI - using config startValue: {startValue}")
    else:
        # Running standalone - allow manual override
        manual_override = input(f"Use current startValue ({startValue}) or enter new value? (press Enter for current, or type new value): ").strip()

        if manual_override and manual_override.isdigit():
            startValue = int(manual_override)
            print(f"Using manual override startValue: {startValue}")
            
            # Ask if user wants to save this to config
            save_to_config = input("Save this startValue to machine config for future use? (y/n): ").strip().lower()
            if save_to_config == 'y':
                config["startValue"] = startValue
                print(f"Saved startValue {startValue} to config")
        else:
            print(f"Using config startValue: {startValue}")

    print(f"Processing data starting from sample index: {startValue}")

    # Apply the start value and process data
    values_c = values_c[startValue:np.size(values_c, 0), :]
    zero_value = np.mean(values_c[1:100, 1:7], 0)
    values_c[:, 1:7] = values_c[:, 1:7] - zero_value

    # Save updated config
    config_path = operating_system_module.path.join(folder, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Plot the processed data
    print("Preparing data visualization...")
    
    # Always save processed data to CSV (useful for GUI and standalone)
    processed_data_path = operating_system_module.path.join(folder, f"{file_name}_processed.csv")
    processed_df = pd.DataFrame({
        'Index': range(len(values_c)),
        'CH1': values_c[:, 2],
        'CH2': values_c[:, 1], 
        'CH3': values_c[:, 3],
        'CH4': values_c[:, 4],
        'CH5': values_c[:, 6],
        'CH6': values_c[:, 5]
    })
    processed_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to: {processed_data_path}")
    
    # Create and display plot based on showPlot setting
    plt.figure(figsize=(12, 8))
    plt.plot(values_c[:, [1]], label='CH2')
    plt.plot(values_c[:, [2]], label='CH1')
    plt.plot(values_c[:, [3]], label='CH3')
    plt.plot(values_c[:, [4]], label='CH4')
    plt.plot(values_c[:, [5]], label='CH6')
    plt.plot(values_c[:, [6]], label='CH5')
    plt.title(f'Raw Data from {file_name} - {config["machine_name"]} (Start: {startValue})')
    plt.xlabel('Sample Index (from measurement phase start)')
    plt.ylabel('Sensor Values')
    plt.legend()
    plt.grid(True)
    
    # Save plot and show if configured
    plt.savefig(operating_system_module.path.join(plots_folder, f'raw_data_{file_name}.png'), dpi=300, bbox_inches='tight')
    print(f"Raw data plot saved to: Plots_Raw_Step/raw_data_{file_name}.png")
    
    if showPlot:
        print("Displaying raw data plot...")
        plt.show(block=True)
    else:
        print("Plot display disabled - figure closed")
        plt.close()  # Close the figure to free memory

    # Load weight step configuration from config
    gewichtsstufen = config.get("gewichtsstufen", [0,1,2,3,4,5,4,3,2,1])
    print(f"Using weight steps: {gewichtsstufen}")

    raw_data = np.sum(np.abs(values_c[:, 1:7]), 1)
    a = raw_data
    shift1 = len(a) - shift

    a1 = a[0:(len(a)-2*shift)]
    a2 = a[2*shift:len(a)]
    a = a[shift:shift1]
    k = np.abs(a1 - a2)

    df_k = pd.DataFrame(k, columns=['k'])
    rolling_mean = df_k.k.rolling(window=MVG).mean()
    k = rolling_mean.to_numpy()
    k[0:MVG] = 0
    # plt.figure()
    # plt.plot(k)
    # plt.title("Smoothed k")
    # plt.savefig(operating_system_module.path.join(plots_folder, f'smoothed_k_{file_name}.png'), dpi=300, bbox_inches='tight')
    # print(f"Smoothed k plot saved to: Plots_Raw_Step/smoothed_k_{file_name}.png")
    
    if showPlot:
        print("Displaying smoothed k plot...")
        plt.show(block=True)
    else:
        print("Plot display disabled - figure closed")
        plt.close()  # Close the figure to free memory

    k = thr * (((k > thr) + 1) - 1)
    k1 = k[1:]
    k2 = k[:-1]

    falling_edges = np.logical_and(k2 == thr, k1 == 0)
    k2_shifted = k2.copy()

    for idx in np.nonzero(falling_edges)[0]:
        if idx + offset < len(k2_shifted):
            k2_shifted[idx] = 0
            k2_shifted[idx + offset] = thr

    k1 = np.append(0, k1)
    k2_shifted = np.append(thr, k2_shifted)
    k = np.int32(np.logical_xor(k1, k2_shifted))
    #
    # plt.figure()
    # plt.plot(3 * k)
    # plt.plot(raw_data)
    # plt.title("Step end with shape offset")
    # plt.savefig(operating_system_module.path.join(plots_folder, f'step_end_{file_name}.png'), dpi=300, bbox_inches='tight')
    # print(f"Step end plot saved to: Plots_Raw_Step/step_end_{file_name}.png")
    
    if showPlot:
        print("Displaying step end plot...")
        plt.show(block=True)
    else:
        print("Plot display disabled - figure closed")
        plt.close()  # Close the figure to free memory

    el = (np.asarray(np.nonzero(k == 1)))
    el = np.append(el, el[0, np.size(el, 1)-1] + 500)

    v0 = np.zeros([number_loads, np.int32(number_load_steps * number_load_cycles + 1)])
    values = [0]

    str_name_data = ['Fx_Step','Fy_Step','Fz_Step','Mx_Step','My_Step','Mz_Step']
    str_name_test = ['Fx_test','Fy_test','Fz_test','Mx_test','My_test','Mz_test']
    str_name_calibration = ['Fx_C2','Fy_C2','Fz_C2','Mx_C2','My_C2','Mz_C2']
    mean_values = np.zeros((6, 6, 50))

    # Define test phase parameters (needed for offset calculations regardless of extraction)
    test_cycles_per_axis = 1  # Each axis runs once in test phase
    test_steps_per_cycle = number_load_steps  # Same number of steps as measurement phase

    # === EXTRACT INITIAL TEST CYCLE DATA ===
    if extract_test_phase:
        print("\n=== EXTRACTING INITIAL TEST CYCLE DATA ===")
        print("The initial test cycle happens before the main measurement phase")
        
        # Extract test cycle data for each axis (6 axes)
        for axis in range(0, number_loads):  # 6 axes: Fx, Fy, Fz, Mx, My, Mz
            # Calculate the position in the step detection array for this test axis
            test_left_element = axis * test_cycles_per_axis * test_steps_per_cycle * 2 * 2
            test_right_element = (axis + 1) * test_cycles_per_axis * test_steps_per_cycle * 2 * 2
            
            # Get the step boundaries for this test axis
            test_left_limit = el[test_left_element]
            test_right_limit = el[test_right_element]
            
            print(f"Test cycle {str_name_test[axis]}: samples {test_left_limit} to {test_right_limit}")
            
            # Calculate zero value for this test cycle (using first few samples)
            test_zero_samples = min(100, test_right_limit - test_left_limit)
            test_zero_value = np.mean(values_c[test_left_limit:test_left_limit + test_zero_samples, 1:7], 0)
            
            # Extract and process test cycle data
            test_temp = values_c[test_left_limit:test_right_limit, :].copy()
            test_temp[:, 1:7] -= test_zero_value
            
            # Save test cycle data to CSV
            df_test_temp = pd.DataFrame(data=test_temp, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
            df_test_temp.to_csv(str_name_test[axis] + '.csv')
            
            print(f"[OK] Saved test cycle data: {str_name_test[axis]}.csv ({len(test_temp)} samples)")
            
           # Plot test cycle data if enabled
            plt.figure(figsize=(12, 6))
            plt.plot(np.arange(test_left_limit, test_right_limit), test_temp[:, 1:7])
            plt.title(f'Test Cycle Data - {str_name_test[axis]} - {config["machine_name"]}')
            plt.xlabel('Sample Index')
            plt.ylabel('Sensor Values (Zero-corrected)')
            plt.legend(['fX', 'fY', 'fZ', 'mX', 'mY', 'mZ'])
            plt.grid(True)
            plt.savefig(operating_system_module.path.join(plots_folder, f'test_cycle_{str_name_test[axis]}_{file_name}.png'), dpi=300, bbox_inches='tight')
            print(f"Test cycle plot saved: Plots_Raw_Step/test_cycle_{str_name_test[axis]}_{file_name}.png")
            
            if showPlot:
                print(f"Displaying test cycle plot for {str_name_test[axis]}...")
                plt.show(block=True)
            else:
                print("Plot display disabled - figure closed")
                plt.close()  # Close the figure to free memory

        print(f"[OK] All 6 test cycle files saved successfully!")
    else:
        print("\n[SKIP] Test cycle extraction disabled by GUI settings")
    
    # === EXTRACT MAIN MEASUREMENT CYCLE DATA ===
    # Calculate offset for main measurement phase - only if test phase extraction was enabled
    if extract_test_phase:
        test_phase_total_steps = number_loads * test_cycles_per_axis * test_steps_per_cycle * 2 * 2
        print(f"Test phase was extracted, measurement data starts after {test_phase_total_steps} steps")
    else:
        test_phase_total_steps = 0  # Start from beginning of data when no test phase
        print("Test phase extraction disabled, measurement data starts from beginning")
    
    if extract_measurement:
        print("\n=== EXTRACTING MAIN MEASUREMENT CYCLE DATA ===")
        
        for x in range(0,number_loads):
            # Calculate element positions - start from test_phase_total_steps offset (0 if no test phase)
            left_element = test_phase_total_steps + x * number_load_cycles * number_load_steps * 2 * 2
            el_detail = el[left_element:(test_phase_total_steps + (x+1)*number_load_cycles*number_load_steps*2*2) + 1]

            left_limit = el[left_element]
            right_limit = el[test_phase_total_steps + (x+1)*number_load_cycles*number_load_steps*2*2 + 1]
            print(f"Measurement cycle {str_name_data[x]}: samples {left_limit} to {right_limit}")

            for xx in range(0, np.int32((np.size(el_detail)-1)), 2):
                b1 = el_detail[xx]
                b2 = el_detail[xx + 1]
                raw_step_values = values_c[b1:b2, 1:7]
                mean_values[x, :, np.int32(xx/2)] = np.mean(raw_step_values, 0)

            max_left_limit = el[left_element + ((1 + 2 * (step_max - 1)) * 2 * number_load_steps)]
            max_right_limit = el[left_element + ((1 + 2 * (step_max - 1)) * 2 * number_load_steps + 1)]

            zero_left_limit = el[left_element + (step_zero - 1) * 2 * 2 * number_load_steps]
            zero_right_limit = el[left_element + 1 + (step_zero - 1) * 2 * 2 * number_load_steps]

            zero_value = np.mean(values_c[zero_left_limit:zero_right_limit, 1:7], 0)
            if showPlot:
                plt.figure()
                plt.plot(np.arange(left_limit, right_limit), values_c[left_limit:right_limit, 1:7] - zero_value)
                plt.plot(np.arange(left_limit, right_limit), 100000 * k[left_limit:right_limit])
                plt.plot(np.arange(max_left_limit, max_right_limit) - 7000, values_c[max_left_limit:max_right_limit, 1:7] - zero_value)
                plt.plot(np.arange(zero_left_limit, zero_right_limit) - 7000, values_c[zero_left_limit:zero_right_limit, 1:7] - zero_value)
                plt.title("Step Analysis")
                plt.savefig(operating_system_module.path.join(plots_folder, f'step_analysis_{file_name}.png'), dpi=300, bbox_inches='tight')
                print(f"Step analysis plot saved: Plots_Raw_Step/step_analysis_{file_name}.png")
                plt.show(block=True)
            else:
                plt.close()  

            # Save full measurement cycle data (10 steps: 0,1,2,3,4,5,4,3,2,1,0)
            temp = values_c[left_limit:right_limit, :].copy()
            temp[:, 1:7] -= zero_value
            df_temp = pd.DataFrame(data=temp, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
            df_temp.to_csv(str_name_data[x] + '.csv')
            print(f"[OK] Saved full measurement data: {str_name_data[x]}.csv ({len(temp)} samples)")

            # Save calibration data (uses step 5 - maximum load)
            temp2 = values_c[max_left_limit:max_right_limit, :].copy()
            temp2[:, 1:7] -= zero_value
            df_temp2 = pd.DataFrame(data=temp2, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
            df_temp2.to_csv(str_name_calibration[x] + '.csv')
            print(f"[OK] Saved calibration data: {str_name_calibration[x]}.csv ({len(temp2)} samples)")

            # === EXTRACT 3-STEP DATA (0,3,5,3,0) FROM FULL MEASUREMENT CYCLE ===
            if extract_3step:
                print(f"Extracting 3-step data for {str_name_data[x]}...")
                print(f"  el_detail length: {len(el_detail)} (should contain all {number_load_cycles} cycles)")
                
                # Weight step sequence in practice: [0,1,2,3,4,5,4,3,2,1] (indices 0-9, 10 steps per cycle)  
                # Use 4 complete cycles plus step 0 of cycle 5 to complete [0,3,5,3,0] pattern
                # Weight sequence per cycle: [0,1,2,3,4,5,4,3,2,1] 
                # Extract [0,3,5,3,0] by taking step 0 of cycle 5 as the final baseline
                target_step_indices = [0, 3, 5, 7]  # Plus step 0 of cycle 5 for final baseline
                
                # Extract data for the 3-step sequence - SIMPLE approach
                temp_3step_data = []
                
                # Calculate the ACTUAL steps per cycle from the data (this was working)
                total_steps = len(el_detail) // 2  # Each step has 2 elements (start, end)
                actual_steps_per_cycle = total_steps // number_load_cycles  # Use actual measured cycle length
                
                print(f"  Total steps in el_detail: {total_steps}")
                print(f"  Actual steps per cycle: {actual_steps_per_cycle}")
                print(f"  Expected cycles: {number_load_cycles}")
                print(f"  Target weight pattern [0,3,5,3,0] maps to indices: {target_step_indices}")
            
                # Extract target steps from cycles 1-4, then get step 0 from cycle 5
                for xx in range(0, np.int32((np.size(el_detail)-1)), 2):
                    overall_step = xx // 2
                    cycle_num = overall_step // actual_steps_per_cycle + 1
                    step_in_cycle = overall_step % actual_steps_per_cycle
                    
                    # Extract steps [0,3,5,7] from cycles 1-4, plus step 0 from cycle 5
                    should_extract = (cycle_num <= 4 and step_in_cycle in target_step_indices) or (cycle_num == 5 and step_in_cycle == 0)  # First step of 5th cycle
                        
                    if should_extract and xx + 1 < len(el_detail):
                        b1 = el_detail[xx]
                        b2 = el_detail[xx + 1]
                        step_data = values_c[b1:b2, :].copy()
                        step_data[:, 1:7] -= zero_value
                        temp_3step_data.append(step_data)
                        if cycle_num == 5:
                            print(f"    Cycle {cycle_num}, Step {step_in_cycle} (final baseline): samples {b1} to {b2}")
                        else:
                            print(f"    Cycle {cycle_num}, Step {step_in_cycle}: samples {b1} to {b2}")
                    elif should_extract:
                        print(f"    Missing: Cycle {cycle_num}, Step {step_in_cycle} (step xx={xx}, xx+1={xx+1}, len={len(el_detail)})")
            
                # Using 4 cycles plus step 0 of cycle 5 gives complete [0,3,5,3,0] pattern
                        
                if temp_3step_data:
                    # Combine all 3-step data
                    temp_3step = np.vstack(temp_3step_data)
                    df_temp_3step = pd.DataFrame(data=temp_3step, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
                    
                    # Create 3-step filename
                    step_3_name = str_name_data[x].replace('_Step', '_3Step')  # Fx_Step -> Fx_3Step
                    df_temp_3step.to_csv(step_3_name + '.csv')
                    print(f"[OK] Saved 3-step data: {step_3_name}.csv ({len(temp_3step)} samples)")
                    
                    # Also save calibration data with 3Step naming (same data, step 5)
                    calib_3_name = str_name_calibration[x].replace('_C2', '_3Step_C2')  # Fx_C2 -> Fx_3Step_C2
                    # Create df_temp2 for 3-step (temp2 should be available from measurement or calculated separately)
                    df_temp2_3step = pd.DataFrame(data=temp2, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
                    df_temp2_3step.to_csv(calib_3_name + '.csv')
                    print(f"[OK] Saved 3-step calibration: {calib_3_name}.csv ({len(temp2)} samples)")
                
                    # Plot 3-step data if enabled
                    plt.figure(figsize=(12, 8))
                    plt.plot(temp_3step[:, 1:7])
                    plt.title(f'3-Step Data (0,3,5,3,0) - {step_3_name} - {config["machine_name"]}')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Sensor Values (Zero-corrected)')
                    plt.legend(['fX', 'fY', 'fZ', 'mX', 'mY', 'mZ'])
                    plt.grid(True)
                    plt.savefig(operating_system_module.path.join(plots_folder, f'3step_analysis_{step_3_name}_{file_name}.png'), dpi=300, bbox_inches='tight')
                    print(f"3-step analysis plot saved: Plots_Raw_Step/3step_analysis_{step_3_name}_{file_name}.png")
                    
                    if showPlot:
                        print(f"Displaying 3-step analysis for {step_3_name}...")
                        plt.show(block=True)
                    else:
                        print("Plot display disabled - figure closed")
                        plt.close()  # Close the figure to free memory
                else:
                    print(f"[WARNING] Could not extract 3-step data for {str_name_data[x]} - insufficient steps")
            else:
                print(f"[SKIP] 3-step data extraction disabled for {str_name_data[x]}")
            
            # Final analysis plot (only when measurement extraction is enabled)
            if showPlot:
                plt.figure()
                plt.plot(temp2[:, 1:7])
                plt.title("Final Analysis")
                plt.savefig(operating_system_module.path.join(plots_folder, f'final_analysis_{file_name}.png'), dpi=300, bbox_inches='tight')
                print(f"Final analysis plot saved: Plots_Raw_Step/final_analysis_{file_name}.png")
                plt.show(block=True)
            else:
                plt.close()  # Close the figure to free memory
    else:
        # Measurement extraction disabled, but we still need to calculate data for 3-step processing if enabled
        if extract_3step:
            print("\n[SKIP] Measurement extraction disabled, but calculating calibration data for 3-step processing...")
            
            for x in range(0,number_loads):
                # Calculate positions for calibration data (step 5 - maximum load) even without measurement extraction
                left_element = test_phase_total_steps + x * number_load_cycles * number_load_steps * 2 * 2
                max_left_limit = el[left_element + ((1 + 2 * (step_max - 1)) * 2 * number_load_steps)]
                max_right_limit = el[left_element + ((1 + 2 * (step_max - 1)) * 2 * number_load_steps + 1)]
                zero_left_limit = el[left_element + (step_zero - 1) * 2 * 2 * number_load_steps]
                zero_right_limit = el[left_element + 1 + (step_zero - 1) * 2 * 2 * number_load_steps]
                
                zero_value = np.mean(values_c[zero_left_limit:zero_right_limit, 1:7], 0)
                temp2 = values_c[max_left_limit:max_right_limit, :].copy()
                temp2[:, 1:7] -= zero_value
                # temp2 is now available for 3-step processing
        else:
            print("\n[SKIP] Measurement extraction disabled by GUI settings")

    # === DETECT AND EXTRACT SECOND MEASUREMENT PHASE ===
    print("\n=== CHECKING FOR SECOND MEASUREMENT PHASE ===")
    
    # Calculate where the first measurement phase ends
    first_phase_total_steps = test_phase_total_steps + (number_loads * number_load_cycles * number_load_steps * 2 * 2)
    
    # Check if there are enough remaining steps for a second measurement phase
    remaining_steps = len(el) - first_phase_total_steps - 1
    expected_second_phase_steps = number_loads * number_load_cycles * number_load_steps * 2 * 2
    
    print(f"First phase ends at step index: {first_phase_total_steps}")
    print(f"Remaining steps available: {remaining_steps}")
    print(f"Second phase would need: {expected_second_phase_steps} steps")
    
    if remaining_steps >= expected_second_phase_steps:
        print("[OK] Second measurement phase detected! Extracting data...")
        
        # Extract second measurement phase data
        str_name_data_2 = ['Fx_Step_2','Fy_Step_2','Fz_Step_2','Mx_Step_2','My_Step_2','Mz_Step_2']
        str_name_calibration_2 = ['Fx_C2_2','Fy_C2_2','Fz_C2_2','Mx_C2_2','My_C2_2','Mz_C2_2']
        
        for x in range(0, number_loads):
            # Calculate offsets for second measurement phase
            left_element_2 = first_phase_total_steps + x * number_load_cycles * number_load_steps * 2 * 2
            el_detail_2 = el[left_element_2:(first_phase_total_steps + (x+1)*number_load_cycles*number_load_steps*2*2) + 1]

            left_limit_2 = el[left_element_2]
            right_limit_2 = el[first_phase_total_steps + (x+1)*number_load_cycles*number_load_steps*2*2 + 1]
            print(f"Second measurement cycle {str_name_data_2[x]}: samples {left_limit_2} to {right_limit_2}")

            # Calculate zero reference for second phase
            max_left_limit_2 = el[left_element_2 + ((1 + 2 * (step_max - 1)) * 2 * number_load_steps)]
            max_right_limit_2 = el[left_element_2 + ((1 + 2 * (step_max - 1)) * 2 * number_load_steps + 1)]

            zero_left_limit_2 = el[left_element_2 + (step_zero - 1) * 2 * 2 * number_load_steps]
            zero_right_limit_2 = el[left_element_2 + 1 + (step_zero - 1) * 2 * 2 * number_load_steps]

            zero_value_2 = np.mean(values_c[zero_left_limit_2:zero_right_limit_2, 1:7], 0)
            
            # Plot second phase analysis if enabled
            if showPlot:
                plt.figure()
                plt.plot(np.arange(left_limit_2, right_limit_2), values_c[left_limit_2:right_limit_2, 1:7] - zero_value_2)
                plt.plot(np.arange(left_limit_2, right_limit_2), 100000 * k[left_limit_2:right_limit_2])
                plt.plot(np.arange(max_left_limit_2, max_right_limit_2) - 7000, values_c[max_left_limit_2:max_right_limit_2, 1:7] - zero_value_2)
                plt.plot(np.arange(zero_left_limit_2, zero_right_limit_2) - 7000, values_c[zero_left_limit_2:zero_right_limit_2, 1:7] - zero_value_2)
                plt.title(f"Second Phase Step Analysis - {str_name_data_2[x]}")
                plt.savefig(operating_system_module.path.join(plots_folder, f'step_analysis_2_{file_name}_{str_name_data_2[x]}.png'), dpi=300, bbox_inches='tight')
                print(f"Second phase step analysis plot saved: Plots_Raw_Step/step_analysis_2_{file_name}_{str_name_data_2[x]}.png")
                
                if showPlot:
                    print(f"Displaying second phase step analysis for {str_name_data_2[x]}...")
                    plt.show(block=True)
                else:
                    print("Plot display disabled - figure closed")
                    plt.close()

            # Save second phase full step data (10 steps: 0,1,2,3,4,5,4,3,2,1,0)
            temp_2 = values_c[left_limit_2:right_limit_2, :].copy()
            temp_2[:, 1:7] -= zero_value_2
            df_temp_2 = pd.DataFrame(data=temp_2, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
            df_temp_2.to_csv(str_name_data_2[x] + '.csv')
            print(f"[OK] Saved second phase full step data: {str_name_data_2[x]}.csv ({len(temp_2)} samples)")

            # Save second phase calibration data (uses step 5 - maximum load)
            temp2_2 = values_c[max_left_limit_2:max_right_limit_2, :].copy()
            temp2_2[:, 1:7] -= zero_value_2
            df_temp2_2 = pd.DataFrame(data=temp2_2, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
            df_temp2_2.to_csv(str_name_calibration_2[x] + '.csv')
            print(f"[OK] Saved second phase calibration data: {str_name_calibration_2[x]}.csv ({len(temp2_2)} samples)")

            # === EXTRACT 3-STEP DATA (0,3,5,3,0) FROM SECOND PHASE MEASUREMENT CYCLE ===
            print(f"Extracting 3-step data for {str_name_data_2[x]}...")
            
            # Weight step sequence in practice: [0,1,2,3,4,5,4,3,2,1] (indices 0-9)
            # We want weight steps: [0,3,5,3,0] which map to indices [0,3,4,5,9]
            target_step_indices_2 = [0, 3, 4, 5, 10]  # Corrected pattern for [0,3,5,3,0]
            
            # Extract data for the 3-step sequence from second phase
            temp_3step_data_2 = []
            
            for step_idx in target_step_indices_2:
                if step_idx * 2 < len(el_detail_2) - 1:  # Check if step exists
                    step_start_2 = el_detail_2[step_idx * 2]
                    step_end_2 = el_detail_2[step_idx * 2 + 1] 
                    step_data_2 = values_c[step_start_2:step_end_2, :].copy()
                    step_data_2[:, 1:7] -= zero_value_2
                    temp_3step_data_2.append(step_data_2)
                    
            if temp_3step_data_2:
                # Combine all 3-step data from second phase
                temp_3step_2 = np.vstack(temp_3step_data_2)
                df_temp_3step_2 = pd.DataFrame(data=temp_3step_2, columns=['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
                
                # Create 3-step filename for second phase
                step_3_name_2 = str_name_data_2[x].replace('_Step_2', '_3Step_2')  # Fx_Step_2 -> Fx_3Step_2
                df_temp_3step_2.to_csv(step_3_name_2 + '.csv')
                print(f"[OK] Saved second phase 3-step data: {step_3_name_2}.csv ({len(temp_3step_2)} samples)")
                
                # Also save calibration data with 3Step naming for second phase (same data, step 5)
                calib_3_name_2 = str_name_calibration_2[x].replace('_C2_2', '_3Step_C2_2')  # Fx_C2_2 -> Fx_3Step_C2_2
                df_temp2_2.to_csv(calib_3_name_2 + '.csv')
                print(f"[OK] Saved second phase 3-step calibration: {calib_3_name_2}.csv ({len(temp2_2)} samples)")
                
                # Plot second phase 3-step data if enabled
                plt.figure(figsize=(12, 8))
                plt.plot(temp_3step_2[:, 1:7])
                plt.title(f'Second Phase 3-Step Data (0,3,5,3,0) - {step_3_name_2} - {config["machine_name"]}')
                plt.xlabel('Sample Index')
                plt.ylabel('Sensor Values (Zero-corrected)')
                plt.legend(['fX', 'fY', 'fZ', 'mX', 'mY', 'mZ'])
                plt.grid(True)
                plt.savefig(operating_system_module.path.join(plots_folder, f'3step_analysis_2_{step_3_name_2}_{file_name}.png'), dpi=300, bbox_inches='tight')
                print(f"Second phase 3-step analysis plot saved: Plots_Raw_Step/3step_analysis_2_{step_3_name_2}_{file_name}.png")
                
                if showPlot:
                    print(f"Displaying second phase 3-step analysis for {step_3_name_2}...")
                    plt.show(block=True)
                else:
                    print("Plot display disabled - figure closed")
                    plt.close()  # Close the figure to free memory
            else:
                print(f"[WARNING] Could not extract 3-step data for {str_name_data_2[x]} - insufficient steps")
            
            # Plot second phase final analysis
            if showPlot:
                plt.figure()
                plt.plot(temp2_2[:, 1:7])
                plt.title(f"Second Phase Final Analysis - {str_name_data_2[x]}")
                plt.savefig(operating_system_module.path.join(plots_folder, f'final_analysis_2_{file_name}_{str_name_data_2[x]}.png'), dpi=300, bbox_inches='tight')
                print(f"Second phase final analysis plot saved: Plots_Raw_Step/final_analysis_2_{file_name}_{str_name_data_2[x]}.png")
                
                if showPlot:
                    print(f"Displaying second phase final analysis for {str_name_data_2[x]}...")
                    plt.show(block=True)
                else:
                    print("Plot display disabled - figure closed")
                    plt.close()

        print("[OK] Second measurement phase extraction completed!")
        
    else:
        print("[INFO] No second measurement phase detected - insufficient remaining data")
        str_name_data_2 = []
        str_name_calibration_2 = []

    # Plot error bars
    # for load_steps in range(6):
    #     fig, ax = plt.subplots(6)
    #     for channel in range(6):
    #         zw_mean = np.mean(np.reshape(mean_values[load_steps, channel, :], (5, 10)), 0)
    #         zw_std = np.std(np.reshape(mean_values[load_steps, channel, :], (5, 10)), 0)
    #         x = np.array([0,1,2,3,4,5,4,3,2,1])
    #         a, b = np.polyfit(x, zw_mean, 1)
    #         zw2 = np.polyval([a, b], x)
    #         ax[channel].errorbar(x, zw2 - zw_mean, zw_std, capsize=4, ecolor='k')
    #         ax[channel].set_ylim(-2000, 2000)
    #     plt.show()

    print("\n=== ANALYSIS COMPLETED ===")
    if running_from_gui:
        print("[SUCCESS] Analysis completed successfully! Check the GUI for results.")
    else:
        print("[SUCCESS] Analysis completed successfully!")
    print(f"Results saved in: {folder}")
    print("\nExtracted data files:")
    print("Test cycle data (initial phase):")
    for test_file in str_name_test:
        print(f"  [OK] {test_file}.csv")
    
    print("Main measurement phase data (10 steps: 0,1,2,3,4,5,4,3,2,1,0):")
    for step_file in str_name_data[:6]:  # Only show first 6 (original measurement phase)
        print(f"  [OK] {step_file}.csv")
    
    print("Main measurement phase data (3 steps: 0,3,5,3,0):")
    for step_file in str_name_data[:6]:
        step_3_name = step_file.replace('_Step', '_3Step')
        print(f"  [OK] {step_3_name}.csv")
    
    print("Main phase calibration data:")
    for calib_file in str_name_calibration:
        print(f"  [OK] {calib_file}.csv")
        # Also show 3-step calibration
        calib_3_name = calib_file.replace('_C2', '_3Step_C2')
        print(f"  [OK] {calib_3_name}.csv")
    
    # Show second phase files if they exist
    if 'str_name_data_2' in locals() and str_name_data_2:
        print("Second measurement phase data (10 steps: 0,1,2,3,4,5,4,3,2,1,0):")
        for step_file_2 in str_name_data_2:
            print(f"  [OK] {step_file_2}.csv")
        
        print("Second measurement phase data (3 steps: 0,3,5,3,0):")
        for step_file_2 in str_name_data_2:
            step_3_name_2 = step_file_2.replace('_Step_2', '_3Step_2')
            print(f"  [OK] {step_3_name_2}.csv")
        
        print("Second phase calibration data:")
        for calib_file_2 in str_name_calibration_2:
            print(f"  [OK] {calib_file_2}.csv")
            # Also show 3-step calibration for second phase
            calib_3_name_2 = calib_file_2.replace('_C2_2', '_3Step_C2_2')
            print(f"  [OK] {calib_3_name_2}.csv")


if __name__ == "__main__":
    main()