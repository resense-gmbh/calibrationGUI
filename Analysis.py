import os
import numpy as np
import pandas as pd
from cycler import cycler
import json
import traceback
import sys
from config_manager import ConfigManager

def setup_matplotlib():
    """Setup matplotlib with interactive backend for showing plots"""
    import matplotlib
    matplotlib.use('TkAgg')  # Use interactive backend
    import matplotlib.pyplot as plt
    plt.ion()  # Turn on interactive mode
    from pylab import double
    return plt, double

def get_machine_and_file():
    """Get machine and file from environment variables or fallback to defaults"""
    machine_id = os.environ.get('CALIBRATION_MACHINE_ID')
    file_path_master = os.environ.get('CALIBRATION_FILE_PATH')
    
    if machine_id and file_path_master:
        print("Using parameters from launcher:")
        print(f"Machine: {machine_id}")
        print(f"File: {file_path_master}")
        return machine_id, file_path_master
    
    # Fallback to direct input if not launched from main script
    print("No launcher parameters found. Running in standalone mode.")
    
    # Direct machine specification
    machine_id = "CM2"  # Change this to CM1, CM2, CM3, CM4, or CM5
    
    # Direct file path - UPDATE THIS PATH FOR YOUR DATA
    file_path_master = r"C:\Users\SahayaJ\OneDrive - WIKA\Desktop\Projects\KalPla_Analysis_Master\Fx_Step.csv"
    
    if not os.path.exists(file_path_master):
        print(f"File not found: {file_path_master}")
        print("Please update the file_path_master variable in the script or use the main launcher.")
        exit(1)
    
    return machine_id, file_path_master

def main():
    """Main analysis function - wraps all executable code"""
    try:
        _main_analysis()
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        # Re-raise the exception so the GUI can catch it
        raise

def _main_analysis():
    """Internal main analysis function"""
    # === GET MACHINE AND FILE ===
    machine_id, file_path_master = get_machine_and_file()

    # === CONFIGURATION SETUP ===
    config_manager = ConfigManager()
    folder = os.path.dirname(file_path_master)
    file_name = os.path.splitext(os.path.basename(file_path_master))[0]

    # Create or load standard configuration
    config = config_manager.create_or_update_local_config(machine_id, file_path_master)
    print(f"Using standard config file: {os.path.join(folder, 'config.json')}")

    print(f"Using machine: {machine_id} - {config.get('machine_name', machine_id)}")

    # Check if running from GUI (non-interactive mode)
    running_from_gui = os.environ.get('CALIBRATION_MACHINE_ID') is not None

    ####
    ###Use these if you want to do individual analysis
    # Read analysis axis flags from environment variables (set by GUI) or use defaults
    FzAnalysis = os.environ.get('CALIBRATION_FZ_ANALYSIS', 'True').lower() == 'true'
    FxAnalysis = os.environ.get('CALIBRATION_FX_ANALYSIS', 'False').lower() == 'true'  
    FyAnalysis = os.environ.get('CALIBRATION_FY_ANALYSIS', 'False').lower() == 'true'
    MzAnalysis = os.environ.get('CALIBRATION_MZ_ANALYSIS', 'False').lower() == 'true'
    MxAnalysis = os.environ.get('CALIBRATION_MX_ANALYSIS', 'False').lower() == 'true'
    MyAnalysis = os.environ.get('CALIBRATION_MY_ANALYSIS', 'False').lower() == 'true'
    FTAnalysis = os.environ.get('CALIBRATION_FT_ANALYSIS', 'True').lower() == 'true'
    
    # Read analysis type flags for Step vs 3Step analysis
    UseStepAnalysis = os.environ.get('CALIBRATION_USE_STEP_ANALYSIS', 'False').lower() == 'true'
    Use3StepAnalysis = os.environ.get('CALIBRATION_USE_3STEP_ANALYSIS', 'True').lower() == 'true'
    UseTestPhase = os.environ.get('CALIBRATION_USE_TEST_PHASE', 'False').lower() == 'true'

    #load the config values
    # Use environment variable file path if available (from GUI), otherwise use config file path
    file_path = os.environ.get('CALIBRATION_FILE_PATH', config["file_path"])
    fZ_shift=config["fZ_shift"]
    fZ_thr=config["fZ_thr"]
    fZ_MVG=config["fZ_MVG"]
    fZ_offset=config["fZ_offset"]
    
    fX_shift=config["fX_shift"]
    fX_thr=config["fX_thr"]
    fX_MVG=config["fX_MVG"]
    fX_offset=config["fX_offset"]
    
    fY_shift=config["fY_shift"]
    fY_thr=config["fY_thr"]
    fY_MVG=config["fY_MVG"]
    fY_offset=config["fY_offset"]
    
    mZ_shift=config["mZ_shift"]
    mZ_thr=config["mZ_thr"]
    mZ_MVG=config["mZ_MVG"]
    mZ_offset=config["mZ_offset"]
    
    mX_shift=config["mX_shift"]
    mX_thr=config["mX_thr"]
    mX_MVG=config["mX_MVG"]
    mX_offset=config["mX_offset"]
    
    mY_shift=config["mY_shift"]
    mY_thr=config["mY_thr"]
    mY_MVG=config["mY_MVG"]
    mY_offset=config["mY_offset"]

    # === 3-STEP SPECIFIC PARAMETERS ===
    fZ_shift_3step=config.get("fZ_shift_3step", fZ_shift)
    fZ_thr_3step=config.get("fZ_thr_3step", fZ_thr)
    fZ_MVG_3step=config.get("fZ_MVG_3step", fZ_MVG)
    fZ_offset_3step=config.get("fZ_offset_3step", fZ_offset)
    
    fX_shift_3step=config.get("fX_shift_3step", fX_shift)
    fX_thr_3step=config.get("fX_thr_3step", fX_thr)
    fX_MVG_3step=config.get("fX_MVG_3step", fX_MVG)
    fX_offset_3step=config.get("fX_offset_3step", fX_offset)
    
    fY_shift_3step=config.get("fY_shift_3step", fY_shift)
    fY_thr_3step=config.get("fY_thr_3step", fY_thr)
    fY_MVG_3step=config.get("fY_MVG_3step", fY_MVG)
    fY_offset_3step=config.get("fY_offset_3step", fY_offset)
    
    mZ_shift_3step=config.get("mZ_shift_3step", mZ_shift)
    mZ_thr_3step=config.get("mZ_thr_3step", mZ_thr)
    mZ_MVG_3step=config.get("mZ_MVG_3step", mZ_MVG)
    mZ_offset_3step=config.get("mZ_offset_3step", mZ_offset)
    
    mX_shift_3step=config.get("mX_shift_3step", mX_shift)
    mX_thr_3step=config.get("mX_thr_3step", mX_thr)
    mX_MVG_3step=config.get("mX_MVG_3step", mX_MVG)
    mX_offset_3step=config.get("mX_offset_3step", mX_offset)
    
    mY_shift_3step=config.get("mY_shift_3step", mY_shift)
    mY_thr_3step=config.get("mY_thr_3step", mY_thr)
    mY_MVG_3step=config.get("mY_MVG_3step", mY_MVG)
    mY_offset_3step=config.get("mY_offset_3step", mY_offset)
    
    showPlotInt_3step=config.get("showPlotInt_3step", False)

    # === WEIGHT CONFIGURATION ===
    # Get weight configurations with defaults
    fZ_weight_per_step = config.get("fZ_weight_per_step", 2500)  # Default 2500g per step
    fX_weight_per_step = config.get("fX_weight_per_step", 2500)  
    fY_weight_per_step = config.get("fY_weight_per_step", 2500)  
    
    # Moment weight configurations (chain weights and distances)
    mZ_chain1_weight_per_step = config.get("mZ_chain1_weight_per_step", 1250)  # Default 1250g
    mZ_chain2_weight_per_step = config.get("mZ_chain2_weight_per_step", 1250)  
    mZ_weight_distance = config.get("mZ_weight_distance", 100)  # Default 100mm
    
    mX_chain1_weight_per_step = config.get("mX_chain1_weight_per_step", 1250)  
    mX_chain2_weight_per_step = config.get("mX_chain2_weight_per_step", 1250)  
    mX_weight_distance = config.get("mX_weight_distance", 100)  
    
    mY_chain1_weight_per_step = config.get("mY_chain1_weight_per_step", 1250)  
    mY_chain2_weight_per_step = config.get("mY_chain2_weight_per_step", 1250) 
    mY_weight_distance = config.get("mY_weight_distance", 100)  

    # === DISPLAY CONFIGURATION ===
    showPlotInt = True  # Changed to False for non-blocking execution
    showPlot = False     # Changed to False for non-blocking execution

    # === CALCULATE SENSOR MATRIX ===
    import calculateMatrix
    import analysis_step as analyseSteps
    [Matrix,K]=calculateMatrix.getSensormatrix('Fx_C2.csv','Fy_C2.csv','Fz_C2.csv','Mx_C2.csv','My_C2.csv','Mz_C2.csv',100,1,file_path)
    
    # === ORIGINAL INDIVIDUAL AXIS ANALYSIS ===
    print("\n=== STARTING ANALYSIS ===")
    print(f"Analysis Axes Configuration:")
    print(f"  FzAnalysis = {FzAnalysis}")
    print(f"  FxAnalysis = {FxAnalysis}")
    print(f"  FyAnalysis = {FyAnalysis}")
    print(f"  MzAnalysis = {MzAnalysis}")
    print(f"  MxAnalysis = {MxAnalysis}")
    print(f"  MyAnalysis = {MyAnalysis}")
    print(f"  FTAnalysis = {FTAnalysis}")
    print(f"\nAnalysis Type Configuration:")
    print(f"  UseStepAnalysis = {UseStepAnalysis}")
    print(f"  Use3StepAnalysis = {Use3StepAnalysis}")
    print("=" * 50)
    
    # Initialize axis error tracking dictionary for comprehensive results
    axis_errors = {}
    axis_names = ['fX', 'fY', 'fZ', 'mX', 'mY', 'mZ']

    # === FZ STEP ANALYSIS ===
    if FzAnalysis and UseStepAnalysis:
        plt, double = setup_matplotlib()  # Setup matplotlib
        
        data = pd.read_csv('Fz_Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))

        main_channel=2
        F_step = np.cumsum((fZ_weight_per_step/5)*9.81*np.array([0, 1,1,1,1,1])/1000)

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-2],fZ_shift,fZ_thr,main_channel,F_step,showPlotInt,fZ_MVG,fZ_offset)
        
        if Ergebniss is not None:
            axis_errors['fZ'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,folder, machine_id)
        plt.close()

    # === FX STEP ANALYSIS ===
    if FxAnalysis and UseStepAnalysis:
        plt, double = setup_matplotlib()  # Setup matplotlib
    
        data = pd.read_csv('Fx_Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[1:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))

        main_channel=0
        F_step = np.cumsum((fX_weight_per_step/5)*9.81*np.array([0, 1,1,1,1,1])/1000)
        M_step = -F_step*0.001
        M_step = np.append(M_step,np.flip(M_step[0:len(M_step)-1]))

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)],fX_shift,fX_thr,main_channel,F_step,showPlotInt,fX_MVG,fX_offset)

        if Ergebniss is not None:
            axis_errors['fX'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        M_step2=M_step[0:len(M_step)-1]
        for x in range(0,10):
            M_step2 = np.append(M_step2,M_step[0:len(M_step)-1])
        M_step2 = M_step2[0:len(achse)]

        mean_step[4,:]=mean_step[4,:]+M_step
        mittelwert[:,4]=mittelwert[:,4]+M_step2
        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,folder, machine_id)
        plt.close()

    #=== FY STEP ANALYSIS === (DISABLED - Using 3-step analysis instead)
    if FyAnalysis and UseStepAnalysis:
        print("[DEBUG] Starting regular FY analysis")
        plt, double = setup_matplotlib()  # Setup matplotlib
    
        data = pd.read_csv('Fy_Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[10:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))
    
        main_channel=1
        F_step = np.cumsum((fY_weight_per_step/5)*9.81*np.array([0, 1,1,1,1,1])/1000)
        M_step = F_step*0.001
        M_step = np.append(M_step,np.flip(M_step[0:len(M_step)-1]))
    
        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-200],fY_shift,fY_thr,main_channel,F_step,showPlotInt,fY_MVG,fY_offset)
    
        if Ergebniss is not None:
            axis_errors['fY'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }
    
        M_step2=M_step[0:len(M_step)-1]
        for x in range(0,10):
            M_step2 = np.append(M_step2,M_step[0:len(M_step)-1])
        M_step2 = M_step2[0:len(achse)]
    
        mean_step[3,:]=mean_step[3,:]+M_step
        mittelwert[:,3]=mittelwert[:,3]+M_step2

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,folder, machine_id)
        print("[OK] Regular FY analysis completed")
        plt.close()

    # === MZ STEP ANALYSIS ===
    if MzAnalysis and UseStepAnalysis:
        plt, double = setup_matplotlib()  # Setup matplotlib
    
        data = pd.read_csv('Mz_Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[10:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))

        main_channel=5
        Kette1=np.cumsum(mZ_chain1_weight_per_step/5*9.81*np.array([0, 1,1,1,1,1])/1000)
        Kette2=np.cumsum(mZ_chain2_weight_per_step/5*9.81*np.array([0,  1,1,1,1,1])/1000)
        M_step = mZ_weight_distance*(Kette1+Kette2)

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-2],mZ_shift,mZ_thr,main_channel,M_step,showPlotInt,mZ_MVG,mZ_offset)

        if Ergebniss is not None:
            axis_errors['mZ'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        if mittelwert is not None:
            analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,folder, machine_id)
            plt.close()
        else:
            print(f"ERROR: Step detection failed for mZ channel. Consider adjusting threshold parameters:")
            print(f"  Current: mZ_shift={mZ_shift}, mZ_thr={mZ_thr}, mZ_MVG={mZ_MVG}, mZ_offset={mZ_offset}")
            print(f"  Try reducing mZ_thr (currently {mZ_thr}) to a lower value like 6 or 8") 

    # === MX STEP ANALYSIS ===
    if MxAnalysis and UseStepAnalysis:
        plt, double = setup_matplotlib()  # Setup matplotlib
    
        data = pd.read_csv('Mx_Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))

        main_channel=3
        Kette1=np.cumsum(mX_chain1_weight_per_step/5*9.81*np.array([0, 1,1,1,1,1])/1000)
        Kette2=np.cumsum(mX_chain2_weight_per_step/5*9.81*np.array([0,  1,1,1,1,1])/1000)
        M = mX_weight_distance*(Kette1+Kette2)
        M_step=M

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,:],mX_shift,mX_thr,main_channel,M_step,showPlotInt,mX_MVG,mX_offset)
        
        if Ergebniss is not None:
            axis_errors['mX'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }
        
        if mittelwert is not None:
            analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,folder, machine_id)
            plt.close()
        else:
            print(f"ERROR: Step detection failed for mX channel. Consider adjusting threshold parameters:")
            print(f"  Current: mX_shift={mX_shift}, mX_thr={mX_thr}, mX_MVG={mX_MVG}, mX_offset={mX_offset}")
            print(f"  Try reducing mX_thr (currently {mX_thr}) to a lower value like 10 or 12") 

    # === MY STEP ANALYSIS ===
    if MyAnalysis and UseStepAnalysis:
        plt, double = setup_matplotlib()  # Setup matplotlib
    
        data = pd.read_csv('My_Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))

        main_channel=4
        Kette1=np.cumsum(mY_chain1_weight_per_step/5*9.81*np.array([0, 1,1,1,1,1])/1000)
        Kette2=np.cumsum(mY_chain2_weight_per_step/5*9.81*np.array([0,  1,1,1,1,1])/1000)
        M = mY_weight_distance*(Kette1+Kette2)
        M_step=M

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)],mY_shift,mY_thr,main_channel,M_step,showPlotInt,mY_MVG,mY_offset)

        if Ergebniss is not None:
            axis_errors['mY'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        if mittelwert is not None:
            analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,folder, machine_id)
            plt.close()
        else:
            print(f"ERROR: Step detection failed for mY channel. Consider adjusting threshold parameters:")
            print(f"  Current: mY_shift={mY_shift}, mY_thr={mY_thr}, mY_MVG={mY_MVG}, mY_offset={mY_offset}")
            print(f"  Try reducing mY_thr (currently {mY_thr}) to a lower value like 15 or 18")

    # === ADDITIONAL TEST CYCLE ANALYSIS ===
    print("\n=== ANALYZING TEST CYCLE DATA ===")

    # === FZ TEST ANALYSIS ===
    if FzAnalysis and os.path.exists('Fz_test.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Fz_test.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=2
        F_step = np.cumsum((fZ_weight_per_step/5)*9.81*np.array([0, 1,1,1,1,1])/1000)

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-2],fZ_shift,fZ_thr,main_channel,F_step,showPlotInt,fZ_MVG,fZ_offset)
        
        if Ergebniss is not None:
            axis_errors['fZ_test'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
            'crosstalk_force': Ergebniss[7],
            'crosstalk_moment': Ergebniss[8]
        }

        try:
            print("  [DEBUG] Calling make_Plot_Steps for Fz test...")
            print(f"  [DEBUG] Parameters - mittelwert shape: {mittelwert.shape}, main_channel: {main_channel}")
            print(f"  [DEBUG] Ergebniss: {Ergebniss}")
            analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Fz_test.csv', machine_id, "Step_Test")
            print("  [OK] Fz test cycle analysis completed")
        except Exception as e:
            print(f"  [ERROR] Failed in make_Plot_Steps for Fz test: {e}")
            print(f"  [ERROR] Error type: {type(e).__name__}")
            traceback.print_exc()
            raise
        plt.close()    # === FX TEST ANALYSIS ===
    if FxAnalysis and os.path.exists('Fx_test.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Fx_test.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[1:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=0
        F_step = np.cumsum((fX_weight_per_step/5)*9.81*np.array([0, 1,1,1,1,1])/1000)

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)],fX_shift,fX_thr,main_channel,F_step,showPlotInt,fX_MVG,fX_offset)

        if Ergebniss is not None:
            axis_errors['fX_test'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Fx_test.csv', machine_id, "Step_Test")
        print("  [OK] Fx test cycle analysis completed")
        plt.close()

    # === FY TEST ANALYSIS ===
    if FyAnalysis and os.path.exists('Fy_test.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Fy_test.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[10:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=1
        F_step = np.cumsum((fY_weight_per_step/5)*9.81*np.array([0, 1,1,1,1,1])/1000)

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-200],fY_shift,fY_thr,main_channel,F_step,showPlotInt,fY_MVG,fY_offset)

        if Ergebniss is not None:
            axis_errors['fY_test'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Fy_test.csv', machine_id, "Step_Test")
        print("  [OK] Fy test cycle analysis completed")
        plt.close()

    # === MZ TEST ANALYSIS ===
    if MzAnalysis and os.path.exists('Mz_test.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Mz_test.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[10:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=5
        Kette1=np.cumsum(mZ_chain1_weight_per_step/5*9.81*np.array([0, 1,1,1,1,1])/1000)
        Kette2=np.cumsum(mZ_chain2_weight_per_step/5*9.81*np.array([0,  1,1,1,1,1])/1000)
        M_step = mZ_weight_distance*(Kette1+Kette2)

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-2],mZ_shift,mZ_thr,main_channel,M_step,showPlotInt,mZ_MVG,mZ_offset)

        if Ergebniss is not None:
            axis_errors['mZ_test'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Mz_test.csv', machine_id, "Step_Test")
        print("  [OK] Mz test cycle analysis completed")
        plt.close()

    # === MX TEST ANALYSIS ===
    if MxAnalysis and os.path.exists('Mx_test.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Mx_test.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=3
        Kette1=np.cumsum(mX_chain1_weight_per_step/5*9.81*np.array([0, 1,1,1,1,1])/1000)
        Kette2=np.cumsum(mX_chain2_weight_per_step/5*9.81*np.array([0,  1,1,1,1,1])/1000)
        M = mX_weight_distance*(Kette1+Kette2)
        M_step=M

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,:],mX_shift,mX_thr,main_channel,M_step,showPlotInt,mX_MVG,mX_offset)
        
        if Ergebniss is not None:
            axis_errors['mX_test'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Mx_test.csv', machine_id, "Step_Test")
        print("  [OK] Mx test cycle analysis completed")
        plt.close()

    # === MY TEST ANALYSIS ===
    if MyAnalysis and os.path.exists('My_test.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('My_test.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=4
        Kette1=np.cumsum(mY_chain1_weight_per_step/5*9.81*np.array([0, 1,1,1,1,1])/1000)
        Kette2=np.cumsum(mY_chain2_weight_per_step/5*9.81*np.array([0,  1,1,1,1,1])/1000)
        M = mY_weight_distance*(Kette1+Kette2)
        M_step=M

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)],mY_shift,mY_thr,main_channel,M_step,showPlotInt,mY_MVG,mY_offset)

        if Ergebniss is not None:
            axis_errors['mY_test'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'My_test.csv', machine_id, "Step_Test")
        print("  [OK] My test cycle analysis completed")
        plt.close()

    #== 3-STEP ANALYSIS ===
    print("\n=== ANALYZING 3-STEP DATA (0,3,5,3,0) ===")

    # === FZ 3-STEP ANALYSIS ===
    if FzAnalysis and Use3StepAnalysis and os.path.exists('Fz_3Step.csv'):
        print("  [DEBUG] Starting Fz 3-step analysis...")
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Fz_3Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[2:len(file),1:7]
        raw_Data = data.astype(double)
        plt.plot(raw_Data)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))
        main_channel=2
        # 3-step sequence: 0,3,5,3,0 (absolute weights, not cumulative)
        F_step = (fZ_weight_per_step/5)*9.81*np.array([0, 3, 5])/1000
        print(f"  [DEBUG] F_step for 3-step: {F_step}")
        print(f"  [DEBUG] Raw data shape: {raw_Data.shape}")
        print(f"  [DEBUG] Channel {main_channel} range: {np.min(raw_Data[:,main_channel]):.3f} to {np.max(raw_Data[:,main_channel]):.3f}")

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-2],fZ_shift_3step,fZ_thr_3step,main_channel,F_step,showPlotInt_3step,fZ_MVG_3step,fZ_offset_3step)
        
        if Ergebniss is not None:
            axis_errors['fZ_3step'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        # Debug output before plotting
        print(f"  [DEBUG] Data for plotting:")
        print(f"    mittelwert shape: {mittelwert.shape if mittelwert is not None else 'None'}")
        print(f"    mean_step shape: {mean_step.shape if mean_step is not None else 'None'}")
        print(f"    achse_2 shape: {achse_2.shape if achse_2 is not None else 'None'}")
        if mittelwert is not None and mittelwert.size > 0:
            print(f"    mittelwert[channel {main_channel}] range: {np.min(mittelwert[:,main_channel]):.3f} to {np.max(mittelwert[:,main_channel]):.3f}")
        if mean_step is not None and mean_step.size > 0:
            print(f"    mean_step[channel {main_channel}]: {mean_step[main_channel,:]}")
            
        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Fz_3Step.csv', machine_id, "Step_3")
        print("  [OK] Fz 3-step analysis completed")
        plt.show()  # This will now work with TkAgg backend
       # plt.pause(20)  # Pause for 2 seconds to view the plot
        #plt.close()

    # === FX 3-STEP ANALYSIS ===
    if FxAnalysis and Use3StepAnalysis and os.path.exists('Fx_3Step.csv'):
        print("  [DEBUG] Starting Fx 3-step analysis...")
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Fx_3Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[1:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=0
        F_step = (fX_weight_per_step/5)*9.81*np.array([0, 3, 5])/1000
        print(f"  [DEBUG] F_step for 3-step: {F_step}")
        print(f"  [DEBUG] Raw data shape: {raw_Data.shape}")
        print(f"  [DEBUG] Channel {main_channel} range: {np.min(raw_Data[:,main_channel]):.3f} to {np.max(raw_Data[:,main_channel]):.3f}")

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)],fX_shift_3step,fX_thr_3step,main_channel,F_step,showPlotInt_3step,fX_MVG_3step,fX_offset_3step)

        if Ergebniss is not None:
            axis_errors['fX_3step'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Fx_3Step.csv', machine_id, "Step_3")
        print("  [OK] Fx 3-step analysis completed")
        plt.close()

    # === FY 3-STEP ANALYSIS ===
    if FyAnalysis and Use3StepAnalysis and os.path.exists('Fy_3Step.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Fy_3Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[2:len(file),1:7]  # Skip header rows
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))
        
        main_channel=1
        # 3-step sequence: 0,3,5,3,0 (absolute weights, not cumulative)
        F_step = (fY_weight_per_step/5)*9.81*np.array([0, 3, 5])/1000

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-2],fY_shift_3step,fY_thr_3step,main_channel,F_step,showPlotInt_3step,fY_MVG_3step,fY_offset_3step)

        if Ergebniss is not None:
            axis_errors['fY'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        # Use make_Plot_Steps with Step_3 folder for 3-step analysis
        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Fy_3Step.csv', machine_id, "Step_3")
        print("[OK] Fy 3-step analysis completed")
      #  plt.close()

    # # === MZ 3-STEP ANALYSIS ===
    if MzAnalysis and Use3StepAnalysis and os.path.exists('Mz_3Step.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('Mz_3Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[10:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        main_channel=5
        # 3-step moment calculation with absolute weights (not cumulative)
        Kette1=(mZ_chain1_weight_per_step/5*9.81*np.array([0, 3, 5])/1000)
        Kette2=(mZ_chain2_weight_per_step/5*9.81*np.array([0, 3, 5])/1000)
        M_step = mZ_weight_distance*(Kette1+Kette2)

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)-2],mZ_shift_3step,mZ_thr_3step,main_channel,M_step,showPlotInt_3step,mZ_MVG_3step,mZ_offset_3step)

        if Ergebniss is not None:
            axis_errors['mZ_3step'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Mz_3Step.csv', machine_id, "Step_3")
        print("  [OK] Mz 3-step analysis completed")
        plt.close()

    # === MX 3-STEP ANALYSIS ===
    if MxAnalysis and Use3StepAnalysis and os.path.exists('Mx_3Step.csv'):
        plt.close()
        plt, double = setup_matplotlib()
        data=None
        file=None
        data = pd.read_csv('Mx_3Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[2:len(file),1:7]
        raw_Data = data.astype(double)
        plt.plot(raw_Data)
        plt.show()
        plt.pause(15)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
       
        plt.plot(raw_Data)
        plt.show()
        plt.pause(15)
        main_channel=3
        Kette1=(mX_chain1_weight_per_step/5*9.81*np.array([0, 3, 5])/1000)
        Kette2=(mX_chain2_weight_per_step/5*9.81*np.array([0, 3, 5])/1000)
        M = mX_weight_distance*(Kette1+Kette2)
        M_step=M

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,:],mX_shift_3step,mX_thr_3step,main_channel,M_step,showPlotInt_3step,mX_MVG_3step,mX_offset_3step)
        
        if Ergebniss is not None:
            axis_errors['mX_3step'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'Mx_3Step.csv', machine_id, "Step_3")
        print("  [OK] Mx 3-step analysis completed")
        plt.close()

    # === MY 3-STEP ANALYSIS ===
    if MyAnalysis and Use3StepAnalysis and os.path.exists('My_3Step.csv'):
        plt, double = setup_matplotlib()
        
        data = pd.read_csv('My_3Step.csv', names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        main_channel=4
        Kette1=(mY_chain1_weight_per_step/5*9.81*np.array([0, 3, 5])/1000)
        Kette2=(mY_chain2_weight_per_step/5*9.81*np.array([0, 3, 5])/1000)
        M = mY_weight_distance*(Kette1+Kette2)
        M_step=M

        [mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss, percision_boxplot]=analyseSteps.getSteps(raw_Data[:,0:len(raw_Data)],mY_shift_3step,mY_thr_3step,main_channel,M_step,showPlotInt_3step,mY_MVG_3step,mY_offset_3step)

        if Ergebniss is not None:
            axis_errors['mY_3step'] = {
                'max_range': Ergebniss[1],
                'slope_error': Ergebniss[2],
                'zero_point_error': Ergebniss[3],
                'repeatability': Ergebniss[4],
                'linearity_error': Ergebniss[5],
                'hysteresis': Ergebniss[6],
                'crosstalk_force': Ergebniss[7],
                'crosstalk_moment': Ergebniss[8]
            }

        analyseSteps.make_Plot_Steps(mittelwert,streuung,achse,mean_step,std_mean_step,achse_2,fit_Gerade, Ergebniss,main_channel, percision_boxplot,'My_3Step.csv', machine_id, "Step_3")
        print("  [OK] My 3-step analysis completed")
        plt.close()

    # Optional FT Analysis plots
    import pylab
    if FTAnalysis:
        print("\nGenerating FT Analysis plots...")
        plt, double = setup_matplotlib()  # Setup matplotlib
        
        data = pd.read_csv(file_path, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        data = file[20:len(file),1:7]
        raw_Data = data.astype(double)
        raw_Data=(np.transpose(np.matmul((Matrix),np.transpose(raw_Data [0:None,:])))) 
        
        # FT Values plot
        fig, axs = plt.subplots(2)
        fig.suptitle(file_path+'_FT-Values.png')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))
        fig = plt.gcf()
        fig.set_size_inches(1.2*18.5, 1.2*10.5)
        plt.savefig(file_path+'_FT-Values.png')
        plt.close()

        # FT Values Detail plot
        fig, axs = plt.subplots(2)
        fig.suptitle(file_path+'_FT-ValuesDetail.png')
        lob=axs[0].plot(raw_Data[:,0:3])
        axs[0].set_ylim(5,10)
        axs[0].legend((lob), ('Fx', 'Fy', 'Fz'))
        lob=axs[1].plot(raw_Data[:,3:7])
        axs[1].set_ylim(-40,40)
        axs[1].legend((lob), ('Mx', 'My', 'Mz'))
        fig = plt.gcf()
        fig.set_size_inches(1.2*18.5, 1.2*10.5)
        plt.savefig(file_path+'_FT-ValuesDetail.png')
        plt.close()
        
        data = pd.read_csv(file_path, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
        file = data.to_numpy()
        
        # Ensure temperature data is numeric and handle any string/categorical issues
        try:
            temp_data = pd.to_numeric(file[:,7], errors='coerce')
            temp_data = temp_data[~np.isnan(temp_data)]  # Remove NaN values
            
            if len(temp_data) > 0:
                fig, axs = plt.subplots(1)
                fig.suptitle(file_path+'_temp.png')
                
                # Plot with explicit numeric data and index
                x_values = np.arange(len(temp_data))
                axs.plot(x_values, temp_data)
                
                mittel=np.mean(temp_data)
                ax=plt.gca()
                ax.set_ylim(mittel-0.5,mittel+0.5)
                fig = plt.gcf()
                fig.set_size_inches(1.2*18.5, 1.2*10.5)
                plt.savefig(file_path+'_temp.png')
                plt.close()
                print(f"  [OK] Temperature plot saved: {file_path}_temp.png")
            else:
                print(f"  [WARNING] No valid temperature data found for plotting")
        except Exception as e:
            print(f"  [ERROR] Failed to create temperature plot: {e}")
            traceback.print_exc()

    print("\n=== ANALYSIS COMPLETED ===")

    # Display comprehensive axis error summary
    print("\n" + "="*80)
    print("                         AXIS ERROR ANALYSIS SUMMARY")
    print("="*80)

    if axis_errors:
        # Header for the table
        print(f"{'AXIS':<6} {'RANGE':<8} {'SLOPE':<8} {'ZERO':<8} {'REPEAT':<8} {'LINEAR':<8} {'HYSTER':<8} {'CROSS-F':<8} {'CROSS-M':<8}")
        print(f"{'[unit]':<6} {'[%FS]':<8} {'[o/ooFS]':<8} {'[%FS]':<8} {'[%FS]':<8} {'[%FS]':<8} {'[%FS]':<8} {'[%FS]':<8} {'[%FS]':<8}")
        print("-" * 80)
        
        # Display results for each axis
        for axis_name in axis_names:
            if axis_name in axis_errors:
                errors = axis_errors[axis_name]
                print(f"{axis_name:<6} "
                      f"{errors['max_range']:<8.2f} "
                      f"{errors['slope_error']:<8.3f} "
                      f"{errors['zero_point_error']:<8.2f} "
                      f"{errors['repeatability']:<8.2f} "
                      f"{errors['linearity_error']:<8.2f} "
                      f"{errors['hysteresis']:<8.2f} "
                      f"{errors['crosstalk_force']:<8.2f} "
                      f"{errors['crosstalk_moment']:<8.2f}")
            else:
                print(f"{axis_name:<6} {'--':<8} {'--':<8} {'--':<8} {'--':<8} {'--':<8} {'--':<8} {'--':<8} {'--':<8}")
        
        print("-" * 80)
        
        # Find axis with highest errors in each category
        print("\nWORST PERFORMING AXES:")
        
        # Helper function to find worst axis for a given error type
        def find_worst_axis(error_type, format_str="{:.2f}"):
            worst_axis = None
            worst_value = 0
            for axis_name, errors in axis_errors.items():
                if error_type in errors and abs(errors[error_type]) > abs(worst_value):
                    worst_value = errors[error_type]
                    worst_axis = axis_name
            return worst_axis, worst_value, format_str.format(abs(worst_value)) if worst_axis else "N/A"
        
        worst_repeatability, _, rep_val = find_worst_axis('repeatability')
        worst_linearity, _, lin_val = find_worst_axis('linearity_error')
        worst_hysteresis, _, hys_val = find_worst_axis('hysteresis')
        worst_crosstalk_f, _, cross_f_val = find_worst_axis('crosstalk_force')
        worst_crosstalk_m, _, cross_m_val = find_worst_axis('crosstalk_moment')
        worst_slope, _, slope_val = find_worst_axis('slope_error', "{:.3f}")
        
        print(f"  - Worst Repeatability:     {worst_repeatability or 'N/A':<4} ({rep_val}%FS)")
        print(f"  - Worst Linearity:         {worst_linearity or 'N/A':<4} ({lin_val}%FS)")
        print(f"  - Worst Hysteresis:        {worst_hysteresis or 'N/A':<4} ({hys_val}%FS)")
        print(f"  - Worst Force Crosstalk:   {worst_crosstalk_f or 'N/A':<4} ({cross_f_val}%FS)")
        print(f"  - Worst Moment Crosstalk:  {worst_crosstalk_m or 'N/A':<4} ({cross_m_val}%FS)")
        print(f"  - Worst Slope Error:       {worst_slope or 'N/A':<4} ({slope_val}o/ooFS)")
        
        print("\nLEGEND:")
        print("  RANGE    = Maximum measurement range")
        print("  SLOPE    = Slope deviation from ideal (o/ooFS = per mille of full scale)")
        print("  ZERO     = Zero point error")
        print("  REPEAT   = Repeatability (measurement scatter)")
        print("  LINEAR   = Linearity error (deviation from straight line)")
        print("  HYSTER   = Hysteresis (difference between up/down measurements)")
        print("  CROSS-F  = Crosstalk from force channels")
        print("  CROSS-M  = Crosstalk from moment channels")
        
    else:
        print("No axis error data collected during analysis.")
        print("This may indicate that step detection failed for all axes.")

    print("="*80)

    # Export sensor KPI data to Excel/CSV
    def export_sensor_kpi_to_excel():
        """Export sensor KPI data to Excel file in the Step folder - comprehensive all analysis types"""
        try:
            # Get the sensor name from the file path
            sensor_name = os.path.basename(file_path_master)
            
            # Define headers as requested (standard format)
            header1 = ['','', 'Fx', '', '', '', '', '','Fy', '', '', '', '', '', 'Fz', '', '', '', '', '', 'Mx', '', '', '', '', '', 'My', '', '', '', '', '', 'Mz', '', '', '', '', '']
            header2 = ['ID', 'Sensor_Name',
                'n_0_Fx', 's_Fx', 'f_lin_Fx', 'f_hys_Fx', 'q_f_Fx', 'q_m_Fx',
                'n_0_Fy', 's_Fy', 'f_lin_Fy', 'f_hys_Fy', 'q_f_Fy', 'q_m_Fy',
                'n_0_Fz', 's_Fz', 'f_lin_Fz', 'f_hys_Fz', 'q_f_Fz', 'q_m_Fz',
                'n_0_Mx', 's_Mx', 'f_lin_Mx', 'f_hys_Mx', 'q_f_Mx', 'q_m_Mx',
                'n_0_My', 's_My', 'f_lin_My', 'f_hys_My', 'q_f_My', 'q_m_My',
                'n_0_Mz', 's_Mz', 'f_lin_Mz', 'f_hys_Mz', 'q_f_Mz', 'q_m_Mz']

            # Prepare data rows for different analysis types
            data_rows = []
            
            # Helper function to create a data row for a specific analysis type
            def create_data_row(row_id, analysis_type, suffix=""):
                data_row = [row_id, f"{sensor_name}_{analysis_type}"]
                axis_order = ['fX', 'fY', 'fZ', 'mX', 'mY', 'mZ']
                
                for axis_key in axis_order:
                    # Create the full axis key with suffix for different analysis types
                    full_axis_key = axis_key + suffix
                    
                    if full_axis_key in axis_errors:
                        errors = axis_errors[full_axis_key]
                        data_row.extend([
                            round(errors['zero_point_error'], 2) if 'zero_point_error' in errors else 0.00,
                            round(errors['repeatability'], 2) if 'repeatability' in errors else 0.00,
                            round(errors['linearity_error'], 2) if 'linearity_error' in errors else 0.00,
                            round(errors['hysteresis'], 2) if 'hysteresis' in errors else 0.00,
                            round(errors['crosstalk_force'], 2) if 'crosstalk_force' in errors else 0.00,
                            round(errors['crosstalk_moment'], 2) if 'crosstalk_moment' in errors else 0.00
                        ])
                    else:
                        # No data for this axis in this analysis type
                        data_row.extend([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
                return data_row

            # Determine which analysis types have data and add corresponding rows
            row_id = 0
            analysis_types_found = []
            
            # Check for regular measurement data (no suffix)
            regular_axes = [axis for axis in axis_errors.keys() if not axis.endswith('_test') and not axis.endswith('_3step')]
            if regular_axes:
                row_id += 1
                data_rows.append(create_data_row(row_id, "measurement", ""))
                analysis_types_found.append("measurement")
            
            # Check for test phase data (_test suffix)
            test_axes = [axis for axis in axis_errors.keys() if axis.endswith('_test')]
            if test_axes:
                row_id += 1
                data_rows.append(create_data_row(row_id, "test_phase", "_test"))
                analysis_types_found.append("test_phase")
            
            # Check for 3-step data (_3step suffix)
            three_step_axes = [axis for axis in axis_errors.keys() if axis.endswith('_3step')]
            if three_step_axes:
                row_id += 1
                data_rows.append(create_data_row(row_id, "3_step", "_3step"))
                analysis_types_found.append("3_step")

            # If no data found at all, create a default empty row
            if not data_rows:
                data_rows.append(create_data_row(1, "no_data", ""))

            # Compose the full table as a list of lists
            table = [header1, header2] + data_rows
            # Create DataFrame for export (header2 as columns, data_rows as data)
            df = pd.DataFrame(data_rows, columns=header2)
            
            # Always use "Step" folder as requested by user
            step_folder = "Step"
            os.makedirs(step_folder, exist_ok=True)
            
            # Generate filename based on what analysis types are included
            if len(analysis_types_found) > 1:
                filename = f'sensor_kpi_comprehensive.xlsx'
                csv_filename = f'sensor_kpi_comprehensive.csv'
            elif analysis_types_found:
                filename = f'sensor_kpi_{analysis_types_found[0]}.xlsx'
                csv_filename = f'sensor_kpi_{analysis_types_found[0]}.csv'
            else:
                filename = 'sensor_kpi.xlsx'
                csv_filename = 'sensor_kpi.csv'
            
            excel_path = os.path.join(step_folder, filename)
            csv_path = os.path.join(step_folder, csv_filename)
            
            try:
                # Try Excel export with header1 and header2 as first two rows
                import openpyxl
                from openpyxl import Workbook
                wb = Workbook()
                ws = wb.active
                for row in table:
                    ws.append(row)
                wb.save(excel_path)
                print(f"\n[SUCCESS] Comprehensive Sensor KPI data exported to Excel: {excel_path}")
            except ImportError:
                # Fallback to CSV export with header1 and header2 as first two rows
                with open(csv_path, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    for row in table:
                        writer.writerow(row)
                print(f"\n[SUCCESS] Comprehensive Sensor KPI data exported to CSV: {csv_path}")
                print("   (Excel export requires 'openpyxl' package - install with: pip install openpyxl)")
                excel_path = csv_path
            
            # Count unique axes across all analysis types
            unique_axes = set()
            for key in axis_errors.keys():
                base_axis = key.replace('_test', '').replace('_3step', '')
                unique_axes.add(base_axis)
            
            print(f"   Contains {len(unique_axes)} axes with {len(analysis_types_found)} analysis type(s): {', '.join(analysis_types_found)}")
            print(f"   Total axis-analysis combinations: {len(axis_errors)}")
            return excel_path
            
        except Exception as e:
            print(f"\n[ERROR] Failed to export sensor KPI: {str(e)}")
            return None

    # Export the KPI data
    if axis_errors:
        export_sensor_kpi_to_excel()
    else:
        print("\n[WARNING] No KPI data to export - step detection failed for all axes")

    if running_from_gui:
        print("[SUCCESS] Analysis completed successfully! Check the output folder for results.")
    else:
        print("[SUCCESS] Analysis completed successfully!")
    print(f"Results saved in: {folder}")

if __name__ == "__main__":
    main()