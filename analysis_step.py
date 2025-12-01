#import pandas as pd
import os
import numpy as np
import pandas as pd

def setup_matplotlib():
    """Setup matplotlib with interactive backend for showing plots"""
    import matplotlib
    matplotlib.use('TkAgg')  # Use interactive backend
    import matplotlib.pyplot as plt
    from matplotlib import interactive
    import matplotlib.patches as mpatches
    interactive(True)  # Enable interactive mode
    return plt, mpatches


def getSteps(raw_data, shift, thr, channel, gewichtsstufen, showPlot, MVG,offset):
 
    temp=(raw_data)
    a=raw_data
    shift1 = len(a)-shift
    shift2 = shift


    a_zw = a[shift:shift1,channel]
    a1 = a[0:(len(a)-2*shift),channel]
    a2 = a[2*shift:len(a),channel]
        
    a=a[shift:shift1,:]

    k = np.abs(a1-a2)  #Subtrahierte Signale die geschiftet sind

    df = pd.DataFrame(k,columns=['k'])
    mvg_coeff=MVG
    rolling_mean = df.k.rolling(window=mvg_coeff).mean()
    k = rolling_mean.to_numpy()
    k[0:mvg_coeff]=0
    k=thr*(((k>thr)+1)-1)
    k1 = k[1:len(k)]
    k2 = k[0:(len(k)-1)]
    k1 = np.append(0,k1)
    k2 = np.append(thr,k2)  
    k=np.int16(np.logical_xor(k1,k2)+1-1)
    el=(np.asarray(np.where(k==1)))
    
    # Debug output to understand step detection
    print(f"Step detection debug - Channel {channel}:")
    print(f"  Found {len(el[0,:])} transition points")
    print(f"  el shape: {el.shape}")
    if len(el[0,:]) > 0:
        print(f"  Transition indices: {el[0,:]}")
    
    # Check if we have enough transitions for step pairs
    if len(el[0,:]) < 2:
        print(f"  WARNING: Not enough transitions found for step detection!")
        print(f"  Parameters: shift={shift}, thr={thr}, MVG={MVG}, offset={offset}")
        return None, None, None, None, None, None, None, None, None
    
    r=np.floor((len(el[0,:]))/2)
    r=r.astype(np.int16)
    r=np.int16(2*len(gewichtsstufen)-2)*np.int16(np.floor(r/(2*len(gewichtsstufen)-2)))


    mittelwert=np.zeros((r,np.size(a,1)))
    streuung = np.zeros((r,np.size(a,1)))
    achse = np.zeros(r)
    anz_Stufen=len(gewichtsstufen) #ink. 0
    count = 0
    countup=1

    bereich=np.zeros((len(a),1))
    bereich2=np.zeros((len(a),1))
    
    v0 = el[0,2*0]
    v1 = el[0,(2*0)+1]
    percision_box=(temp[v0:200,:])

    for x in range(0,r):
        print(str(x))
        
        v0 = el[0,2*x]
        v1 = el[0,(2*x)+1]
        
        if v0 > 0:
            v0=v0+offset
        else:
            v0=v0
            
        if v1-offset < len(a):
            v1=v1-offset
        else:
            v1=v1
        #print(v0)
        #print(v1)
        #bereich[v0:v1,:]=4
        #print(bereich)
        
        mittelwert[x,:]=np.mean(a[v0:v1,:],0)
        streuung[x,:]=1*np.std(a[v0:v1,:],0)
        achse[x]=gewichtsstufen[count]

        bereich[v0:v1,:]=np.mean(a[v0:v1,channel],0)+1
        #bereich2[v0:v1,:]=1*(a[v0:v1,channel],0)
        
        if countup==1:
            count=count+1
        elif countup==0:
            count=count-1
            
        if count > anz_Stufen-1:
            countup = 0
            count=anz_Stufen-2
        elif count < 0:
            countup = 1
            count=1
     #--------------------------- 
    if showPlot==1:
        plt, _ = setup_matplotlib()  # Setup matplotlib when needed
        plt.figure()
        plt.plot(k1)
        #plt.plot(k2)      
        plt.plot(a_zw)
        plt.plot(bereich)
        #plt.plot(bereich2)
        plt.show()
              
    v0 = el[0,2*0]
    v1 = el[0,(2*0)+1]
    #percision_boxplot=(raw_data[0:1000,:])   

    if showPlot==1:
        plt, mpatches = setup_matplotlib()  # Setup matplotlib when needed
        plt.figure()
        plt.plot(mittelwert[:,channel])
        
        plt.show()    
        
            
    mean_step=np.zeros((np.size(mittelwert,1),np.int16(2*len(gewichtsstufen)-2)))
    std_mean_step=np.zeros((np.size(mittelwert,1),np.int16(2*len(gewichtsstufen)-2)))
    achse_2 = np.append(gewichtsstufen,np.flip(gewichtsstufen[1:len(gewichtsstufen)-1]))


    for k in range(0,np.size(mittelwert,1)):
        #print(str(k))
        mean_step[k,:]=np.mean(np.reshape(mittelwert[:,k],(np.int16(np.size(mittelwert,0)/len(achse_2)),len(achse_2))),0)
        std_mean_step[k,:]=np.std(np.reshape(mittelwert[:,k],(np.int16(np.size(mittelwert,0)/len(achse_2)),len(achse_2))),0)


    [Steigungsfehler, Nullpunktfehler]=np.polyfit(achse_2,mean_step[channel,:],1)
        
        

    mean_step=np.concatenate((mean_step,np.transpose([mean_step[:,0]])),1)
    std_mean_step=np.concatenate((std_mean_step,np.transpose([std_mean_step[:,0]])),1)
    achse_2=np.append(achse_2,achse_2[0])


    fit_Gerade = achse_2*Steigungsfehler+Nullpunktfehler    
    Hysterese=np.max(np.abs(np.flip(mean_step[channel,:],0)-mean_step[channel,:]))
    Linearitätsfehler = np.max(np.abs(fit_Gerade-mean_step[channel,:]))

    streuung_mean_step = np.max(1*std_mean_step[channel,:]) #  Sigam Intervall

    Crosstalk = np.max(np.abs(mean_step),1)
    Crosstalk[channel]=0
    Crosstalk_F= np.max(Crosstalk[0:3])
    Crosstalk_M= np.max(Crosstalk[3:6])

        
        
        
    Ergebniss=([channel,np.max(np.abs(achse_2)),Steigungsfehler, Nullpunktfehler,streuung_mean_step,Linearitätsfehler,Hysterese,Crosstalk_F,Crosstalk_M])


    Ergebniss[1]=np.round(Ergebniss[1]*100)/100
    Ergebniss[2]=np.round(Ergebniss[2]*1000)/1000
    Ergebniss[3]=np.round(Ergebniss[3]*100)/100
    Ergebniss[4]=np.round(Ergebniss[4]*100)/100
    Ergebniss[5]=np.round(Ergebniss[5]*100)/100
    Ergebniss[6]=np.round(Ergebniss[6]*100)/100
    Ergebniss[7]=np.round(Ergebniss[7]*100)/100
    Ergebniss[8]=np.round(Ergebniss[8]*100)/100

    #return percision_box
    return mittelwert, streuung, achse, mean_step, std_mean_step, achse_2, fit_Gerade, Ergebniss, percision_box





def make_Plot_Steps(mittelwert, streuung, achse, mean_step, std_mean_step, achse_2, fit_Gerade, Ergebniss, main_Channel, percision_boxplot,file_path, machine_id=None, folder_name="Step"):
    
    # Setup matplotlib at the beginning of the function
    plt, mpatches = setup_matplotlib()
    
    # Machine-specific force percentage limits (1% force range)
    machine_force_limits = {
        'CM1': 0.5,
        'CM2': 0.25,
        'CM3': 0.25,
        'CM4': 0.5,
        'CM5': 1.4
    }
    machine_torque_limits = {
        'CM1': 5,
        'CM2': 1.25,
        'CM3': 1.25,
        'CM4': 5,
        'CM5': 22
    }

    # Get force limit for current machine or use default
    if machine_id and machine_id in machine_force_limits:
        force_limit = machine_force_limits[machine_id]
        torque_limit = machine_torque_limits[machine_id]  # Use machine-specific torque limit
    else:
        force_limit = 0.25  # Default value
        torque_limit = 5  # Default torque limit

    Crosstalk = np.max(np.abs(mean_step),1)
    Crosstalk[main_Channel]=0
    Crosstalk_F= np.max(Crosstalk[0:3])
    Crosstalk_M= np.max(Crosstalk[3:6])
    
    Ergebniss[7]=np.round(Crosstalk_F*100)/100
    Ergebniss[8]=np.round(Crosstalk_M*100)/100
    
    
    current_Dir=os.getcwd()
    if (os.path.exists(current_Dir+'\\Step')==False)==True:
        print('Make Directory Step')
        os.mkdir(current_Dir+'\\Step')

    main_Channel=Ergebniss[0]

    fig_size=10
    fig=plt.figure(figsize=(fig_size, 1.4*fig_size))


    col = ['orange', 'blue','green', 'deepskyblue','lime', 'maroon','red']

    # Debug output for plotting variables
    print(f"[PLOT DEBUG] achse shape: {achse.shape}, values: {achse[:10] if len(achse) > 10 else achse}")
    print(f"[PLOT DEBUG] achse_2 shape: {achse_2.shape}, values: {achse_2}")
    print(f"[PLOT DEBUG] mittelwert shape: {mittelwert.shape}")
    print(f"[PLOT DEBUG] mean_step shape: {mean_step.shape}")
    print(f"[PLOT DEBUG] main_Channel: {main_Channel}")
    print(f"[PLOT DEBUG] mittelwert[:,main_Channel] first 10 values: {mittelwert[:10, main_Channel] if len(mittelwert) > 10 else mittelwert[:, main_Channel]}")
    print(f"[PLOT DEBUG] mean_step[main_Channel,:]: {mean_step[main_Channel,:]}")
    print(f"[PLOT DEBUG] std_mean_step[main_Channel,:]: {std_mean_step[main_Channel,:]}")

    if main_Channel==0:
        fig_titel='Fx'
    elif main_Channel==1:
        fig_titel='Fy'
    elif main_Channel==2:
        fig_titel='Fz'
    elif main_Channel==3:
        fig_titel='Mx'
    elif main_Channel==4:
        fig_titel='My'
    elif main_Channel==5:
        fig_titel='Mz'



    fig.suptitle('Analysis of axis '+fig_titel, fontsize=16)

  
    ax1 = plt.subplot2grid((9,2), (0, 0), rowspan=3 ,colspan=2)
    ax2 = plt.subplot2grid((9,2), (3, 0),colspan=2)
    ax3 = plt.subplot2grid((9,2), (4, 0),colspan=2)
    ax4 = plt.subplot2grid((9,2), (5, 0),colspan=2)
    ax5 = plt.subplot2grid((9,2), (6, 0),colspan=2)

    ax6 = plt.subplot2grid((9,2), (8, 0))
    ax7 = plt.subplot2grid((9,2), (8, 1))
    #ax8= plt.subplot2grid((10,2), (4, 0),colspan=2) ##Ffit 0.5 percent



    ax1.plot(achse,mittelwert[:,main_Channel],linewidth=0.3,color='lightgrey')
    ax1.errorbar(achse_2,mean_step[main_Channel,:],3*std_mean_step[main_Channel,:],capsize=4,color=col[main_Channel],ecolor='k')
    ax1.set_xticks(achse_2)
    ax1.set_xticklabels([])
    ax1.grid('on')


    if -1< main_Channel <3 :
        ax2.set_ylim(-force_limit,force_limit)
        #ax8.set_ylim(-force_limit/2,force_limit/2)
        ax2.set_yticks(([-force_limit,-force_limit*0.48,0,force_limit*0.48,force_limit]))
        #ax8.set_yticks(([-force_limit/2,-force_limit*0.48/2,0,force_limit*0.48/2,force_limit/2]))
        ax1.set_ylabel('F in N')
        ax2.set_ylabel('Ffit N 1%')
        ax2.grid('on')
        #ax8.set_ylabel('Ffit N 0.5%')
        #ax8.grid('on')
    if 2< main_Channel <6 :
        ax2.set_ylim(-torque_limit,torque_limit)
        #ax8.set_ylim(-torque_limit/2,torque_limit/2)
        ax2.set_yticks(([-torque_limit,-torque_limit*0.5,0,torque_limit*0.5,torque_limit]))
        #ax8.set_yticks(([-torque_limit/2,-torque_limit*0.48/2,0,torque_limit*0.48/2,torque_limit/2]))
        ax1.set_ylabel('M in mNm')
        ax2.set_ylabel('Mfit mNm')
        ax2.grid('on')
        #ax8.set_ylabel('Mfit 0.5%')
        #ax8.grid('on')

        

    #ax2.plot(achse,mittelwert[:,main_Channel]-achse,linewidth=0.3,color='lightgrey')
    ax2.errorbar(achse_2,mean_step[main_Channel,:]-fit_Gerade,3*std_mean_step[main_Channel,:],capsize=4,color=col[main_Channel],ecolor='k')
    ax2.plot(achse_2,fit_Gerade-fit_Gerade,color=col[6])
    ax2.set_xticks(achse_2)
    ax2.set_xticklabels([])
   # ax8.errorbar(achse_2,mean_step[main_Channel,:]-fit_Gerade,3*std_mean_step[main_Channel,:],capsize=4,color=col[main_Channel],ecolor='k')
    #ax8.plot(achse_2,fit_Gerade-fit_Gerade,color=col[6])
    #ax8.set_xticks(achse_2)
    #ax8.set_xticklabels([]) 

    F_lim=3*force_limit # ~ 3% von FN= 140N

    ax3.set_ylim(-F_lim,F_lim)
    ax3.set_yticks(([-F_lim,-np.floor(10*F_lim/2)/10,0,np.floor(10*F_lim/2)/10,F_lim]))

    for x in range(0,3,1):
        print(x)
        if main_Channel !=x:
            ax3.plot(achse,mittelwert[:,x],linewidth=0.3,color='lightgrey')
            ax3.errorbar(achse_2,mean_step[x,:],3*std_mean_step[x,:],color=col[x],capsize=4,ecolor='k')

    ax3.set_xticks(achse_2)
    ax3.set_xticklabels([])
    ax3.set_ylabel('F in N')
    ax3.grid('on')

    M_lim=3*torque_limit # ~3% von MN=2250 mNm

    ax4.set_ylim(-M_lim,M_lim)
    ax4.set_yticks(([-M_lim,-np.floor(M_lim/2),0,M_lim,np.floor(M_lim/2)]))


    for l in range(3,6,1):    
        if main_Channel !=l:
            ax4.plot(achse,mittelwert[:,l],linewidth=0.3,color='lightgrey')
            ax4.errorbar(achse_2,mean_step[l,:],3*std_mean_step[x,:],color=col[l],capsize=4,ecolor='k')


    ax4.set_xticks(np.round(achse_2*100)/100)
    ax4.set_ylabel('M in mNm')

    if -1< main_Channel <3 :
        ax4.set_xlabel('Fsoll in N')
    if 2< main_Channel <6 :
        ax4.set_xlabel('Msoll in mNm')


    ax4.grid('on')

    if -1< main_Channel <3 :
        val1 = ['Channel\n '+ fig_titel, 'Fsoll\n in\n N','k\n in\n N/N','n_0\n in\n N','s\n in\n N','f_Lin\n in\n N','f_Hys\n in\n N','Q_F\n in\n N','Q_M\n in\n mNm']
    if 2< main_Channel <6 :
        val1 = ['Channel\n '+ fig_titel, 'Msoll\n in\n mNm','k\n in\n Nm/Nm','n_0\n in\n mNm','s\n in\n mNm','f_Lin\n in\n mNm','f_Hys\n in\n mNm','Q_F\n in\n N','Q_M\n in\n mNm']
      
      

    val2 = [''] 
    # Ensure table cell text are strings to avoid backend requiring str/bytes
    try:
        # First ensure Ergebniss is a numpy array and handle any potential issues
        if not isinstance(Ergebniss, np.ndarray):
            Ergebniss = np.array(Ergebniss)
        
        # Convert all values to strings explicitly
        val3 = []
        for value in Ergebniss:
            try:
                # Handle NaN, inf, and other special float values
                if np.isnan(value) or np.isinf(value):
                    val3.append("N/A")
                else:
                    val3.append(str(float(value)))
            except (ValueError, TypeError):
                val3.append(str(value))
        
        val3 = np.asarray([val3])
        
    except Exception as e:
        print(f"[ERROR] Issue with table cell conversion: {e}")
        # Ultimate fallback: convert everything to string representation
        val3 = np.asarray([[str(v) for v in np.atleast_1d(Ergebniss)]])

    #fig, ax = plt.subplots() 
    ax5.set_axis_off() 
    table = ax5.table( 
        cellText = val3,  
        rowLabels = val2,  
        colLabels = val1, 
        #rowColours =["palegreen"] * 10,  
        #colColours =["palegreen"] * 10, 
        cellLoc ='center',  
        bbox = [0.05, -0.7, 0.9, 1.3])         
      
    #ax5.set_subtitle('Kennwerte für Hauptachse '+fig_titel, 
    #             fontweight ="bold") 
      
    #plt.show() 
    Fx = mpatches.Patch(color=col[0], label='Fx')
    Fy = mpatches.Patch(color=col[1], label='Fy')
    Fz = mpatches.Patch(color=col[2], label='Fz')
    Mx = mpatches.Patch(color=col[3], label='Mx')
    My = mpatches.Patch(color=col[4], label='My')
    Mz = mpatches.Patch(color=col[5], label='Mz')

    names=[Fx, Fy, Fz, Mx, My, Mz]
    ax1.legend(ncol=len(names),handles=names,loc='lower left',bbox_to_anchor=(0, 1))
    ax6.set_ylim([-0.01,0.01])
    ax7.set_ylim([-0.1,0.1])
    
    ax6.violinplot(percision_boxplot[:,0:3]-np.mean(percision_boxplot[:,0:3],0))
    streuung=(np.int16(np.std(percision_boxplot[:,0:3],0)*1000)/1000)
    ax6.set_xticks( [1,2,3])
    # Matplotlib expects tick labels to be strings/bytes - ensure robust conversion
    try:
        tick_labels = []
        for x in streuung:
            try:
                if np.isnan(x) or np.isinf(x):
                    tick_labels.append("N/A")
                else:
                    tick_labels.append(f"{float(x):.3f}")
            except (ValueError, TypeError):
                tick_labels.append(str(x))
        ax6.set_xticklabels(tick_labels)
    except Exception as e:
        print(f"[ERROR] Issue with ax6 tick labels: {e}")
        ax6.set_xticklabels([str(x) for x in streuung])
    
    ax7.violinplot(percision_boxplot[:,3:6]-np.mean(percision_boxplot[:,3:6],0))
    streuung=(np.int16(np.std(percision_boxplot[:,3:6],0)*1000)/1000)
    ax7.set_xticks( [1,2,3])
    # Matplotlib expects tick labels to be strings/bytes - ensure robust conversion
    try:
        tick_labels = []
        for x in streuung:
            try:
                if np.isnan(x) or np.isinf(x):
                    tick_labels.append("N/A")
                else:
                    tick_labels.append(f"{float(x):.3f}")
            except (ValueError, TypeError):
                tick_labels.append(str(x))
        ax7.set_xticklabels(tick_labels)
    except Exception as e:
        print(f"[ERROR] Issue with ax7 tick labels: {e}")
        ax7.set_xticklabels([str(x) for x in streuung])
    #plt.ylim([-2000,2000])
    #plt.violinplot(L_1-K)
    #plt.xticks([1,2,3,4,5,6], K.astype(str))



    ax6.set_xlabel('Std in N')
    ax7.set_xlabel('Std in mNm')
    ax7.set_ylabel('Torque\n in mNm')
    ax6.set_ylabel('Force in N')

    ax6.set_title('Precision of measurement')
    ax7.set_title('Precision of measurement')
    
    current_Dir=os.getcwd()
    print(current_Dir)
    
    # Create the main folder (Step_3, Step, etc.)
    folder_path = current_Dir + '\\' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    
    # Save plots to the main folder
    os.chdir(folder_path)
    
    # Extract just the filename without path and extension for safe filename
    safe_filename = os.path.basename(file_path)
    if safe_filename.endswith('.csv'):
        safe_filename = safe_filename[:-4]  # Remove .csv extension
    
    # Save plots to main folder only
    plot_filename_base = 'Step'+fig_titel+'_'+safe_filename
    fig.savefig(plot_filename_base+'.pdf')
    fig.savefig(plot_filename_base+'.png')
    
    os.chdir(current_Dir)

