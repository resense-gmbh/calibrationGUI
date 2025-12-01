#import pandas as pd
import os
import numpy as np
import pandas as pd
from pylab import *
from config_manager import ConfigManager  # Assuming ConfigManager is in config_manager module

def setup_matplotlib():
    """Setup matplotlib with interactive backend"""
    import matplotlib
    matplotlib.use('TkAgg')  # Use interactive backend
    import matplotlib.pyplot as plt
    plt.ion()  # Enable interactive mode
    return plt


def matprint(mat, fmt="g"):

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]

    for x in mat:

        for i, y in enumerate(x):

            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")

        print("")



def showPlots(dataName,data,xLimit,K,str_Lastvektor,L_1):
    """Display plots for calibration data"""
    plt = setup_matplotlib()  # Setup matplotlib when needed

    plt.figure(figsize=(12, 10))
    ax1 = plt.subplot2grid((10,1), (0, 0), rowspan=6)
    ax2 = plt.subplot2grid((10,1), (7, 0), rowspan=3)
    plt.sca(ax1)
    print('Hallo')
    plt.yticks(K)
    plt.plot(data)
    plt.plot([0,len(data)],[K,K],'k', linestyle='dashed')
    plt.plot([xLimit,xLimit],[np.min(data),np.max(data)],'k', linestyle='dashed')
    plt.plot([0,len(data)],[0,0],'k')
    plt.title(dataName+' Lastvektor: '+ str_Lastvektor)
    K=np.round(K)
    plt.sca(ax2)
    plt.ylim([-2000,2000])
    plt.violinplot(L_1-K)
    plt.xticks([1,2,3,4,5,6], K.astype(str))
    
    
    plt.show()
    current_Dir=os.getcwd()
    os.chdir(current_Dir+'\\Kalibrierung')
    plt.savefig(dataName+'_Lastvektor_'+str_Lastvektor+'.png')
    os.chdir(current_Dir)
    



def getSensormatrix(str_Fx,str_Fy,str_Fz,str_FxMy,str_FyMx,str_FyMz,xLimit,shPl,file_path):
    
    # Load weight chain parameters from config using proper method
    # Get machine_id from environment variable (set by GUI) or use default
    machine_id = os.environ.get('CALIBRATION_MACHINE_ID', 'CM2')
    config_manager = ConfigManager()
    config = config_manager.create_or_update_local_config(machine_id, file_path)
    
    fX_weight_per_step = config.get('fX_weight_per_step', 2500)  # Default 2500g
    fY_weight_per_step = config.get('fY_weight_per_step', 2500)
    fZ_weight_per_step = config.get('fZ_weight_per_step', 2500)
    mX_weight_distance = config.get('mX_weight_distance', 127.33)  # Default 127.33
    mY_weight_distance = config.get('mY_weight_distance', 127.33)  # Default 127.33
    mZ_weight_distance = config.get('mZ_weight_distance', 81.03)   # Default 81.03
    # Moment weight chains (both chains per moment)
    mX_chain1_weight_per_step = config.get('mX_chain1_weight_per_step', 236)  # Default 236g
    mX_chain2_weight_per_step = config.get('mX_chain2_weight_per_step', 236)
    mY_chain1_weight_per_step = config.get('mY_chain1_weight_per_step', 236)
    mY_chain2_weight_per_step = config.get('mY_chain2_weight_per_step', 236)
    mZ_chain1_weight_per_step = config.get('mZ_chain1_weight_per_step', 236)
    mZ_chain2_weight_per_step = config.get('mZ_chain2_weight_per_step', 236)

    os.chdir(os.path.dirname(file_path))
    current_Dir=os.getcwd()
    print(os.getcwd()  +'Hallo' )
    if (os.path.exists(current_Dir+'\\Kalibrierung')==False)==True:
        print('Make Directory Kalibrierung')
        os.mkdir(current_Dir+'\\Kalibrierung')
    
    Index_Names=['Fx.csv','Fx2.csv','Fx3.csv','Fx4.csv','FxMy.csv','FxMy2.csv','FxMy3.csv','Fy.csv','Fy2.csv','Fy3.csv','Fy4.csv','FyMx.csv','FyMx2.csv','FyMx3.csv','Fz.csv','Fz2.csv','Fz3.csv','Fz4.csv','FyMz.csv','FyMz2.csv','FyMz2Alu.csv','Fx_S.csv','Fy_S.csv','Fz_S.csv','Mx_S.csv','My_S.csv','Mz_S.csv','Fx_C.csv','Fy_C.csv','Fz_C.csv','Mx_C.csv','My_C.csv','Mz_C.csv','Fx_C2.csv','Fy_C2.csv','Fz_C2.csv','Mx_C2.csv','My_C2.csv','Mz_C2.csv']
    
    Last_soll= np.array([    [ 3.405,      0,        0,          0,   1.627,       0],   #0-Fx                         
                         [ 16.618,      0,        0,          0,      4.818,       0],   #1-Fx2
                         [ 29.889,      0,        0,          0,      7.962,       0],   #1-Fx3
                         [ 43.151,      0,        0,          0,     11.193,       0],   #1-Fx4 
                         [ 10.016,      0,        0,          0,     55.953,       0],   #2-FxMy
                         [ 16.618,      0,        0,          0,    163.080,       0],   #3-FxMy2
                         [ 23.279,      0,        0,          0,    323.822,       0],   #3-FxMy3
                         [     0,  3.405,        0,      -1.627,           0,       0],  #4-Fy 
                         [     0,  16.618,        0,     -4.818,          0,       0],   #5-Fy2
                         [     0,  29.889,        0,     -7.962,          0,       0],   #5-Fy3
                         [     0,  43.151,        0,    -11.193,          0,       0],   #5-Fy4 
                         [     0,  10.016,        0,    -55.953,          0,       0],   #6-FyMx
                         [     0,  16.618,        0,   -163.080,          0,       0],   #7-FyMx2
                         [     0,  23.279,        0,   -323.822,          0,       0],   #7-FyMx3
                         [     0,       0,    3.405,          0,          0,       0],   #8-Fz  
                         [     0,       0,   16.618,          0,          0,       0],   #9-Fz2
                         [     0,       0,   29.889,          0,          0,       0],   #9-Fz3
                         [     0,       0,   43.151,          0,          0,       0],   #9-Fz4 
                         #[  0,      3.394,        0,      -1.597,         0,  23.760],   #12-FyMz
                         [  0,      3.394,        0,      -1.597,         0,  13.58],   #12-FyMz
                         [  0,     16.608,        0,      -4.790,         0, 116.258],  #13-FyMz2
                         [  0,     28.942/2,        0,      -3.800,         0, 175.000/2],  #13-FyMz2Alu mit ME k6d gemessen
                         [  50.008/2,       0,        0,           0,        10,        0],  #Fx_S
                         [       0,  50.008/2,        0,         -10,         0,        0],  #Fy_S
                         [       0,       0,   50.008/2,           0,         0,        0],  #Fz_S
                         [       0,  -0.1/2,     0.54/2,    488.89/2,         0,        0],  #Mx_S
                         [       0,  -0.1/2,     0.54/2,           0,  488.89/2,        0],  #My_S
                         [       0,       0,          0,           0,         0, 488.89/2], #Mz_S
                         [  (fX_weight_per_step/1000*9.81)/2,       0,        0,           0,         0,        0],  #Fx_C
                         [       0,  (fY_weight_per_step/1000*9.81)/2,        0,           0,         0,        0],  #Fy_C
                         [       0,       0,   -(fZ_weight_per_step/1000*9.81)/2,           0,         0,        0],  #Fz_C
                         [       0,       0,          0,       (mX_weight_distance * (mX_chain1_weight_per_step + mX_chain2_weight_per_step) / 1000)/2,         0,        0],  #Mx_C
                         [       0,       0,          0,           0,         (mY_weight_distance * (mY_chain1_weight_per_step + mY_chain2_weight_per_step) / 1000)/2,        0],  #My_C
                         [       0,       0,          0,           0,         0,        (mZ_weight_distance * (mZ_chain1_weight_per_step + mZ_chain2_weight_per_step) / 1000)/2], #Mz_C
                         [  (fX_weight_per_step/1000*9.81)/2,       0,        0,           0,         0,        0],  #Fx_C2
                         [       0,  (fY_weight_per_step/1000*9.81)/2,        0,           0,         0,        0],  #Fy_C2
                         [       0,       0,   -(fZ_weight_per_step/1000*9.81)/2,           0,         0,        0],  #Fz_C2
                         [       0,       0,          0,       (mX_weight_distance * 9.81 * (mX_chain1_weight_per_step + mX_chain2_weight_per_step) / 1000)/2,         0,        0],  #Mx_C2
                         [       0,       0,          0,           0,         (mY_weight_distance * 9.81 * (mY_chain1_weight_per_step + mY_chain2_weight_per_step) / 1000)/2,        0],  #My_C2
                         [       0,       0,          0,           0,         0,        (mZ_weight_distance * 9.81 * (mZ_chain1_weight_per_step + mZ_chain2_weight_per_step) / 1000)/2]]) #Mz_C2


                        
                   
                        
    
    names=np.array([str_Fx,str_Fy,str_Fz,str_FxMy,str_FyMx,str_FyMz])
    Last=np.zeros((len(names),6),double)
    Lastvektor= np.zeros(len(names))
    str_Lastvektor = ''
    
    for x in range(0,len(names)):
        for z in range(0,len(Index_Names)):
            if names[x]==Index_Names[z]:
                print('Found')
                Lastvektor[x]=z
                Last[x,:]=Last_soll[z,:]
                print(Last)
                name=Index_Names[z]
                name=name[0:len(name)-4]
                str_Lastvektor=str_Lastvektor+'_'+name
                break
            

#
    
    data = pd.read_csv(str_Fx, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    file = data.to_numpy()
    #file=np.genfromtxt(str_Fx,dtype=np.float32,delimiter='\t', skip_footer=20, skip_header=25)
    Fx = file[20:len(file),1:7]
    Fx = Fx.astype(double)
    L_1 = Fx[np.r_[1:xLimit],:]
    K1 = np.mean((L_1),axis=0)
    print('Hallo')
    if shPl == 1:
        showPlots(str_Fx,Fx,xLimit,K1,str_Lastvektor,L_1)

    
    #data = pd.read_csv(str_Fy, names=['delete', 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    #data.drop('delete', axis=1, inplace=True)
    #file = data.to_numpy()
    data = pd.read_csv(str_Fy, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    file = data.to_numpy() 
    #file=np.genfromtxt(str_Fy,dtype=np.float32,delimiter='\t', skip_footer=20, skip_header=25)
    Fy = file[20:len(file),1:7]
    Fy = Fy.astype(double)
    L_2 = Fy[np.r_[1:xLimit],:]
    K2 = np.mean((L_2),axis=0)
    if shPl == 1:
        showPlots(str_Fy,Fy,xLimit,K2,str_Lastvektor,L_2)
    
    
    #data = pd.read_csv(str_Fz, names=['delete', 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    #data.drop('delete', axis=1, inplace=True)
    #file = data.to_numpy()
    data = pd.read_csv(str_Fz, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    file = data.to_numpy()
    #file=np.genfromtxt(str_Fz,dtype=np.float32,delimiter='\t', skip_footer=20, skip_header=25)
    Fz = file[20:len(file),1:7]
    Fz = Fz.astype(double)
    L_3 = Fz[np.r_[0:xLimit],:]
    K3 = np.mean((L_3),axis=0)
    if shPl == 1:
        showPlots(str_Fz,Fz,xLimit,K3,str_Lastvektor,L_3)


    #data = pd.read_csv(str_FxMy, names=['delete', 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    #data.drop('delete', axis=1, inplace=True)
    #file = data.to_numpy()
    data = pd.read_csv(str_FxMy, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    file = data.to_numpy()
    #file=np.genfromtxt(str_FxMy,dtype=np.float32,delimiter='\t', skip_footer=20, skip_header=25)
    FxMy = file[20:len(file),1:7]
    FxMy = FxMy.astype(double)
    L_4 = FxMy[np.r_[1:xLimit],:]
    K4 = np.mean((L_4),axis=0)
    if shPl == 1:
        showPlots(str_FxMy,FxMy,xLimit,K4,str_Lastvektor,L_4)
    
    #data = pd.read_csv(str_FyMx, names=['delete', 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    #data.drop('delete', axis=1, inplace=True)
    #file = data.to_numpy()
    data = pd.read_csv(str_FyMx, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    file = data.to_numpy()
    #file=np.genfromtxt(str_FyMx,dtype=np.float32,delimiter='\t', skip_footer=20, skip_header=25)
    FyMx = file[20:len(file),1:7]
    FyMx = FyMx.astype(double)
    L_5 = FyMx[np.r_[1:xLimit],:]
    K5 = np.mean((L_5),axis=0)
    if shPl == 1:    
        showPlots(str_FyMx,FyMx,xLimit,K5,str_Lastvektor,L_5)

    #data = pd.read_csv(str_FyMz, names=['delete', 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp', ])
    #data.drop('delete', axis=1, inplace=True)
    #file = data.to_numpy()
    data = pd.read_csv(str_FyMz, names=[ 'date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp'])
    file = data.to_numpy()
    #file=np.genfromtxt(str_FyMz,dtype=np.float32,delimiter='\t', skip_footer=20, skip_header=50)
    FyMz = file[20:len(file),1:7]
    FyMz = FyMz.astype(double)
    L_6 = FyMz[np.r_[1:xLimit],:]
    K6 = np.mean((L_6),axis=0)
    if shPl == 1:
        showPlots(str_FyMz,FyMz,xLimit,K6,str_Lastvektor,L_6)
    

    Last=2*Last
    print(Last)
    Last=np.transpose(Last)
    K=np.transpose(np.array([K1,K2,K3,K4,K5,K6]))
    print(Last)  
    print("K value is") 
    print("\n")
    print(K)
    print("\n")
    print("Raw ADC Reference voltage is 8772000")
    rawADC_Ref= 8772000
    mVpV_Last= 1000*K/rawADC_Ref
    mVpV_Last=np.transpose(mVpV_Last)
    # Reshape the 6x6 matrix into a 1x36 matrix
    mVpV_Last = mVpV_Last.reshape(1, 36)
    
    Last_inv=(np.linalg.inv((Last)))
    Matrix= np.matmul((K),Last_inv)
    Matrix= (np.linalg.inv((Matrix)))
    os.chdir(current_Dir+'\\Kalibrierung')
    #np.save('Matrix_'+str(str_Lastvektor) ,Matrix)
    #np.savetxt('Matrix_'+str(str_Lastvektor)+'.txt',Matrix,fmt='%.20f',delimiter=',')
    #index_mvPv= ["Fx", "Fy", "Fz", "Mx","My","Mz"]
    #header_mvPv= ["CH2", "CH1", "CH3", "CH4","CH5","CH6"]
    #df = pd.DataFrame(mVpV_Last, index=index_mvPv,columns=header_mvPv)
    df = pd.DataFrame(mVpV_Last)
    df.to_excel(excel_writer = "ignore_MvPv_1row.xlsx")
    os.chdir(current_Dir)
    if shPl==1:
        plt.close('all')
        
   
    
    return Matrix, K




