# Copyright 1996-2022 Cyberbotics Ltd.
#
# Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2023b
# Inspiré de vehicle_driver_altino controller
# Kévin Hoarau, Anthony Juton, Bastien Lhopitallier, Martin Raynaud
# août 2023

import psutil
import os

# --- Set high priority for the current process ---
try:
    p = psutil.Process(os.getpid())
    # For Windows, use HIGH_PRIORITY_CLASS. Other options exist for other OSes.
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    print(f"Successfully set process priority to High.")
except Exception as e:
    print(f"Could not set process priority: {e}")


from vehicle import Driver
from controller import Lidar
import numpy as np
import time
import random

driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

#Lidar
lidar = Lidar("RpLidarA2")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud() 

#clavier
keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

# vitesse en km/h
speed = 0
maxSpeed = 28 #km/h

# angle de la direction
angle = 0
maxangle_degre = 16

# mise a zéro de la vitesse et de la direction
driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)


tableau_lidar_mm=[0]*360
step=0
step_max=500
PERT_ANG_MAX = 20
PERT_ANG_PERIOD = 58
PERT_VIT_MAX = 4
PERT_VIT_PERIOD = 32

def set_vitesse_m_s(vitesse_m_s):
    speed = vitesse_m_s*3.6
    if speed > maxSpeed :
        speed = maxSpeed
    if speed < 0 :
        speed = 0
    driver.setCruisingSpeed(speed)
     
def set_direction_degre(angle_degre):
    if angle_degre > maxangle_degre:
        angle_degre = maxangle_degre
    elif angle_degre < -maxangle_degre:
        angle_degre = -maxangle_degre   
    angle = -angle_degre * 3.14/180
    driver.setSteeringAngle(angle)

def recule(): #sur la voiture réelle, il y a un stop puis un recul pendant 1s.
    driver.setCruisingSpeed(-1)  

# mode auto desactive
modeAuto = False
print("cliquer sur la vue 3D pour commencer")
print("a pour mode auto (pas de mode manuel sur TT02_jaune), n pour stop")

header_list = ["time", "vitesse", "angle"] + [str(i) for i in range(-180, 180)];
header_string = ",".join(header_list)
data_matrix = np.empty((0, len(header_list)), dtype=float)

while driver.step() != -1:
    while True:
    #acquisition des donnees du lidar
         # recuperation de la touche clavier
        currentKey = keyboard.getKey()
 
        if currentKey == -1:
            break
       
        elif currentKey == ord('n') or currentKey == ord('N'):
            if modeAuto :
                modeAuto = False
                print("--------Modes Auto TT-02 jaune Désactivé-------")
        elif currentKey == ord('a') or currentKey == ord('A'):
            if not modeAuto : 
                modeAuto = True
                print("------------Mode Auto TT-02 jaune Activé-----------------")
    
    #acquisition des donnees du lidar
    donnees_lidar_brutes = lidar.getRangeImage()
    for i in range(360) :
        if (donnees_lidar_brutes[-i]>0) and (donnees_lidar_brutes[-i]<20) :
            tableau_lidar_mm[i-180] = 1000*donnees_lidar_brutes[-i]
        else :
            tableau_lidar_mm[i-180] = 0
   
    if not modeAuto:
        set_direction_degre(0)
        set_vitesse_m_s(0)
        
    if modeAuto:
        if (step%PERT_ANG_PERIOD == 0):
            pert_ang = PERT_ANG_MAX*2*(random.uniform(0, 1)-0.5)
            print("pert_ang:",pert_ang)
        if (step%PERT_VIT_PERIOD == 0):
            pert_vit = PERT_VIT_MAX*random.uniform(0, 1)
            print("pert_vit:",pert_vit)
        step=step+1
        angle_degre = pert_ang + 0.03*(tableau_lidar_mm[60]-tableau_lidar_mm[-60])
        angle_degre = min(maxangle_degre,max(-maxangle_degre,angle_degre))
        vitesse_m_s = pert_vit + 0.5
        vitesse_m_s = min(maxSpeed,max(-maxSpeed,vitesse_m_s))
        set_direction_degre(angle_degre)
        set_vitesse_m_s(vitesse_m_s)
        data_row_part = np.array([1, vitesse_m_s, angle_degre])
        data_row = np.append(data_row_part, tableau_lidar_mm)
        data_matrix = np.vstack([data_matrix, data_row])
        if (step==step_max):
            filename="pert2_"+f'{vitesse_m_s:06.2f}'+"_"+f'{PERT_VIT_MAX:06.2f}'+"_"+f'{PERT_VIT_PERIOD:06.2f}'+"_"+f'{PERT_ANG_MAX:06.2f}'+"_"+f'{PERT_ANG_PERIOD:06.2f}'+".csv"
            print(os.getcwd())
            print(filename)
            np.savetxt(filename, data_matrix, delimiter=",", header=header_string);
            set_vitesse_m_s(0);
            print("finished !");
            exit();
 
    #########################################################

