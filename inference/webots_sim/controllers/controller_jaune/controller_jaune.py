# Copyright 1996-2022 Cyberbotics Ltd.
#
# Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2023b
# Inspiré de vehicle_driver_altino controller
# Kévin Hoarau, Anthony Juton, Bastien Lhopitallier, Martin Raynaud
# août 2023

import psutil
import os
from armax3_jaune import MyLinPerturb
from yk_controller_jaune import YKControllerJaune

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

vitesse_m_s = angle_degre = 0

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

for _ in range(5):
    if driver.step() == -1:
        break

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

def get_tableau_lidar_mm():
    donnees_lidar_brutes = lidar.getRangeImage()
    for i in range(360) :
        if (donnees_lidar_brutes[-i]>0) and (donnees_lidar_brutes[-i]<11999) :
            tableau_lidar_mm[i-180] = 1000*donnees_lidar_brutes[-i]
        elif ((donnees_lidar_brutes[-i]>11998)):
            tableau_lidar_mm[i-180] = 12000
        else :
            tableau_lidar_mm[i-180] = 0
    return tableau_lidar_mm

Q_WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../automatique/identif_dyn/scripts/out/MyQParameter_weights.pth'))
armax = YKControllerJaune(1, get_tableau_lidar_mm(), Q_WEIGHTS_PATH)
#armax = MyLinPerturb(1,get_tableau_lidar_mm(), rebuild=False)

# mode auto desactive
modeAuto = False
print("cliquer sur la vue 3D pour commencer")
print("a pour mode auto (pas de mode manuel sur TT02_jaune), n pour stop")




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
    tableau_lidar_mm = get_tableau_lidar_mm()
   
    if not modeAuto:
        set_direction_degre(0)
        set_vitesse_m_s(0)
        
    if modeAuto:
        vitesse_m_s, angle_degre = armax.control(vitesse_m_s, angle_degre, tableau_lidar_mm)
        print(vitesse_m_s, angle_degre)
        set_direction_degre(angle_degre)
        set_vitesse_m_s(vitesse_m_s)
 
    #########################################################

