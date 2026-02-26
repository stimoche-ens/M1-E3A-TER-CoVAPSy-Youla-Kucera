#!/usr/bin/env python

import mytoggle
import mymotor
import statistics

tableau_lidar_mm=[0]*360
global lidar

def init_lidar_SIM():
    #Lidar
    global lidar
    from controller import Lidar
    
    basicTimeStep = int(mymotor.driver.getBasicTimeStep())
    sensorTimeStep = 4 * basicTimeStep
    lidar = Lidar("RpLidarA2")
    lidar.enable(sensorTimeStep)
    lidar.enablePointCloud()
    
def init_lidar_RPI():
    global lidar
    from lidar_lib import LidarLibrary
    
    lidar = LidarLibrary('/dev/ttyUSB0')
    lidar.connect_lidar()

def get_tableau_lidar_mm_SIM():
    global lidar
    
    donnees_lidar_brutes = lidar.getRangeImage()
    for i in range(360) :
        if (donnees_lidar_brutes[-i]>0) and (donnees_lidar_brutes[-i]<20) :
            tableau_lidar_mm[i] = 1000*donnees_lidar_brutes[-i]
        else :
            tableau_lidar_mm[i] = 0
    return tableau_lidar_mm

def get_tableau_lidar_mm_RPI():
    global lidar
    tableau_lidar_mm = lidar.read_lidar_measurements()
    for i in range(-180, 180):
        if (tableau_lidar_mm[i] == 0):
            tableau_lidar_mm[i] = statistics.mean(tableau_lidar_mm[i-4:i-1]+tableau_lidar_mm[i+1:i+4]) 
    return tableau_lidar_mm


init_lidar = [0,0]
init_lidar[mytoggle.SIM_FN_INDEX] = init_lidar_SIM
init_lidar[mytoggle.RPI_FN_INDEX] = init_lidar_RPI

get_tableau_lidar_mm = [0,0]
get_tableau_lidar_mm[mytoggle.SIM_FN_INDEX] = get_tableau_lidar_mm_SIM
get_tableau_lidar_mm[mytoggle.RPI_FN_INDEX] = get_tableau_lidar_mm_RPI
