#!/usr/bin/env python

import mytoggle
import mygpio
import mycom
global driver
maxAngle_deg = 16
global maxSpeed_km_h

def init_motor_SIM():
    from vehicle import Driver
    global driver
    global maxSpeed_km_h
    
    driver = Driver()
    driver.setSteeringAngle(0)
    driver.setCruisingSpeed(0)
    maxSpeed_km_h = 28

def init_motor_RPI():
    global maxSpeed_km_h
    
    maxSpeed_km_h = 25.25

def set_speed_m_s_SIM(speed_m_s):
    global driver
    
    speed = speed_m_s*3.6
    if speed > maxSpeed_km_h :
        speed = maxSpeed_km_h
    if speed < 0 :
        speed = 0
    driver.setCruisingSpeed(speed)
    
def set_speed_m_s_RPI(speed_m_s):
    speed = speed_m_s*3.6
    if(speed >= maxSpeed_km_h):
        speed = maxSpeed_km_h
    print("vitesse:",(int)(speed)) #*2.525
    mycom.transmission([mycom.ADDR_speed_req_m_s,int(speed/0.11)])
    #mycom.execution_transmission(10, 0)
    #mycom.transmission([10],[(int)(speed*2.525)])

def set_wheeldir_deg_SIM(angle_deg):
    global driver
    
    if angle_deg > maxAngle_deg:
        angle_deg = maxAngle_deg
    elif angle_deg < -maxAngle_deg:
        angle_deg = -maxAngle_deg
    angle_final_rad = -angle_deg * 3.14/180
    driver.setSteeringAngle(angle_final_rad)
    
def set_wheeldir_deg_RPI(angle_deg):
    angle_deg = -angle_deg
    if angle_deg > maxAngle_deg:
        angle_deg = maxAngle_deg
    elif angle_deg < -maxAngle_deg:
        angle_deg = -maxAngle_deg
    #time_ms = 1.5 + angle_deg*(2.1-1.5)/45
    time_ms = 1.5 + angle_deg*(2.1-1.5)/45
    dutyPercent = time_ms*0.001/(1/mygpio.ServoFreq) * 100
    if (dutyPercent > 0 and dutyPercent < 100):
        mygpio.ServoPWM.ChangeDutyCycle(dutyPercent)

def set_recule_SIM(): #sur la voiture réelle, il y a un stop puis un recul pendant 1s.
    driver.setCruisingSpeed(-1)
    
def set_recule_RPI(): #sur la voiture réelle, il y a un stop puis un recul pendant 1s.
    print("recule pas truc")

def while_condition_SIM():
    if driver.step() != -1:
        return True
    return False

def while_condition_RPI():
    return True

while_condition = [0,0]
while_condition[mytoggle.SIM_FN_INDEX] = while_condition_SIM
while_condition[mytoggle.RPI_FN_INDEX] = while_condition_RPI

init_motor = [0,0]
init_motor[mytoggle.SIM_FN_INDEX] = init_motor_SIM
init_motor[mytoggle.RPI_FN_INDEX] = init_motor_RPI

set_speed_m_s = [0,0]
set_speed_m_s[mytoggle.SIM_FN_INDEX] = set_speed_m_s_SIM
set_speed_m_s[mytoggle.RPI_FN_INDEX] = set_speed_m_s_RPI

set_wheeldir_deg = [0,0]
set_wheeldir_deg[mytoggle.SIM_FN_INDEX] = set_wheeldir_deg_SIM
set_wheeldir_deg[mytoggle.RPI_FN_INDEX] = set_wheeldir_deg_RPI

set_recule = [0,0]
set_recule[mytoggle.SIM_FN_INDEX] = set_recule_SIM
set_recule[mytoggle.RPI_FN_INDEX] = set_recule_RPI
