#!/usr/bin/env python

# Copyright 1996-2022 Cyberbotics Ltd.
#
# Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2023b
# Inspiré de vehicle_driver_altino controller
# Kévin Hoarau, Anthony Juton, Bastien Lhopitallier, Martin Raynaud
# août 2023


import click
import sys
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '../lib')

import mytoggle
import mygpio
import mycom
import mymotor
import mylidar
import myalgo
import time


def init_all():
    mygpio.init_GPIO[mytoggle.SEL_FN_INDEX]()
    mycom.init_com[mytoggle.SEL_FN_INDEX]()
    mymotor.init_motor[mytoggle.SEL_FN_INDEX]()
    mylidar.init_lidar[mytoggle.SEL_FN_INDEX]()

def kill_all():
    mymotor.set_speed_m_s[mytoggle.SEL_FN_INDEX](0)
    mylidar.lidar.disconnect_lidar()
    mygpio.destroy_GPIO[mytoggle.SEL_FN_INDEX]()
    mycom.spi.close()


def mode_manu():
    manu_speed = 0
    manu_angle = 0
    while (1):
        print("manu speed:", manu_speed)
        print("manu angle:", manu_angle)
        mymotor.set_speed_m_s[mytoggle.SEL_FN_INDEX](manu_speed)
        mymotor.set_wheeldir_deg[mytoggle.SEL_FN_INDEX](manu_angle)
        c = click.getchar()
        click.echo()
        if (c == 'z' or c == 'w' or c == '\x1b[A'):
            manu_speed = manu_speed+1
        elif (c == 's' or c == '\x1b[B'):
            manu_speed = manu_speed-1
        elif (c == 'q' or c == 'a' or c == '\x1b[D'):
            manu_angle = manu_angle+1
        elif (c == 'd' or c == '\x1b[C'):
            manu_angle = manu_angle-1
        else:
            print("invalid key")

def mode_auto():
    import RPi.GPIO as GPIO
    mymotor.set_speed_m_s[mytoggle.SEL_FN_INDEX](0)
    time.sleep(1)

    while mymotor.while_condition[mytoggle.SEL_FN_INDEX]() != -1:
        tableau_lidar_mm = mylidar.get_tableau_lidar_mm[mytoggle.SEL_FN_INDEX]()
        myalgo.main_algo(tableau_lidar_mm)
        if GPIO.input(mygpio.BTN1_GPIO) == 0:
            kill_all()
            break


def select_mode_SIM():
    init_all()
    basicTimeStep = int(mymotor.driver.getBasicTimeStep())
    sensorTimeStep = 4 * basicTimeStep
    keyboard = mymotor.driver.getKeyboard()
    keyboard.enable(sensorTimeStep)
    
    mode_auto()
    '''
    while (1):
        print("choose mode (m for manu, a for auto)")
        currentKey = keyboard.getKey()
        if currentKey == -1:
            break
        elif (currentKey == 'a'):
            mode_auto()
        elif (currentKey == 'm'):
            mode_manu()
        else:
            print("invalid mode!!!")
            '''

def select_mode_RPI():
    import RPi.GPIO as GPIO

    while (1):
        init_all()
        print("choose mode (m for manu, a for auto)")
        c = click.getchar()
        click.echo()
        if (c == 'a'):
            mode_auto()
        elif (c == 'm'):
            mode_manu()
        else:
            print("invalid mode!!!")


select_mode = [0,0]
select_mode[mytoggle.SIM_FN_INDEX] = select_mode_SIM
select_mode[mytoggle.RPI_FN_INDEX] = select_mode_RPI

    

try :
    select_mode[mytoggle.SEL_FN_INDEX]()
except KeyboardInterrupt :
    kill_all()


#########################################################

