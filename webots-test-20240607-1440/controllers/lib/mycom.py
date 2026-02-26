#!/usr/bin/env python

import mytoggle
global spi
#def pour la communication spi
ADDR_wheeldir_req_deg = 1
ADDR_speed_req_m_s = 2
ADDR_rngfinder_meas = 11
ADDR_acclmtr_meas = 12
ADDR_battvoltage_meas = 13


###################################################
#definition des fonctions pour la communication spi

def init_com_SIM():
    print("not initializing SPI communication because simulating")

def init_com_RPI():
    import spidev
    global spi
    
    spi = spidev.SpiDev(0, 1)
    spi.max_speed_hz = 1000000 
    spi.mode = 1

#def transmission_OLD(data_send):
#    infos = 0
#    spi.writebytes([data_send[0]])
#    match data_send[0]:
#        case mycom.ADDR_wheeldir_req_deg | mycom.ADDR_speed_req_m_s:
#            spi.writebytes([data_send[1]])
#            infos = None
#        case _:
#            infos = spi.xfer([0x00])
#    return infos
    
def transmission(data_send):
    spi.xfer([data_send[0]])
    infos = spi.xfer([(int)(data_send[1])])
    if (data_send[0] != ADDR_wheeldir_req_deg) and (data_send[0] != ADDR_speed_req_m_s):
        infos = None
    return infos

#fonction qui faut modifier pour ajouter des valeurs a recuperer de la rpi
def get_sensor_data():
    infos_rngfinder = transmission([ADDR_rngfinder_meas, 0x00])
    infos_acclmtr = transmission([ADDR_acclmtr_meas, 0x00])
    return {'rngfinder':infos_rngfinder, 'acclmtr':infos_acclmtr}

#fonction principal pour le code    
def execution_transmission(speed_req_m_s, wheeldir_req_deg):
    transmission([ADDR_speed_req_m_s, (int)(speed_req_m_s)])
    transmission([ADDR_wheeldir_req_deg, (int)(wheeldir_req_deg)])
    return get_sensor_data()

init_com = [0,0]
init_com[mytoggle.SIM_FN_INDEX] = init_com_SIM
init_com[mytoggle.RPI_FN_INDEX] = init_com_RPI
