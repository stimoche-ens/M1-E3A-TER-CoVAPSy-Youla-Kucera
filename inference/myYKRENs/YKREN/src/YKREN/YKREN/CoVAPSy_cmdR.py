from ackermann_msgs.msg import AckermannDrive

from std_msgs.msg import Float32
import re

import rclpy
from rclpy.node import Node

import serial as s

port_serie = s.Serial(port='/dev/ttyACM0', baudrate=115200, bytesize=8, parity='N',
                      stopbits=1, timeout=0.01, write_timeout=None,
                      xonxoff=False, rtscts=False, dsrdtr=False)


class NoeudCommande(Node):
    def __init__(self):
        super().__init__('CoVAPSy_cmdR')
        self.__vitesse_m_s = 0.0
        self.__direction_degre = 0
        self.create_subscription(AckermannDrive, 'cmd_ackermann', self.__cmd_ackermann_callback, 1)
        self.__pub_vitesse = self.create_publisher(Float32, 'vitesse_mesuree', 1) # Publication de la vitesse mesurée dans un topic
        self.__pub_ultrason = self.create_publisher(Float32, 'distance_ultrason', 1)
        self.create_timer(0.1, self.__lire_serie_callback)
        self.get_logger().info('noeud cree')

    def __cmd_ackermann_callback(self, message):
        self.__vitesse_m_s = message.speed
        self.__direction_degre = message.steering_angle
        if self.__direction_degre > 25:
            self.__direction_degre = 25
        elif self.__direction_degre < -25:
            self.__direction_degre = -25
        try:
            direction = int(float(90 + self.__direction_degre))
        except:
            self.get_logger().warn('Bug direction:{},{}'.format(direction, type(direction)))
        vitesse = int(4000 + self.__vitesse_m_s*1000)   # 4000 vitesse nulle
        port_serie.write(str.encode('v{0:05}d{1:03}\r'.format(vitesse, direction)))
        self.get_logger().info('v{0:05}d{1:03}'.format(vitesse, direction))

    def __lire_serie_callback(self):
        while port_serie.in_waiting > 0:
            try:
                ligne = port_serie.readline().decode('utf-8').strip()
            
                # Split sur les lettres pour isoler les nombres
                parts = re.split(r'[vub]', ligne)
                V = int(parts[1]) # vitesse mesurée par capteur vitesse
                U = int(parts[2]) # distance mesurée par ultrasons
                B = int(parts[3]) # tension batterie (élec) 
                
                        
                # Conversion de la valeur brute vers m/s
                vitesse_lue_m_s = (V - 4000) / 1000.0
                
                msg_vitesse = Float32()
                msg_vitesse.data = vitesse_lue_m_s
                self.get_logger().info('Vitesse mesurée: {:.2f} m/s'.format(vitesse_lue_m_s))
                self.__pub_vitesse.publish(msg_vitesse)

                # Publication de la distance ultrasons (valeur brute supposee en cm)
                msg_ultrason = Float32()
                msg_ultrason.data = float(U)
                self.__pub_ultrason.publish(msg_ultrason)
                        
            except (UnicodeDecodeError, ValueError, IndexError):
                pass


def main(args=None):
    rclpy.init(args=args)
    noeud = NoeudCommande()
    rclpy.spin(noeud)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

