from setuptools import find_packages, setup
import os
import glob

package_name = 'YKREN'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        #(os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'models'), glob.glob(os.path.join('YKREN', '*.joblib'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='voituremaxime',
    maintainer_email='voituremaxime@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'CoVAPSy_cmdR = YKREN.CoVAPSy_cmdR:main',
            'CoVAPSy_conduiteM = YKREN.CoVAPSy_conduiteM:main',
            'CoVAPSy_conduiteR = YKREN.CoVAPSy_conduiteR:main'
        ],
    },
)
