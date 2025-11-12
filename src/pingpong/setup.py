from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pingpong'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools',
                      'opencv-python',
                      'deepface',
                      'numpy<2',
                      'ultralytics',
                      ],
    zip_safe=True,
    maintainer='ogi',
    maintainer_email='ioshelg@gmail.com',
    description='Face age estimation nodes (dev)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher = pingpong.image_publisher:main',
            'age_estimation = pingpong.age_estimation:main',
            'vision_processor = pingpong.vision_processor:main',
            'trigger = pingpong.vision_trigger:main',
            'controller = pingpong.controller:main',
            'serve_calculator = pingpong.serve_calculator:main',
            'sample_image_subscriber = pingpong.sample_image_subscriber:main',
            'motor_controller = pingpong.motor_controller:main'
        ],
    },
)
