#!/home/ogi/PingPongRobot/.venv/bin/python

from setuptools import find_packages, setup

package_name = 'face_age_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
                      'mediapipe',
                      'opencv-python',
                      'numpy'],
    zip_safe=True,
    maintainer='ogi',
    maintainer_email='ioshelg@gmail.com',
    description='Face age estimation nodes (dev)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher = face_age_estimation.image_publisher:main',
            'age_estimation = face_age_estimation.age_estimation:main'
        ],
    },
)
