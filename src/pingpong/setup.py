from setuptools import find_packages, setup

package_name = 'pingpong'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
                      'opencv-python',
                      'deepface',
                      'numpy<2',
                      'ultralytics',
                      'mediapipe',
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
            'sample_image_subscriber = pingpong.sample_image_subscriber:main',
            'skeleton_recognition = pingpong.skeleton_recognition:main',
        ],
    },
)
