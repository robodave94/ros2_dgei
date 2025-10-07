from setuptools import find_packages, setup

package_name = 'l2cs_net'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch_gaze_tracking.launch']),
        ('share/' + package_name + '/weights', ['weights/L2CSNet_gaze360.pkl']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vscode',
    maintainer_email='david.hinwood@canberra.edu.au',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_gaze_tracking = l2cs_net.run_gaze_tracking:main',
        ],
    },
)
