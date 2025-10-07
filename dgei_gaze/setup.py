from setuptools import find_packages, setup

package_name = 'dgei_gaze'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/dgei_gaze_tracking.launch']),
        ('share/' + package_name + '/config', ['config/data.yaml', 'config/rviz_image_conf.rviz']),
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
            'dgei_attention_tracking = dgei_gaze.dgei_gaze_tracker:main',
        ],
    },
)
