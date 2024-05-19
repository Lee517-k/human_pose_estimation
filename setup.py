import os
from glob import glob
from setuptools import setup,find_packages

package_name = 'human_pose_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share',package_name),['package.xml']),
        (os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyeongseok',
    maintainer_email='hyeongseok@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)

# from setuptools import setup

# package_name = 'human_pose_estimation'

# setup
