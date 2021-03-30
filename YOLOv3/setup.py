from setuptools import find_packages, setup

REQUIRED_PACKAGES = ['torch', 'numpy', 'pypng']


setup(
    name = 'YOLOv3',
    version = '0.1',
    author = 'Sajeeb Roy Chowdhury',
    author_email = 'src_s@rocketmail.com',
    packages = find_packages(),
    description = 'setup file for YOLOv3.',
    install_requires = REQUIRED_PACKAGES,
)