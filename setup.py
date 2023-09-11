from setuptools import setup, find_packages
with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]

setup(name='fibseg',
	version='1.0.1',
	description='DL method for fiber bundle segmentation on histological sections',
	author='Vaanathi Sundaresan',
	install_requires=install_requires,
    scripts=['fibseg/scripts/fibseg'],
	packages=find_packages(),
	include_package_data=True)
