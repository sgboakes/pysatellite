"""
Setup script for pysatellite using PBR package.
(Use 'setup.cfg' to update package details and PBR will pick them up automatically)
"""

from setuptools import setup

# setup(setup_requires=["pbr"],pbr=True)
setup(name='pysatellite',
      maintainer='Benedict Oakes',
      maintainer_email='sgboakes@liverpool.ac.uk',
      url='https://github.com/sgboakes/pysatellite',
      install_requires=[
          'numpy', 'scipy', 'matplotlib', 'pandas', 'astropy', 'filterpy', 'sgp4'
      ],
      pbr=True,
      )
