from setuptools import setup, find_packages


setup(name='SFRS',
      version='0.1.0',
      description='Deep Learning Library for Image-based Localization',
      author_email='geyixiao831@gmail.com',
      url='https://github.com/yxgeee/SFRS',
      license='MIT',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Image Localization',
          'Computer Vision',
          'Deep Learning'
      ])
