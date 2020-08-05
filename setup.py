from setuptools import setup, find_packages


setup(name='OpenIBL',
      version='0.1.0',
      description='Open-source toolbox for Image-based Localization (Place Recognition)',
      author_email='geyixiao831@gmail.com',
      url='https://github.com/yxgeee/OpenIBL',
      license='MIT',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Image Localization',
          'Image Retrieval',
          'Place Recognition'
      ])
