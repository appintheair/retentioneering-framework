from setuptools import setup, find_packages

setup(
    name='retentioneering',
    version='0.1.0',
    description='Python package for user trajectories analysis in the app',
    long_description='Python package for user trajectories analysis in the app',
    author='App in the Air',
    url='https://github.com/appintheair/aita-ml-retentioneering-python',
    install_requires=[
        'seaborn>=0.8.0',
        'pandas>=0.23.1',
        'scikit-learn>=0.19.0',
        'google-cloud-bigquery>=1.6.0',
        'plotly>=3.3.0',
        'MulticoreTSNE>=0.0.1.1',
        'Wand>=0.4.4',
        'networkx>=2.2'
    ],
    packages=find_packages()
)
