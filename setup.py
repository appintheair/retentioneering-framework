from setuptools import setup, find_packages

setup(
    name='retentioneering',
    version='0.3.0',
    license='Mozilla Public License',
    description='Python package for user trajectories analysis in the app',
    long_description='Python package for user trajectories analysis in the app',
    author='App in the Air',
    author_email='maxim.godzi@gmail.com',
    url='https://github.com/appintheair/retentioneering-framework',
    keywords=['ANALYTICS', 'CLICKSTREAM', 'RETENTIONEERING', 'RETENTION', 'GRAPHS', 'TRAJECTORIES'],
    install_requires=[
        'cmake',
        'google-cloud-bigquery>=1.6.0',
        'grandalf>=0.6',
        'hdbscan>=0.8.19',
        'matplotlib>=2.2.3',
        'MulticoreTSNE>=0.0.1.1',
        'networkx>=2.2',
        'numpy>=1.14.4',
        'pandas>=0.23.1',
        'plotly>=3.3.0',
        'pyyaml>=4.2b4',
        'requests>=2.20.0',
        'scikit-learn>=0.19.2',
        'scipy>=1.1.0',
        'seaborn>=0.9.0',
        'tqdm>=4.26.0',
        'jupyter>=1.0.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Analysts',
        'License :: OSI Approved :: Mozilla Public License 2.0'
    ],
    packages=find_packages()
)
