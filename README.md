# retentioneering-framework
Tools for user trajectories analysis in the app (python package)

## installation

Fistly you need to download c++ compiler

### Mac

Install [homebrew](https://brew.sh/)

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Then you should install gcc

```bash
brew install gcc
```

You also need python if you haven't it

```bash
brew install python3
```

And git

```bash
brew install git
```

### Windows

Install [Microsoft Visual Studio Build tools](https://visualstudio.microsoft.com/ru/downloads/). You can find it at `Инструменты для Visual Studio 2017 > Build Tools для Visual Studio 2017`.

If you need python then install it from [here](https://www.python.org/downloads/release/python-368/). Please, be sure that you select `Add python to path` in the bottom of installer.

And git from [here](https://git-scm.com/downloads).

### Python package

- To install Python package from github, you need to clone that repository.

```bash
git clone https://github.com/appintheair/retentioneering-framework.git
```
- Install dependencies from requirements.txt file from that directory
```bash 
sudo pip install -r requirements.txt
```
or if previous command don't work
```bash 
sudo pip3 install -r requirements.txt
```
- Then just run the setup.py file from that directory
```bash
sudo python setup.py install
```
or if previous command don't work
```bash
sudo python3 setup.py install
```
## First steps
- Put path to your google cloud credentials and your project name in settings yaml ([example](examples/new_users_lost_prediction/settings_yaml.yaml)).
```yaml
---
settings:
  service_account_path: "../credentials.yaml"
  project: project
```
- Now you are ready to do [examples](examples).
## Examples
There are 3 examples in this project you can play with. All of them are in "Examples" directory.
- First of them is "delete account prediction" in directory delete_account_prediction. With this example one can compare users who have deleted their account with those who don't, plot graphs with their transitions between events and use our ML alghoritm to predict their behaviour at the first steps. To start you should correct settings.yaml to fit the project to your app: you should select event which is responsible for account deletion, times in user's history you are interesting, destination table for your data in BigQuery and parameters for filtering raw data.
- Second is "new_users lost prediction" in directory new_users_lost_prediction. With this example one can compare users behavior who started using your app with those who stopped doing it after severeal steps. You mast correct settings.yaml to fit the project to your app: you should select filters with parameters from BigQuery to choose suitable users and also filter their events, data you are interesting to investigate, destination_table and parameters for filtering raw data at the preprocessing steps.
- Third one is "new users lost prunned visualization" in directory new_users_lost_prunned_visualization. This example provides simple code for graph visualization. Graph presents users avereged path for the cases described for second example. You should correct settings.yaml in same way as for the second case.
For all graphs we offer our own graph plotter api that can provide nice-looking graphs for better visualisation your results. Of course you can use default visualization python tools to plot your results.

## Analysis
You can use retentioneering.analysis toolset with your data.

Data should have at least three columns `event_name`, `user_pseudo_id`, `event_timestamp`.

You can put empty dict as `settings` for analysis. It is needed here only to set experiment folder. By default it will be `experiments` folder in you current directory. For each new experiment (kernel restart) new folder with experiment results named with current datetime will be created.

```python
import pandas as pd
from retentioneering import analysis
# your data import
df = pd.read_csv('path to your data')
settings = dict()
analysis.get_desc_table(df, settings=settings)
```
