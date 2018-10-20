# retentioneering-framework
Tools for user trajectories analysis in the app (python package)

## installation
- To install Python package from github, you need to clone that repository.
```bash
git clone git@github.com:appintheair/aita-ml-retentioneering-python.git
```
or
```bash
git clone https://github.com/appintheair/aita-ml-retentioneering-python.git
```
- Install dependencies from requirements.txt file from that directory
```bash
sudo pip install -r requirements.txt
```
- Then just run the setup.py file from that directory
```bash
sudo python setup.py install
```
## First steps
- Put path to your google cloud credentials and your project name in settings json ([example](tests/new_users_lost_prediction/settings.json)).
```json
"settings": {
    "service_account_path": "../credentials.json",
    "project": "project"
  }
```
- Now you are ready to do [examples](tests).
## Analysis
You can use retentioneering.analysis toolset with your data.

Data should have at least three columns `event_name`, `user_pseudo_id`, `event_timestamp`.

You can put empty dict as settings.

```python
import pandas as pd
from retentioneering import analysis
# your data import
df = pd.read_csv('path to your data')
settings = dict()
analysis.get_desc_table(df, settings=settings)
```
