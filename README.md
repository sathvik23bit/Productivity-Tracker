# URL Productivity Classifier

## Setup
```bash
pip install -r requirements.txt
```

## Run (train the model)
```bash
python url_classifier.py
```

## Test custom URLs
Edit test.py with your URLs, then:
```bash
python test.py
```

## Add your real Chrome history
1. Export Chrome history as CSV (use "Export Chrome History" extension)
2. Save as my_history.csv in this folder
3. Run:
```python
from url_classifier import auto_label
import pandas as pd
df = pd.read_csv("my_history.csv")
df["label"] = df.apply(lambda r: auto_label(r["url"], str(r.get("title",""))), axis=1)
df.to_csv("my_history_labeled.csv", index=False)
```
4. Retrain:
```python
from url_classifier import train
train(csv_path="my_history_labeled.csv")
```

## Project Structure
- url_classifier.py  → main ML code
- requirements.txt   → libraries
- test.py            → test your own URLs
- sample_browsing_data.csv  → auto-generated training data
- url_classifier.pkl        → saved trained model
