"""Historical USGS data access for the XGBoost replacement models.

Reuses evaluation/usgs_data.py's fetch_site_history rather than duplicating
the USGS RDB parsing/quality-code-filtering logic -- that module already
mirrors microservice/app.py's feature engineering exactly, which we want to
keep matching here too.
"""
import os
import sys

# append(), not insert(0, ...): both this directory and evaluation/ have a
# module named backtest.py. Inserting evaluation/ at the front of sys.path
# would shadow xgboost_model/backtest.py for any script that imports this
# module first -- append keeps the current directory's own modules winning.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))

from usgs_data import fetch_historical_series, fetch_site_history, get_site_coordinates  # noqa: E402
