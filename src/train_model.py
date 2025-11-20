from __future__ import annotations
import argparse, sqlite3, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
def load_df(db: Path)->pd.DataFrame:
    con=sqlite3.connect(db)
    try: df=pd.read_sql_query('SELECT * FROM study_sessions', con)
    finally: con.close()
    return df
def main():
    ap=argparse.ArgumentParser(description='Train ML model to predict productivity (1..5)')
    ap.add_argument('--db',default='data/study.db'); ap.add_argument('--out',default='models/pipeline.joblib'); args=ap.parse_args()
    df=load_df(Path(args.db))
    if df.empty: raise SystemExit('No sessions found. Generate & ingest first.')
    y=df['productivity'].values; X=df[['duration_min','subject','technique','distractions','mood','caffeine_mg']].copy()
    cat=['subject','technique']; num=['duration_min','distractions','mood','caffeine_mg']
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num','passthrough',num)])
    model=RandomForestRegressor(n_estimators=200, random_state=42); pipe=Pipeline([('pre',pre),('rf',model)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    print(f'R2: {r2_score(yte,pred):.3f}\nMAE: {mean_absolute_error(yte,pred):.3f}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); joblib.dump(pipe, args.out); print(f'Saved pipeline -> {args.out}')
if __name__=='__main__': main()
print("hello")
