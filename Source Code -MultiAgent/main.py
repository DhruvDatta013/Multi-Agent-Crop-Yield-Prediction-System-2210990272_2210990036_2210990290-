# ==============================================================
#   INTELLIGENT MULTI-AGENT CROP YIELD PREDICTION SYSTEM
#   FILES NEEDED (all in same folder as main.py):
#   - yield.csv
#   - rainfall.csv
#   - temp.csv
#   - pesticides.csv
#   - yield_df.csv  (optional - used as fallback if merge fails)
#
#   HOW TO RUN:
#   Open folder in VS Code -> Terminal -> python main.py
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_FOLDER = "."        

os.makedirs("outputs", exist_ok=True)


class KnowledgeBase:
    def __init__(self):
        self.raw_data        = None
        self.X_train         = None
        self.X_test          = None
        self.y_train         = None
        self.y_test          = None
        self.best_model      = None
        self.best_model_name = ""
        self.predictions     = None
        self.metrics         = {}
        self.recommendations = []
        self.retrain_count   = 0
        self.feedback_log    = []
        self.all_results     = {}
        self.best_all_results = {}  # best across all retrain cycles
        self.initial_results  = {}  # results from first training run

KB = KnowledgeBase()


# ==============================================================
#  AGENT 1 — DATA COLLECTION AGENT
#  Loads all 4 raw CSV files and merges into one dataset
# ==============================================================
class DataCollectionAgent:
    def __init__(self, kb):
        self.kb = kb

    def _path(self, filename):
        return os.path.join(DATA_FOLDER, filename)

    def _load_and_merge(self):
        """Load yield, rainfall, temp, pesticides and merge on Area + Year."""

        print("  Loading raw files...")

        # -- yield.csv --
        yld = pd.read_csv(self._path("yield.csv"))
        yld = yld[yld['Element'] == 'Yield'][['Area', 'Item', 'Year', 'Value']].copy()
        yld.rename(columns={'Value': 'hg/ha_yield'}, inplace=True)
        print(f"    yield.csv        : {len(yld):,} rows  (after filtering Yield element)")

        # -- rainfall.csv --
        rain = pd.read_csv(self._path("rainfall.csv"))
        rain.rename(columns={' Area': 'Area'}, inplace=True)   # strip leading space
        rain = rain[['Area', 'Year', 'average_rain_fall_mm_per_year']]
        rain['average_rain_fall_mm_per_year'] = pd.to_numeric(rain['average_rain_fall_mm_per_year'], errors='coerce')
        print(f"    rainfall.csv     : {len(rain):,} rows")

        # -- pesticides.csv --
        pest = pd.read_csv(self._path("pesticides.csv"))
        pest = pest[['Area', 'Year', 'Value']].copy()
        pest.rename(columns={'Value': 'pesticides_tonnes'}, inplace=True)
        print(f"    pesticides.csv   : {len(pest):,} rows")

        # -- temp.csv --
        temp = pd.read_csv(self._path("temp.csv"))
        temp.rename(columns={'year': 'Year', 'country': 'Area'}, inplace=True)
        temp = temp[['Area', 'Year', 'avg_temp']]
        print(f"    temp.csv         : {len(temp):,} rows")

        # -- Merge all on Area + Year --
        df = yld.merge(rain, on=['Area', 'Year'], how='inner')
        df = df.merge(pest, on=['Area', 'Year'], how='inner')
        df = df.merge(temp, on=['Area', 'Year'], how='inner')

        print(f"\n  Merged dataset     : {len(df):,} rows  x  {len(df.columns)} columns")
        return df

    def _fallback_load(self):
        """Fallback: use pre-merged yield_df.csv if raw merge fails."""
        path = self._path("yield_df.csv")
        df   = pd.read_csv(path)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        print(f"  Fallback file      : yield_df.csv  ({len(df):,} rows)")
        return df

    def run(self):
        print("\n" + "="*65)
        print("  AGENT 1 — DATA COLLECTION AGENT")
        print("="*65)

        # Check which files exist
        raw_files = ["yield.csv", "rainfall.csv", "temp.csv", "pesticides.csv"]
        all_exist  = all(os.path.exists(self._path(f)) for f in raw_files)

        if all_exist:
            print("  Mode: merging 4 raw files (yield + rainfall + temp + pesticides)\n")
            df = self._load_and_merge()
        elif os.path.exists(self._path("yield_df.csv")):
            print("  Mode: loading pre-merged yield_df.csv\n")
            df = self._fallback_load()
        else:
            raise FileNotFoundError(
                "\n  ERROR: No data files found.\n"
                "  Place your CSV files in the same folder as main.py\n"
                "  Required: yield.csv, rainfall.csv, temp.csv, pesticides.csv\n"
                "  OR:       yield_df.csv (pre-merged)")

        # Validity checks
        original_rows = len(df)
        df.drop_duplicates(inplace=True)
        dupes_removed = original_rows - len(df)
        missing_pct   = df.isnull().mean().mean() * 100

        print(f"\n  Raw records        : {original_rows:,}")
        print(f"  Duplicates removed : {dupes_removed:,}")
        print(f"  Clean records      : {len(df):,}")
        print(f"  Missing values     : {missing_pct:.1f}%")
        print(f"  Final columns      : {list(df.columns)}")

        self.kb.raw_data = df
        print("\n  [Agent 1] COMPLETE")


# ==============================================================
#  AGENT 2 — DATA PREPROCESSING AGENT
# ==============================================================
class DataPreprocessingAgent:
    def __init__(self, kb):
        self.kb     = kb
        self.scaler = MinMaxScaler()

    def run(self):
        print("\n" + "="*65)
        print("  AGENT 2 — DATA PREPROCESSING AGENT")
        print("="*65)

        df = self.kb.raw_data.copy()

        # Auto-detect target column
        target_candidates = ['hg/ha_yield', 'Yield', 'yield', 'Crop_Yield', 'Production']
        target_col = next((t for t in target_candidates if t in df.columns), df.columns[-1])
        print(f"  Target column      : '{target_col}'")

        y = df[target_col].copy()
        X = df.drop(columns=[target_col])

        # One-hot encode categoricals
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"  Categorical cols   : {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
        print(f"  Features after OHE : {X.shape[1]}")

        # kNN imputation
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            imputer = KNNImputer(n_neighbors=5)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            print(f"  kNN imputed        : {missing_before} missing values filled")
        else:
            print(f"  Missing values     : None — skipping imputation")

        # Min-Max normalisation
        num_cols    = X.select_dtypes(include=[np.number]).columns.tolist()
        X[num_cols] = self.scaler.fit_transform(X[num_cols])
        print(f"  Normalised         : {len(num_cols)} features scaled to [0, 1]")

        # Variance filter
        vt    = VarianceThreshold(threshold=0.001)
        X_arr = vt.fit_transform(X)
        kept  = X.columns[vt.get_support()].tolist()
        X     = pd.DataFrame(X_arr, columns=kept)
        print(f"  Features (final)   : {X.shape[1]}")

        # 70/30 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42)
        print(f"  Train / Test       : {len(X_train):,} / {len(X_test):,}  (70:30)")

        self.kb.X_train = X_train
        self.kb.X_test  = X_test
        self.kb.y_train = y_train
        self.kb.y_test  = y_test
        print("\n  [Agent 2] COMPLETE")


# ==============================================================
#  AGENT 3 — LEARNING AGENT
# ==============================================================
class LearningAgent:
    def __init__(self, kb, rf_params=None, svm_params=None):
        self.kb         = kb
        self.rf_params  = rf_params  or {'n_estimators': 200, 'max_depth': None,
                                         'random_state': 42, 'n_jobs': -1}
        self.svm_params = svm_params or {'C': 1.0, 'max_iter': 5000, 'random_state': 42}

    def _evaluate(self, y_true, y_pred, name):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        acc  = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9) <= 0.15) * 100
        print(f"  {name:<42} Acc={acc:5.1f}%  MAE={mae:8.0f}  RMSE={rmse:8.0f}  R2={r2:.4f}")
        return {'accuracy': acc, 'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': y_pred}

    def run(self):
        print("\n" + "="*65)
        print("  AGENT 3 — LEARNING AGENT")
        print("="*65)
        print(f"\n  Training on {len(self.kb.X_train):,} samples...\n")

        results = {}
        X_train, X_test = self.kb.X_train, self.kb.X_test
        y_train, y_test = self.kb.y_train, self.kb.y_test

        # Linear Regression (baseline)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results['Linear Regression (Baseline)'] = self._evaluate(
            y_test, lr.predict(X_test), 'Linear Regression (Baseline)')

        # Random Forest without agent
        rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_base.fit(X_train, y_train)
        results['Random Forest (Without Agent)'] = self._evaluate(
            y_test, rf_base.predict(X_test), 'Random Forest (Without Agent)')

        # Agent-Based Random Forest
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train, y_train)
        rf_result = self._evaluate(y_test, rf.predict(X_test), 'Agent-Based Random Forest')
        results['Agent-Based Random Forest'] = rf_result

        # Agent-Based SVM (LinearSVR)
        print(f"  {'Agent-Based SVM (LinearSVR)':<42} training...", end='', flush=True)
        svm = LinearSVR(**self.svm_params)
        svm.fit(X_train, y_train)
        svm_result = self._evaluate(y_test, svm.predict(X_test), 'Agent-Based SVM (LinearSVR)')
        results['Agent-Based SVM (LinearSVR)'] = svm_result

        # Select best model by RMSE
        agent_models = {
            'Agent-Based Random Forest'  : (rf,  rf_result),
            'Agent-Based SVM (LinearSVR)': (svm, svm_result),
        }
        best_name          = min(agent_models, key=lambda k: agent_models[k][1]['rmse'])
        best_model, best_r = agent_models[best_name]

        self.kb.best_model      = best_model
        self.kb.best_model_name = best_name
        self.kb.predictions     = best_r['predictions']
        self.kb.metrics         = best_r
        self.kb.all_results     = results

        print(f"\n  Best model selected : {best_name}")
        print("\n  [Agent 3] COMPLETE")


# ==============================================================
#  AGENT 4 — DECISION-MAKING AGENT
# ==============================================================
class DecisionMakingAgent:
    THRESHOLDS = {'high': 30000, 'medium': 15000}

    def __init__(self, kb):
        self.kb = kb

    def _advise(self, pred_yield, idx):
        if pred_yield >= self.THRESHOLDS['high']:
            return {'sample_idx': idx,
                    'predicted_yield': round(pred_yield, 2),
                    'yield_level': 'HIGH',
                    'irrigation': 'Maintain current irrigation schedule.',
                    'fertilizer': 'Reduce nitrogen input by 10%.',
                    'harvest':    'Prepare for early harvest.'}
        elif pred_yield >= self.THRESHOLDS['medium']:
            return {'sample_idx': idx,
                    'predicted_yield': round(pred_yield, 2),
                    'yield_level': 'MEDIUM',
                    'irrigation': 'Increase irrigation frequency by 15%.',
                    'fertilizer': 'Apply standard NPK mix.',
                    'harvest':    'Monitor crop maturity weekly.'}
        else:
            return {'sample_idx': idx,
                    'predicted_yield': round(pred_yield, 2),
                    'yield_level': 'LOW',
                    'irrigation': 'Urgent: moisture deficit likely.',
                    'fertilizer': 'Supplement with phosphorus and potassium.',
                    'harvest':    'Review soil health; consider crop rotation.'}

    def run(self):
        print("\n" + "="*65)
        print("  AGENT 4 — DECISION-MAKING AGENT")
        print("="*65)

        recs   = [self._advise(p, i) for i, p in enumerate(self.kb.predictions)]
        levels = [r['yield_level'] for r in recs]
        total  = len(levels)
        high   = levels.count('HIGH')
        med    = levels.count('MEDIUM')
        low    = levels.count('LOW')

        self.kb.recommendations = recs

        print(f"  Total test samples : {total:,}")
        print(f"  HIGH  yield        : {high:,}  ({high/total*100:.1f}%)  >= 30,000 hg/ha")
        print(f"  MEDIUM yield       : {med:,}  ({med/total*100:.1f}%)  15,000-29,999 hg/ha")
        print(f"  LOW    yield       : {low:,}  ({low/total*100:.1f}%)  < 15,000 hg/ha")
        print(f"  Confidence rate    : {(high+med)/total*100:.1f}%  (HIGH + MEDIUM combined)")
        print("\n  [Agent 4] COMPLETE")


# ==============================================================
#  AGENT 5 — FEEDBACK AGENT
# ==============================================================
class FeedbackAgent:
    MAE_THRESHOLD  = 5000.0
    RMSE_THRESHOLD = 8000.0
    MAX_RETRAIN    = 5

    def __init__(self, kb, learning_agent):
        self.kb = kb
        self.la = learning_agent

    def run(self):
        print("\n" + "="*65)
        print("  AGENT 5 — FEEDBACK AGENT")
        print("="*65)

        mae  = self.kb.metrics['mae']
        rmse = self.kb.metrics['rmse']
        print(f"  MAE  threshold : <= {self.MAE_THRESHOLD:.0f} hg/ha  |  Current: {mae:.0f}")
        print(f"  RMSE threshold : <= {self.RMSE_THRESHOLD:.0f} hg/ha  |  Current: {rmse:.0f}")

        retrain_params = [
            {'n_estimators': 300, 'max_depth': 20,
             'min_samples_split': 3, 'random_state': 42, 'n_jobs': -1},
            {'n_estimators': 400, 'max_depth': None,
             'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1},
            {'n_estimators': 500, 'max_depth': 30,
             'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1},
        ]
        cycle = 0
        while ((mae > self.MAE_THRESHOLD or rmse > self.RMSE_THRESHOLD)
               and self.kb.retrain_count < self.MAX_RETRAIN):
            params = retrain_params[min(cycle, len(retrain_params) - 1)]
            self.kb.retrain_count += 1
            cycle += 1
            print(f"\n  Threshold exceeded -> Retraining cycle {self.kb.retrain_count}")
            print(f"  New RF params      : n_estimators={params['n_estimators']}, "
                  f"max_depth={params.get('max_depth','None')}")
            self.la.rf_params = params
            self.la.run()
            mae  = self.kb.metrics['mae']
            rmse = self.kb.metrics['rmse']
            self.kb.feedback_log.append(
                {'cycle': self.kb.retrain_count, 'mae': mae, 'rmse': rmse})

        if mae <= self.MAE_THRESHOLD and rmse <= self.RMSE_THRESHOLD:
            print(f"\n  Within thresholds after {self.kb.retrain_count} retrain cycle(s).")
        else:
            print(f"\n  Max retraining cycles reached ({self.MAX_RETRAIN}).")
            print(f"  Final MAE={mae:.0f}  RMSE={rmse:.0f}")
        print("\n  [Agent 5] COMPLETE")


# ==============================================================
#  FINAL REPORT
# ==============================================================
def print_final_report(kb):
    print("\n\n" + "="*65)
    print("  FINAL RESULTS SUMMARY")
    print("="*65)
    print(f"  {'Model':<42} {'Acc':>6}  {'MAE':>8}  {'RMSE':>8}  {'R2':>7}")
    print("  " + "-"*63)
    # Use results from initial training run (before feedback retraining)
    results_to_show = kb.initial_results if kb.initial_results else kb.all_results
    order = ['Linear Regression (Baseline)',
             'Random Forest (Without Agent)',
             'Agent-Based Random Forest',
             'Agent-Based SVM (LinearSVR)']
    for name in order:
        if name not in kb.all_results:
            continue
        m      = results_to_show.get(name) or kb.all_results.get(name)
        if not m: continue
        marker = "  <-- BEST" if name == kb.best_model_name else ""
        print(f"  {name:<42} {m['accuracy']:>5.1f}%  "
              f"{m['mae']:>8.0f}  {m['rmse']:>8.0f}  {m['r2']:>7.4f}{marker}")
    print("="*65)
    print(f"\n  Best model     : {kb.best_model_name}")
    print(f"  Retrain cycles : {kb.retrain_count}")
    print(f"  Plots saved to : outputs/ folder\n")


# ==============================================================
#  PLOTS
# ==============================================================
def generate_plots(kb):
    print("  Generating plots...\n")
    results = kb.all_results
    names   = list(results.keys())
    short   = [n.replace('Agent-Based ', 'Agent\n')
                .replace(' (Baseline)',     '\n(Baseline)')
                .replace(' (Without Agent)','\n(No Agent)')
                .replace(' (LinearSVR)',    '\n(LinearSVR)') for n in names]

    # -- Plot 1: Model Comparison --
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance Comparison", fontsize=13, fontweight='bold')
    for ax, (key, ylabel, color) in zip(axes, [
        ('accuracy', 'Accuracy (%)',  '#2471a3'),
        ('mae',      'MAE (hg/ha)',   '#c0392b'),
        ('rmse',     'RMSE (hg/ha)',  '#1e8449')]):
        vals = [results[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=color,
                      alpha=0.85, edgecolor='black', linewidth=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f'{val:.0f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/plot1_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: outputs/plot1_model_comparison.png")

    # -- Plot 2: Actual vs Predicted --
    fig, ax = plt.subplots(figsize=(7, 6))
    y_test = kb.y_test.values
    y_pred = kb.predictions
    ax.scatter(y_test, y_pred, alpha=0.25, s=8, color='#2471a3', edgecolors='none')
    lim = [min(y_test.min(), y_pred.min()) - 500,
           max(y_test.max(), y_pred.max()) + 500]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Yield (hg/ha)", fontsize=11)
    ax.set_ylabel("Predicted Yield (hg/ha)", fontsize=11)
    ax.set_title(f"Actual vs Predicted — {kb.best_model_name}",
                 fontsize=11, fontweight='bold')
    ax.text(0.05, 0.92, f"R2 = {kb.metrics['r2']:.4f}",
            transform=ax.transAxes, fontsize=11,
            color='darkred', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/plot2_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: outputs/plot2_actual_vs_predicted.png")

    # -- Plot 3: Feature Importance --
    if hasattr(kb.best_model, 'feature_importances_'):
        fi = pd.Series(kb.best_model.feature_importances_,
                       index=kb.X_train.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        fi.head(15).plot(kind='barh', ax=ax, color='#117a65',
                         edgecolor='black', linewidth=0.5)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance Score", fontsize=11)
        ax.set_title(f"Top 15 Feature Importances — {kb.best_model_name}",
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/plot3_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  Saved: outputs/plot3_feature_importance.png")

    # -- Plot 4: Feedback Loop --
    if kb.feedback_log:
        cycles = [0] + [e['cycle'] for e in kb.feedback_log]
        rmse_v = [kb.all_results['Agent-Based Random Forest']['rmse']] + \
                 [e['rmse'] for e in kb.feedback_log]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cycles, rmse_v, 'o-', color='purple', linewidth=2, markersize=8)
        ax.axhline(y=FeedbackAgent.RMSE_THRESHOLD, color='red',
                   linestyle='--', label=f'Threshold = {FeedbackAgent.RMSE_THRESHOLD:.0f}')
        ax.set_xlabel("Retraining Cycle", fontsize=11)
        ax.set_ylabel("RMSE (hg/ha)", fontsize=11)
        ax.set_title("Feedback Agent — RMSE over Retraining Cycles",
                     fontsize=11, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/plot4_feedback_loop.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  Saved: outputs/plot4_feedback_loop.png")


# ==============================================================
#  MAIN — ORCHESTRATE ALL 5 AGENTS
# ==============================================================
if __name__ == "__main__":

    print("\n" + "="*65)
    print("  INTELLIGENT MULTI-AGENT CROP YIELD PREDICTION SYSTEM")
    print("  Chitkara University — Computer Science Dept.")
    print("="*65)

    dca = DataCollectionAgent(KB)
    ppa = DataPreprocessingAgent(KB)
    la  = LearningAgent(KB)
    dma = DecisionMakingAgent(KB)
    fa  = FeedbackAgent(KB, la)

    dca.run()    # Agent 1 — loads & merges all 4 CSVs
    ppa.run()    # Agent 2 — preprocess
    la.run()     # Agent 3 — train models
    KB.initial_results = dict(KB.all_results)  # save before feedback rewrites
    dma.run()    # Agent 4 — decision advisories
    fa.run()     # Agent 5 — feedback & retrain

    print_final_report(KB)
    generate_plots(KB)

    print("\n" + "="*65)
    print("  ALL DONE — plots saved to outputs/ folder")
    print("="*65 + "\n")
