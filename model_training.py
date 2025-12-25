import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class BettingModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def load_data(self):
        """Load historical game data"""
        print("\n1. Loading training data...")
        
        if not os.path.exists(config.HISTORICAL_DATA_PATH):
            print(f"✗ Training data not found at {config.HISTORICAL_DATA_PATH}")
            print("Run 'python src/data_generator.py' first")
            return None, None
        
        df = pd.read_csv(config.HISTORICAL_DATA_PATH)
        print(f"✓ Loaded {len(df)} games")
        
        # Define features
        feature_columns = [
            'home_ppg', 'away_ppg',
            'home_def_rating', 'away_def_rating',
            'home_form_l10', 'away_form_l10',
            'home_rest_days', 'away_rest_days',
            'home_injury_impact', 'away_injury_impact',
            'pace',
            'home_3pt_pct', 'away_3pt_pct',
            'is_home'
        ]
        
        X = df[feature_columns]
        y = df['home_win']
        
        self.feature_names = feature_columns
        
        return X, y
    
    def train(self, X, y):
        """Train XGBoost model with time-series cross-validation"""
        print("\n2. Training model...")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Model parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Cross-validation scores
        cv_scores = []
        cv_acc = []
        cv_auc = []
        
        print("\nPerforming 5-fold time-series cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Metrics
            ll = log_loss(y_val, y_pred_proba)
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            cv_scores.append(ll)
            cv_acc.append(acc)
            cv_auc.append(auc)
            
            print(f"  Fold {fold}: Log Loss={ll:.4f}, Accuracy={acc:.4f}, AUC={auc:.4f}")
        
        print(f"\nAverage CV Log Loss: {np.mean(cv_scores):.4f}")
        print(f"Average CV Accuracy: {np.mean(cv_acc):.4f} ({np.mean(cv_acc)*100:.1f}%)")
        print(f"Average CV AUC: {np.mean(cv_auc):.4f}")
        
        # Train final model on all data
        print("\n3. Training final model on full dataset...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y, verbose=False)
        
        # Final evaluation
        final_pred_proba = self.model.predict_proba(X)[:, 1]
        final_pred = self.model.predict(X)
        
        final_acc = accuracy_score(y, final_pred)
        final_auc = roc_auc_score(y, final_pred_proba)
        
        print(f"✓ Final model accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
        print(f"✓ Final model AUC: {final_auc:.4f}")
        
        return np.mean(cv_acc)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self):
        """Save trained model"""
        os.makedirs('models', exist_ok=True)
        model_path = 'models/betting_model.pkl'
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, model_path)
        
        print(f"\n✓ Model saved to: {model_path}")
    
    def load_model(self):
        """Load trained model"""
        model_path = 'models/betting_model.pkl'
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found at {model_path}")
            return False
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        
        print(f"✓ Model loaded from: {model_path}")
        return True
    
    def predict(self, features):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        prob = self.model.predict_proba(features)[0, 1]
        return prob

def main():
    """Main training pipeline"""
    
    print("SPORTS BETTING MODEL TRAINING")
    
    
    model = BettingModel()
    
    # Load data
    X, y = model.load_data()
    
    if X is None:
        return
    
    # Train model
    cv_accuracy = model.train(X, y)
    
    # Feature importance
    print("\n4. Feature importance:")
    print("=" * 60)
    importance_df = model.get_feature_importance()
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Save model
    print("\n5. Saving model...")
    model.save_model()
    
    
    print("✓ TRAINING COMPLETE!")
    
    print(f"Model accuracy: {cv_accuracy*100:.1f}%")
    

if __name__ == "__main__":
    main()