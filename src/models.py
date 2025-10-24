from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import joblib
from typing import Tuple, Optional


class SentimentClassifier:
    """
    Generic class for sentiment classification.
    bear various of  models.
    """
    
    def __init__(self, model_type: str = 'logistic', **kwargs):
        """
         classifier init. 
        
        Args:
            model_type:  ('logistic', 'random_forest')
            **kwargs: addintional Parameters  for the  model
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        # Initialize  modèle 
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **kwargs
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f" uuh mdoel unknown: {model_type}")
    
    def train(self, X_train, y_train) -> 'SentimentClassifier':
    
        print(f"model training {self.model_type}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(" training done")
        return self
    
    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test) -> np.ndarray:
        """
        Returns prediction probailities.
        
        Args:
            X_test: Features de test
            
        Returns:
            Arrayprobabilities (shape: [n_samples, n_classes])
        """
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test, verbose: bool = True) -> dict:
        """
        Args:
            X_test: Features test
            y_test: true labels
            verbose: details
            
        Returns:
            Dictionnary with metrics (accuracy, precision, recall, f1)
        """
        if not self.is_trained:
            raise ValueError("train the mdoel before evaluation")
        
        predictions = self.predict(X_test)
        
        # 
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        if verbose:
            print("\n" + "="*50)
            print(f"ÉVALUATION - {self.model_type.upper()}")
            print("="*50)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print("\n classification report:")
            print(classification_report(y_test, predictions))
            print("\n confusion Matrix:")
            print(confusion_matrix(y_test, predictions))
            print("="*50 + "\n")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Args:
            filepath:  (.pkl)
        """
        
        joblib.dump(self.model, filepath)
        print(f"model saved at : {filepath}")
    
    @classmethod
    def load(cls, filepath: str, model_type: str = 'logistic') -> 'SentimentClassifier':
        """
        load the saved model.
        
        Args:
            filepath: 
            model_type: Type 
        
        """
        instance = cls(model_type=model_type)
        instance.model = joblib.load(filepath)
        instance.is_trained = True
        print(f" Modèle chargé: {filepath}")
        return instance