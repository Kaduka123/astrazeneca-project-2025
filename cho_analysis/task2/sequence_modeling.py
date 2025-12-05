# cho_analysis/task2/sequence_modeling.py
"""Sequence-based expression prediction modeling.

This module provides functionality to build predictive models of gene expression
from sequence features, analyze feature importance, and simulate sequence optimization.
"""

import logging
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="The objective has been validated", module="sklearn")


class SequenceExpressionModeling:
    """Builds predictive models of gene expression from sequence features."""

    def __init__(self):
        """Initialize the modeling class."""
        self.results: dict[str, Any] = {}
        self.models: dict[str, Any] = {}
        self.feature_importances: dict[str, float] = {}
        self.scaler = None

    def select_predictive_features(
        self,
        feature_df: pd.DataFrame,
        expression_metric: str = "cv",
        selection_method: str = "statistical",
        max_features: int = 15,
    ) -> pd.DataFrame:
        """Select optimal subset of features for prediction.

        Args:
            feature_df: DataFrame with sequence features
            expression_metric: Target metric to predict ('cv' or 'mean')
            selection_method: Feature selection method(s) ('statistical', 'rfe', 'lasso' or comma-separated combination)
            max_features: Maximum number of features to select

        Returns:
            DataFrame with selected features and importance scores
        """
        logger.info(
            f"Selecting predictive features for {expression_metric} using {selection_method}..."
        )

        if feature_df.empty:
            logger.error("Empty feature DataFrame provided")
            return pd.DataFrame()

        # Map target names to ensure compatibility
        target_map = {"mean": "mean_expression", "cv": "cv"}
        actual_target = target_map.get(expression_metric, expression_metric)

        # Check target column
        if actual_target not in feature_df.columns:
            logger.error(f"Target column {actual_target} not found in DataFrame. Available columns: {list(feature_df.columns)}")
            return pd.DataFrame()

        # Get feature columns (numeric only, excluding target and other metrics)
        exclude_cols = ["cv", "mean_expression", "std", "min", "max", "ensembl_transcript_id", "symbol", "stability_category"]
        feature_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in exclude_cols]

        if not feature_cols:
            logger.error("No numeric feature columns found")
            return pd.DataFrame()

        # Prepare data
        X = feature_df[feature_cols].copy()
        y = feature_df[actual_target].copy()

        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)

        # Use median imputation for missing values
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Check if multiple methods are specified (comma-separated)
        selection_methods = [method.strip() for method in selection_method.split(",")]
        all_selected_features = []
        all_importance_scores = []

        # Run each selection method
        for method in selection_methods:
            try:
                selected_features, importance_scores = self._run_single_feature_selection(
                    X_scaled, y, method, max_features, feature_cols
                )

                if selected_features:
                    all_selected_features.extend(selected_features)
                    all_importance_scores.extend(importance_scores)
            except Exception as e:
                logger.exception(f"Error during feature selection with method {method}: {e}")

        # Remove duplicates while preserving order
        unique_selected_features = []
        unique_importance_scores = []
        seen_features = set()

        for feat, score in zip(all_selected_features, all_importance_scores, strict=False):
            if feat not in seen_features:
                unique_selected_features.append(feat)
                unique_importance_scores.append(score)
                seen_features.add(feat)

        if not unique_selected_features:
            logger.warning(f"No features selected using methods: {selection_method}")
            return pd.DataFrame()

        # Create results DataFrame
        result_df = pd.DataFrame(
            {"feature": unique_selected_features, "importance_score": unique_importance_scores}
        )

        # Normalize importance scores
        if len(unique_importance_scores) > 0:
            max_score = max(unique_importance_scores)
            if max_score > 0:
                result_df["normalized_importance"] = result_df["importance_score"] / max_score

        # Sort by importance
        result_df = result_df.sort_values("importance_score", ascending=False)

        # Limit to max_features
        if len(result_df) > max_features:
            result_df = result_df.head(max_features)

        logger.info(f"Selected {len(result_df)} features using {selection_method}")
        self.results["selected_features"] = result_df

        return result_df

    def _run_single_feature_selection(
        self,
        X_scaled: pd.DataFrame,
        y: pd.Series,
        selection_method: str,
        max_features: int,
        feature_cols: list[str],
    ) -> tuple[list[str], list[float]]:
        """Run a single feature selection method.

        Args:
            X_scaled: Scaled feature matrix
            y: Target values
            selection_method: Feature selection method
            max_features: Maximum number of features to select
            feature_cols: List of all feature column names

        Returns:
            Tuple of (selected_features, importance_scores)
        """
        selected_features: list[str] = []
        importance_scores: list[float] = []

        if selection_method == "statistical":
            # F-regression for continuous target
            selector = SelectKBest(f_regression, k=min(max_features, len(feature_cols)))
            selector.fit(X_scaled, y)

            # Get selected features and scores
            selected_mask = selector.get_support()

            # Safely get selected features and their importance scores
            for i, is_selected in enumerate(selected_mask):
                if is_selected:
                    # Fix for linter error - get string from index
                    col_name = str(X_scaled.columns[i])
                    selected_features.append(col_name)
                    importance_scores.append(float(selector.scores_[i]))

        elif selection_method == "rfe":
            # Recursive Feature Elimination with Cross-Validation
            estimator = Ridge(alpha=1.0)
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=5,
                scoring="neg_mean_squared_error",
                min_features_to_select=1,
                n_jobs=-1,
            )
            selector.fit(X_scaled, y)

            # Get selected features and importance
            selected_mask = selector.support_
            selected_features = [str(col) for col in X_scaled.columns[selected_mask].tolist()]

            # Train a model on selected features to get importances
            if selected_features:
                model = Ridge(alpha=1.0)
                model.fit(X_scaled[selected_features], y)
                importance_scores = np.abs(model.coef_).tolist()

        elif selection_method == "lasso":
            # Lasso for inherent feature selection
            alpha = 0.01  # Can be optimized with cross-validation
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_scaled, y)

            # Get non-zero coefficients
            feature_importance = np.abs(lasso.coef_)
            selected_mask = feature_importance > 0
            selected_features = [str(col) for col in X_scaled.columns[selected_mask].tolist()]
            importance_scores = feature_importance[selected_mask].tolist()

        else:
            logger.error(f"Unknown feature selection method: {selection_method}")
            return [], []

        return selected_features, importance_scores

    def build_expression_prediction_model(
        self,
        feature_df: pd.DataFrame,
        target: str = "cv",
        model_type: str = "ensemble",
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Train models to predict expression metrics from sequence features.

        Args:
            feature_df: DataFrame with sequence features
            target: Target metric to predict ('cv' or 'mean')
            model_type: Model type or comma-separated list of model types ('linear', 'tree', 'ensemble')
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with model performance metrics
        """
        logger.info(f"Building {model_type} model to predict {target}...")

        # Validate input data
        if feature_df is None or feature_df.empty:
            logger.error("Empty feature DataFrame provided")
            return {}

        # Map target names to ensure compatibility
        target_map = {"mean": "mean_expression", "cv": "cv"}
        actual_target = target_map.get(target, target)

        # Check target column
        if actual_target not in feature_df.columns:
            logger.error(f"Target column '{actual_target}' not found in DataFrame. Available columns: {list(feature_df.columns)}")
            return {}

        # Initialize model dictionary to store the trained models
        self.models = {}

        # Store the target for later use
        self.target_column = actual_target

        # Get feature columns (numeric only, excluding target and other metrics)
        exclude_cols = ["cv", "mean_expression", "std", "min", "max", "ensembl_transcript_id", "symbol", "stability_category"]
        feature_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in exclude_cols]

        if not feature_cols:
            logger.error("No numeric feature columns found for model training")
            return {}

        logger.info(f"Using {len(feature_cols)} numeric feature columns for initial feature selection")

        # Use selected features if available
        if "selected_features" in self.results and not self.results["selected_features"].empty:
            selected_features = self.results["selected_features"]["feature"].tolist()
            # Ensure all selected features are available in the data
            available_selected = [f for f in selected_features if f in feature_cols]

            if available_selected:
                logger.info(f"Using {len(available_selected)} pre-selected features")
                feature_cols = available_selected
            else:
                logger.warning("None of the pre-selected features were found in the data")

        # Prepare data
        X = feature_df[feature_cols].copy()
        y = feature_df[actual_target].copy()

        # Check for sufficient data
        if len(X) < cv_folds * 2:
            logger.error(f"Insufficient data points ({len(X)}) for {cv_folds}-fold cross-validation")
            return {}

        # Save initial feature data statistics for consistent preprocessing during prediction
        self.feature_stats = {
            'features': feature_cols,
            'medians': {col: X[col].median() for col in feature_cols},
            'means': {col: X[col].mean() for col in feature_cols},
            'std_devs': {col: X[col].std() for col in feature_cols},
            'min_values': {col: X[col].min() for col in feature_cols},
            'max_values': {col: X[col].max() for col in feature_cols},
        }

        # Handle missing values consistently - store and save the process
        X = X.replace([np.inf, -np.inf], np.nan)

        # Calculate NaN statistics - properly handle pandas Series
        total_nan_count = X.isna().sum().sum()
        if total_nan_count > 0:
            # Calculate percentage of NaN values
            nan_percentage = float((total_nan_count / (X.shape[0] * X.shape[1])) * 100)
            logger.warning(f"Data contains {total_nan_count} NaN values ({nan_percentage:.2f}% of all values)")

            # Check if any columns have too many NaNs (>50%)
            columns_to_check = X.columns.tolist()
            high_nan_cols = []
            nan_counts = X.isna().sum()

            for col in columns_to_check:
                # Fix for linter error - use Series directly
                if nan_counts[col] > len(X) * 0.5:
                    high_nan_cols.append(col)

            if high_nan_cols:
                logger.warning(f"Columns with >50% NaN values: {high_nan_cols}")
                # Optionally drop these columns
                X = X.drop(columns=high_nan_cols)
                feature_cols = [col for col in feature_cols if col not in high_nan_cols]
                # Update feature stats after dropping columns
                for col in high_nan_cols:
                    if col in self.feature_stats['features']:
                        self.feature_stats['features'].remove(col)
                        del self.feature_stats['medians'][col]
                        del self.feature_stats['means'][col]
                        del self.feature_stats['std_devs'][col]
                        del self.feature_stats['min_values'][col]
                        del self.feature_stats['max_values'][col]

                logger.info(f"Dropped {len(high_nan_cols)} columns with too many NaNs, {len(feature_cols)} columns remaining")

                if len(feature_cols) < 2:
                    logger.error("Insufficient features remaining after dropping high-NaN columns")
                    return {}

        # Use median imputation for missing values - store the medians for prediction time
        self.feature_medians = {}
        for col in X.columns:
            median_val = X[col].median()
            self.feature_medians[col] = median_val
            X[col] = X[col].fillna(median_val)

        # Check for any remaining NaNs using numpy to avoid linter errors
        if np.isnan(X.values).any():
            logger.error("Failed to impute all NaN values. Check data.")
            return {}

        # Split data for validation - save the indices for tracking
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.train_indices = X_train.index.tolist()
            self.test_indices = X_test.index.tolist()
        except Exception as e:
            logger.exception(f"Error splitting data: {e}")
            return {}

        # Create scaler - save for reuse during prediction
        try:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            # Save the scaler mean and std for debugging
            self.scaler_means = self.scaler.mean_.copy()
            self.scaler_stds = self.scaler.scale_.copy()
            # Check for extremely small standard deviations that could cause scaling issues
            small_std_features = [
                (col, std) for col, std in zip(X_train.columns, self.scaler.scale_)
                if std < 1e-6
            ]
            if small_std_features:
                logger.warning(f"Features with very small std dev (potential scaling issues): {small_std_features}")

            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            logger.exception(f"Error scaling data: {e}")
            return {}

        # Save the feature columns for future predictions
        self.feature_cols = feature_cols
        logger.info(f"Using {len(self.feature_cols)} features for model training")

        # Check if multiple model types are specified (comma-separated)
        model_types = [mt.strip() for mt in model_type.split(",")]

        # Track best model performance
        best_model_r2 = -float("inf")
        best_model_type = None
        all_results = []

        # Try each model type
        for mt in model_types:
            try:
                # Initialize model based on type
                if mt == "linear":
                    model = Ridge(alpha=1.0, random_state=42)
                    models = {"ridge": model}
                elif mt == "tree":
                    model = GradientBoostingRegressor(
                        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
                    )
                    models = {"gbr": model}
                elif mt == "ensemble":
                    # Simple ensemble: averaging predictions from Ridge and GradientBoosting
                    models = {
                        "ridge": Ridge(alpha=1.0, random_state=42),
                        "gbr": GradientBoostingRegressor(
                            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
                        ),
                    }
                    model = models["ridge"]  # Just for initialization
                else:
                    logger.warning(f"Skipping unknown model type: {mt}")
                    continue

                # Cross-validation
                cv = KFold(n_splits=min(cv_folds, len(X_train)), shuffle=True, random_state=42)
                cv_scores: dict[str, np.ndarray | list[float]] = {}

                if mt == "ensemble":
                    # Train each model in the ensemble
                    ensemble_preds_test = np.zeros_like(y_test.values)
                    ensemble_models = {}

                    for name, m in models.items():
                        try:
                            # Cross-validation
                            cv_scores[name] = cross_val_score(
                                m, X_train_scaled, y_train, cv=cv, scoring="r2"
                            )

                            # Train on full training set
                            m.fit(X_train_scaled, y_train)

                            # Predict on test set
                            y_pred = m.predict(X_test_scaled)
                            ensemble_preds_test += y_pred

                            # Store the trained model
                            ensemble_models[name] = m

                        except Exception as e:
                            logger.exception(f"Error training {name} model: {e}")

                    # Check if any models were trained
                    if not ensemble_models:
                        logger.error(f"No models were successfully trained for {mt}")
                        continue

                    # Average the predictions
                    ensemble_preds_test /= len(ensemble_models)

                    # Calculate metrics for ensemble
                    test_r2 = r2_score(y_test, ensemble_preds_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds_test))

                    # Store ensemble CV scores as average of individual model scores
                    cv_scores["ensemble"] = [np.mean([np.mean(cv_scores[m]) for m in ensemble_models])]

                    # Save models
                    models = ensemble_models
                    models["ensemble_avg"] = "average"  # Marker for prediction function
                else:
                    # Single model
                    try:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="r2")

                        # Train on full training set
                        model.fit(X_train_scaled, y_train)

                        # Predict on test set
                        y_pred = model.predict(X_test_scaled)

                        # Calculate metrics
                        test_r2 = r2_score(y_test, y_pred)
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                        # Store the trained model
                        models = {mt: model}
                    except Exception as e:
                        logger.exception(f"Error training model {mt}: {e}")
                        continue

                # Create results dictionary
                if isinstance(cv_scores, dict):
                    # Ensemble model
                    cv_r2_mean = np.mean([np.mean(scores) for scores in cv_scores.values()])
                    cv_r2_std = np.mean([np.std(scores) for scores in cv_scores.values()])
                else:
                    # Single model
                    cv_r2_mean = np.mean(cv_scores)
                    cv_r2_std = np.std(cv_scores)

                results = {
                    "model_type": mt,
                    "cv_r2_mean": float(cv_r2_mean),
                    "cv_r2_std": float(cv_r2_std),
                    "test_r2": float(test_r2),
                    "test_rmse": float(test_rmse),
                    "n_features": len(feature_cols),
                    "n_samples": len(X),
                    "feature_names": feature_cols,
                    "target": target,
                }

                all_results.append(results)

                logger.info(
                    f"Model ({mt}) performance: CV R² = {cv_r2_mean:.3f} ± {cv_r2_std:.3f}, Test R² = {test_r2:.3f}"
                )

                # Track the best performing model
                if test_r2 > best_model_r2:
                    best_model_r2 = test_r2
                    best_model_type = mt
                    self.models = models  # Save the best models

            except Exception as e:
                logger.exception(f"Error building model type {mt}: {e}")
                continue

        # If no models were successfully trained
        if not self.models:
            logger.error("No models could be trained successfully")
            return {}

        # Store training data for consistency checks
        self.results["original_data"] = feature_df
        self.results["X_train"] = X_train
        self.results["y_train"] = y_train
        self.results["X_test"] = X_test
        self.results["y_test"] = y_test
        self.results["X_train_scaled"] = X_train_scaled
        self.results["X_test_scaled"] = X_test_scaled

        # Create a combined results DataFrame if we tried multiple models
        if len(all_results) > 1:
            combined_results = pd.DataFrame(all_results)
            logger.info(f"Trained {len(all_results)} model types. Best: {best_model_type} (R²={best_model_r2:.3f})")
            self.results["all_models_performance"] = combined_results

            # Use the best model's results
            best_result = next((r for r in all_results if r["model_type"] == best_model_type), all_results[0])
            self.results["model_performance"] = best_result

            # Calculate feature importance for the best model
            self.analyze_feature_importance(self.models, feature_cols)

            return best_result
        elif len(all_results) == 1:
            self.results["model_performance"] = all_results[0]

            # Calculate feature importance
            self.analyze_feature_importance(self.models, feature_cols)

            return all_results[0]
        else:
            return {}

    def predict_expression_from_sequence(
        self, sequence_features: pd.DataFrame, target: str = "cv"
    ) -> pd.DataFrame:
        """Predict expression metrics from sequence features using trained models.

        Args:
            sequence_features: DataFrame with sequence features to predict from
            target: Target metric to predict ('cv' or 'mean')

        Returns:
            DataFrame with input features and predicted expression metrics
        """
        logger.info(f"Predicting {target} from sequence features...")

        # Map target names to ensure compatibility
        target_map = {"mean": "mean_expression", "cv": "cv"}
        actual_target = target_map.get(target, target)

        # Check if models are available
        if not self.models:
            logger.error("No trained models available. Run build_expression_prediction_model first.")
            return pd.DataFrame()

        # Check if feature columns are saved
        if not hasattr(self, "feature_cols") or not self.feature_cols:
            logger.error("No feature columns saved. Run build_expression_prediction_model first.")
            return pd.DataFrame()

        # Check if scaler is available
        if self.scaler is None:
            logger.error("No scaler available. Run build_expression_prediction_model first.")
            return pd.DataFrame()

        # Check if we have saved the feature stats for preprocessing
        if not hasattr(self, "feature_stats"):
            logger.error("No feature statistics available from training. Model may not be properly trained.")
            return pd.DataFrame()

        # Copy the input data frame to avoid modifying the original
        result_df = sequence_features.copy()

        # Check for feature compatibility, use only the features we trained on
        missing_features = [f for f in self.feature_cols if f not in result_df.columns]
        if missing_features:
            logger.error(f"Missing required features in prediction data: {missing_features}")
            return pd.DataFrame()

        # Extract only the features used during training, in the same order
        try:
            X_pred = result_df[self.feature_cols].copy()
            logger.info(f"Preparing {len(X_pred)} samples with {len(self.feature_cols)} features for prediction")

            # Apply the EXACT same preprocessing steps used during training
            # 1. Replace inf values with NaN
            X_pred = X_pred.replace([np.inf, -np.inf], np.nan)

            # 2. Apply the same median imputation as during training
            if hasattr(self, "feature_medians"):
                # Use the medians from training
                for col in X_pred.columns:
                    if col in self.feature_medians:
                        X_pred[col] = X_pred[col].fillna(self.feature_medians[col])
                    else:
                        # Fallback to current median if we don't have a stored value
                        logger.warning(f"No saved median for {col}, using current data median")
                        X_pred[col] = X_pred[col].fillna(X_pred[col].median())
            else:
                # Fallback to simple median imputation with warning
                logger.warning("No saved medians from training, using current data medians")
                for col in X_pred.columns:
                    X_pred[col] = X_pred[col].fillna(X_pred[col].median())

            # Verify all NaNs are handled
            if X_pred.isna().any().any():
                logger.warning("NaN values remain after imputation, checking count...")
                nan_counts = X_pred.isna().sum()
                problem_cols = [col for col in X_pred.columns if nan_counts[col] > 0]
                logger.warning(f"Columns with remaining NaNs: {problem_cols} with counts: {nan_counts[problem_cols]}")

                # Last resort: drop rows with remaining NaNs
                clean_mask = ~X_pred.isna().any(axis=1)
                X_pred = X_pred.loc[clean_mask]
                result_df = result_df.loc[clean_mask]

                if len(X_pred) == 0:
                    logger.error("All rows have NaN values after processing, cannot make predictions")
                    return pd.DataFrame()

                logger.warning(f"Dropped rows with NaNs, remaining samples: {len(X_pred)}")

            # 3. Apply the same scaling transformation
            try:
                # Sanity check for scaling - compare feature distributions
                for col in X_pred.columns:
                    pred_std = X_pred[col].std()
                    if pred_std < 1e-6:
                        logger.warning(f"Feature {col} has near-zero variance in prediction data, scaling may cause issues")

                    # Check for distribution shift between training and prediction
                    if col in self.feature_stats:
                        train_mean = self.feature_stats['means'][col]
                        train_std = self.feature_stats['std_devs'][col]
                        pred_mean = X_pred[col].mean()

                        # Calculate z-score of the difference to detect large shifts
                        if train_std > 0:
                            z_diff = abs(train_mean - pred_mean) / train_std
                            if z_diff > 3.0:  # More than 3 standard deviations
                                logger.warning(f"Feature {col} shows significant distribution shift between training and prediction data (z={z_diff:.2f})")

                # Apply the SAME scaler that was fit on training data
                X_pred_scaled = self.scaler.transform(X_pred)

                # Check for extreme values after scaling
                if np.any(np.abs(X_pred_scaled) > 10):
                    logger.warning("Extreme values detected after scaling, may indicate distribution mismatch")

            except Exception as e:
                logger.exception(f"Error during scaling of prediction data: {e}")
                logger.error("Prediction will likely be unreliable due to scaling issues")
                return pd.DataFrame()

            # Make predictions based on model type
            if "ensemble_avg" in self.models:
                # This is an ensemble model - need to average component model predictions
                ensemble_preds = None
                ensemble_weights = []
                successful_models = 0

                for name, model in self.models.items():
                    if name == "ensemble_avg":
                        continue  # Skip ensemble marker

                    try:
                        # Predict with this component model
                        model_pred = model.predict(X_pred_scaled)

                        # Validate predictions
                        if not np.all(np.isfinite(model_pred)):
                            invalid_count = np.sum(~np.isfinite(model_pred))
                            logger.warning(f"Model {name} produced {invalid_count} invalid predictions, skipping")
                            continue

                        # Initialize or add to ensemble predictions
                        if ensemble_preds is None:
                            ensemble_preds = model_pred
                        else:
                            ensemble_preds += model_pred

                        # Add to model count
                        successful_models += 1
                        ensemble_weights.append(1.0)  # Equal weighting

                        # Store individual model predictions
                        result_df[f"{name}_pred"] = model_pred

                    except Exception as e:
                        logger.exception(f"Error during prediction with {name} model: {e}")

                # Check if we have any valid predictions
                if ensemble_preds is None or successful_models == 0:
                    logger.error("No successful predictions from any ensemble component, prediction failed")
                    return pd.DataFrame()

                # Calculate ensemble prediction (average of component models)
                ensemble_preds /= successful_models
                result_df["ensemble_pred"] = ensemble_preds

                # Calculate performance metrics if actual values are available
                if actual_target in result_df.columns:
                    try:
                        actuals = result_df[actual_target].values
                        valid_indices = np.isfinite(actuals) & np.isfinite(ensemble_preds)
                        if np.sum(valid_indices) > 10:
                            rmse = np.sqrt(mean_squared_error(actuals[valid_indices], ensemble_preds[valid_indices]))
                            r2 = r2_score(actuals[valid_indices], ensemble_preds[valid_indices])

                            # Check if R² is extremely negative
                            if r2 < -1.0:
                                # Log prediction vs actual values for diagnostics
                                logger.warning(f"Extremely negative R² detected: {r2:.4f}")
                                logger.warning(f"RMSE: {rmse:.4f}, number of valid samples: {np.sum(valid_indices)}")

                                # Add diagnostics about the data distributions
                                actu_mean = np.mean(actuals[valid_indices])
                                actu_std = np.std(actuals[valid_indices])
                                pred_mean = np.mean(ensemble_preds[valid_indices])
                                pred_std = np.std(ensemble_preds[valid_indices])

                                logger.warning(f"Actual values - mean: {actu_mean:.4f}, std: {actu_std:.4f}")
                                logger.warning(f"Predicted values - mean: {pred_mean:.4f}, std: {pred_std:.4f}")

                                # Calculate the residual sum of squares and total sum of squares
                                rss = np.sum((actuals[valid_indices] - ensemble_preds[valid_indices])**2)
                                tss = np.sum((actuals[valid_indices] - actu_mean)**2)
                                logger.warning(f"RSS: {rss:.4f}, TSS: {tss:.4f}")

                                if pred_std < 1e-6:
                                    logger.warning("Predictions have near-zero variance, indicating a failed model")

                                if abs(pred_mean - actu_mean) / actu_std > 10:
                                    logger.warning("Large shift between prediction and actual means detected")

                            else:
                                logger.info(f"Ensemble model performance - RMSE: {rmse:.4f}, R²: {r2:.4f}")
                        else:
                            logger.warning(f"Too few valid samples ({np.sum(valid_indices)}) to calculate performance metrics")
                    except Exception as e:
                        logger.exception(f"Error calculating prediction performance: {e}")
            else:
                # Single model approach
                model_name = list(self.models.keys())[0]
                model = self.models[model_name]

                try:
                    # Make predictions
                    predictions = model.predict(X_pred_scaled)

                    # Store predictions
                    result_df[f"{model_name}_pred"] = predictions

                    # Calculate performance metrics if actual values are available
                    if actual_target in result_df.columns:
                        try:
                            actuals = result_df[actual_target].values
                            valid_indices = np.isfinite(actuals) & np.isfinite(predictions)
                            if np.sum(valid_indices) > 10:
                                rmse = np.sqrt(mean_squared_error(actuals[valid_indices], predictions[valid_indices]))
                                r2 = r2_score(actuals[valid_indices], predictions[valid_indices])

                                if r2 < -1.0:
                                    logger.warning(f"Extremely negative R² detected: {r2:.4f}")
                                    logger.warning(f"RMSE: {rmse:.4f}, samples: {np.sum(valid_indices)}")
                                else:
                                    logger.info(f"Model performance - RMSE: {rmse:.4f}, R²: {r2:.4f}")
                            else:
                                logger.warning(f"Too few valid samples ({np.sum(valid_indices)}) to calculate performance metrics")
                        except Exception as e:
                            logger.exception(f"Error calculating prediction performance: {e}")
                except Exception as e:
                    logger.exception(f"Error making predictions with {model_name} model: {e}")
                    return pd.DataFrame()

            # Add prediction error metrics if actuals are available
            if actual_target in result_df.columns:
                for col in result_df.columns:
                    if col.endswith("_pred"):
                        result_df[f"{col}_error"] = result_df[actual_target] - result_df[col]
                        result_df[f"{col}_abs_error"] = abs(result_df[actual_target] - result_df[col])

            return result_df

        except Exception as e:
            logger.exception(f"Critical error during prediction process: {e}")
            return pd.DataFrame()

    def analyze_feature_importance(
        self, models: dict[str, Any], feature_names: list[str]
    ) -> pd.DataFrame:
        """Calculate feature importance metrics.

        Args:
            models: Dictionary of trained models
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance rankings and scores
        """
        logger.info("Analyzing feature importance...")

        if not models or not feature_names:
            logger.error("Missing models or feature names")
            return pd.DataFrame()

        # Initialize results
        importance_data = {}

        # Calculate importance for each model type
        for model_name, model in models.items():
            if model_name == "ensemble_avg":
                # Skip ensemble model marker
                continue

            # Extract feature importance
            try:
                if hasattr(model, "coef_") and model_name in ["ridge", "lasso"]:
                    # Linear models: coefficient magnitude
                    importance = np.abs(model.coef_)
                    if len(importance) == len(feature_names):
                        importance_data[model_name] = {
                            name: float(score) for name, score in zip(feature_names, importance, strict=False)
                        }
                    else:
                        logger.warning(f"Feature names length ({len(feature_names)}) doesn't match coefficients length ({len(importance)})")

                elif model_name in ("random_forest", "gbr") or hasattr(model, "feature_importances_"):
                    # Tree-based models: feature importance
                    if hasattr(model, "feature_importances_"):
                        importance = model.feature_importances_
                        if len(importance) == len(feature_names):
                            importance_data[model_name] = {
                                name: float(score) for name, score in zip(feature_names, importance, strict=False)
                            }
                        else:
                            logger.warning(f"Feature names length ({len(feature_names)}) doesn't match feature importances length ({len(importance)})")
                    else:
                        logger.warning(f"Model {model_name} doesn't have feature_importances_ attribute")

                elif isinstance(model, dict) and "importances_mean" in model:
                    # Handle permutation importance result returned as dictionary
                    importance = model["importances_mean"]
                    if len(importance) == len(feature_names):
                        importance_data[model_name] = {
                            name: float(score) for name, score in zip(feature_names, importance, strict=False)
                        }
                    else:
                        logger.warning(f"Feature names length ({len(feature_names)}) doesn't match permutation importance length ({len(importance)})")

                else:
                    logger.warning(f"No suitable importance extraction method for model type {model_name}")

            except Exception as e:
                logger.warning(f"Error extracting importance from {model_name}: {e}")

        if not importance_data:
            logger.warning("No feature importance could be extracted from models")
            return pd.DataFrame()

        # Combine importance scores across models
        combined_importance = {}
        for feature in feature_names:
            scores = [
                data[feature] for data in importance_data.values() if feature in data
            ]
            if scores:
                combined_importance[feature] = float(np.mean(scores))

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"feature": list(combined_importance.keys()), "importance_score": list(combined_importance.values())}
        ).sort_values("importance_score", ascending=False)

        # Add normalized importance
        max_score = importance_df["importance_score"].max()
        if max_score > 0:
            importance_df["normalized_importance"] = importance_df["importance_score"] / max_score

        # Store feature importances in the instance
        self.feature_importances = combined_importance

        logger.info(f"Analyzed importance for {len(importance_df)} features across {len(importance_data)} models")
        return importance_df

    def simulate_sequence_optimization(
        self,
        model: Any,
        feature_ranges: dict[str, tuple[float, float]],
        optimization_target: str = "cv_minimization",
    ) -> pd.DataFrame:
        """Use trained models to predict optimal sequence features.

        Args:
            model: Trained prediction model
            feature_ranges: Dictionary of feature names to (min, max) ranges
            optimization_target: Target for optimization ('cv_minimization' or 'mean_maximization')

        Returns:
            DataFrame with optimization results
        """
        logger.info(f"Simulating sequence optimization for {optimization_target}...")

        if model is None:
            logger.error("No model provided")
            return pd.DataFrame()

        if not feature_ranges:
            logger.error("No feature ranges provided")
            return pd.DataFrame()

        # Check if feature importance is available
        if not self.feature_importances:
            logger.warning("No feature importance data available")

        # Create optimization results structure
        optimization_results: list[dict[str, Any]] = []

        # Generate parameter grid
        n_points = 10  # Number of points to sample per feature
        param_grid: dict[str, np.ndarray] = {}

        for feature, (min_val, max_val) in feature_ranges.items():
            param_grid[feature] = np.linspace(min_val, max_val, n_points)

        # Get baseline feature values (median)
        baseline_values = {feature: np.median(param_grid[feature]) for feature in param_grid}

        # Sort features by importance (if available)
        if self.feature_importances:
            sorted_features = sorted(
                [f for f in feature_ranges if f in self.feature_importances],
                key=lambda x: self.feature_importances.get(x, 0),
                reverse=True,
            )
        else:
            sorted_features = list(feature_ranges.keys())

        # Test each feature individually, holding others at baseline
        for feature in sorted_features:
            feature_values = param_grid[feature]

            # Create test data
            X_test = []
            for value in feature_values:
                # Create a data point with baseline values
                data_point = baseline_values.copy()
                # Modify the current feature
                data_point[feature] = value
                # Convert to list in the correct order
                X_test.append([data_point[f] for f in sorted_features])

            # Convert to numpy array
            X_test_np = np.array(X_test)

            try:
                # Predict target values
                y_pred = model.predict(X_test_np)

                # Find optimal value based on optimization target
                if optimization_target == "cv_minimization":
                    # Lower CV is better (more stable)
                    optimal_idx = np.argmin(y_pred)
                    optimal_direction = "minimize"
                else:
                    # Higher mean expression is better
                    optimal_idx = np.argmax(y_pred)
                    optimal_direction = "maximize"

                # Record results
                optimal_value = feature_values[optimal_idx]
                optimal_prediction = y_pred[optimal_idx]

                # Calculate improvement over baseline
                baseline_idx = np.argmin(np.abs(feature_values - baseline_values[feature]))
                baseline_prediction = y_pred[baseline_idx]

                if optimization_target == "cv_minimization":
                    improvement = (baseline_prediction - optimal_prediction) / baseline_prediction
                else:
                    improvement = (optimal_prediction - baseline_prediction) / baseline_prediction

                optimization_results.append(
                    {
                        "feature": feature,
                        "importance": self.feature_importances.get(feature, 0),
                        "optimal_value": optimal_value,
                        "baseline_value": baseline_values[feature],
                        "direction": optimal_direction,
                        "improvement": improvement,
                        "baseline_prediction": baseline_prediction,
                        "optimal_prediction": optimal_prediction,
                    }
                )

            except Exception as e:
                logger.warning(f"Error optimizing feature {feature}: {e}")

        if not optimization_results:
            logger.warning("No optimization results could be calculated")
            return pd.DataFrame()

        # Create results DataFrame
        result_df = pd.DataFrame(optimization_results)

        # Sort by improvement potential
        result_df = result_df.sort_values("improvement", ascending=False)

        logger.info(
            f"Optimized {len(result_df)} features, max improvement: {result_df['improvement'].max():.2%}"
        )
        self.results["optimization"] = result_df

        return result_df


class SequenceModelingVisualization:
    """Creates visualizations for sequence-based expression prediction."""

    def __init__(self):
        """Initialize the visualization class."""
        self.figures = {}

    def plot_feature_importance(
        self, importance_df: pd.DataFrame, figsize: tuple[int, int] = (10, 8)
    ) -> Figure | None:
        """Generate a bar chart of feature importance scores.

        Args:
            importance_df: DataFrame with feature importance statistics
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if importance_df is None or importance_df.empty:
            logger.error("Empty importance DataFrame provided")
            return None

        # Check required columns and prepare the data
        if "feature" not in importance_df.columns:
            logger.error("Missing required column: 'feature'")
            return None

        # Handle different column naming conventions and ensure normalized_importance exists
        if "normalized_importance" in importance_df.columns:
            importance_col = "normalized_importance"
        elif "importance_score" in importance_df.columns:
            # Calculate normalized importance if it doesn't exist
            importance_col = "importance_score"
            max_score = importance_df[importance_col].max()
            if max_score > 0:
                importance_df["normalized_importance"] = importance_df[importance_col] / max_score
                importance_col = "normalized_importance"
        elif "importance" in importance_df.columns:
            # Try the simple 'importance' column name
            importance_col = "importance"
            max_score = importance_df[importance_col].max()
            if max_score > 0:
                importance_df["normalized_importance"] = importance_df[importance_col] / max_score
                importance_col = "normalized_importance"
        else:
            logger.error("No importance column found in DataFrame")
            return None

        try:
            # Limit to top 15 features for better readability
            if len(importance_df) > 15:
                plot_data = importance_df.sort_values(importance_col, ascending=False).head(15)
                plot_data = plot_data.sort_values(importance_col, ascending=True)  # Reverse for horizontal bars
            else:
                # Sort by importance (ascending for horizontal bars)
                plot_data = importance_df.sort_values(importance_col, ascending=True)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot horizontal bars
            bars = ax.barh(
                plot_data["feature"], plot_data[importance_col], color="#1f77b4", alpha=0.8
            )

            # Add values to bars
            for i, v in enumerate(plot_data[importance_col]):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center")

            # Add labels
            ax.set_xlabel("Relative Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance")

            # Add grid
            ax.grid(alpha=0.3, axis="x")

            # Adjust layout
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating feature importance plot: {e}")
            return None

    def plot_prediction_performance(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
        target_name: str = "cv",
        figsize: tuple[int, int] = (12, 8),
    ) -> Figure | None:
        """Generate a scatter plot of predicted vs. actual values.

        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: Target values
            feature_names: Names of features (optional)
            target_name: Name of target variable (e.g. 'cv' or 'mean')
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if model is None:
            logger.error("Model is None, cannot generate prediction performance plot")
            return None

        try:
            # Convert pandas objects to numpy arrays if needed
            X_np = X.values if isinstance(X, pd.DataFrame) else X
            y_np = y.values if isinstance(y, pd.Series) else y

            # Create masks for valid data points
            valid_X = np.isfinite(X_np).all(axis=1)
            valid_y = np.isfinite(y_np)
            valid_mask = valid_X & valid_y

            if valid_mask.sum() < 10:
                logger.error("Insufficient valid data points for prediction visualization")
                return None

            # Apply mask to get only valid data
            X_valid = X_np[valid_mask]
            y_valid = y_np[valid_mask]

            # IMPORTANT: Ensure data is preprocessed the same way as during training
            # Check if we have a scaler available in the modeling object
            try:
                # Data preprocessing similar to training - this is crucial
                # 1. Replace infinities with NaN
                X_valid_df = pd.DataFrame(X_valid, columns=feature_names)
                X_valid_df = X_valid_df.replace([np.inf, -np.inf], np.nan)

                # 2. Fill NaN values with column medians
                for col in X_valid_df.columns:
                    X_valid_df[col] = X_valid_df[col].fillna(X_valid_df[col].median())

                # 3. Apply scaling if a scaler is available
                if hasattr(self, "scaler") and self.scaler is not None:
                    X_valid_scaled = self.scaler.transform(X_valid_df)
                else:
                    # Fall back to standard scaling if the original scaler isn't available
                    logger.warning("No scaler found. Applying StandardScaler for visualization.")
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_valid_scaled = scaler.fit_transform(X_valid_df)

                # Check if we have an ensemble model
                if isinstance(model, dict) and "ensemble_avg" in model:
                    # We're dealing with ensemble model - need to average predictions
                    ensemble_preds = np.zeros_like(y_valid)
                    model_count = 0

                    for name, m in model.items():
                        if name != "ensemble_avg" and m is not None:
                            try:
                                ensemble_preds += m.predict(X_valid_scaled)
                                model_count += 1
                            except Exception as e:
                                logger.warning(f"Error predicting with {name} model: {e}")

                    if model_count > 0:
                        y_pred = ensemble_preds / model_count
                    else:
                        raise ValueError("No valid models in ensemble")
                else:
                    # Single model prediction
                    y_pred = model.predict(X_valid_scaled)
            except Exception as e:
                logger.warning(f"Error during data preprocessing for prediction: {e}")
                # Fallback to direct prediction, but log the issue
                logger.warning("Falling back to direct prediction without preprocessing")
                if isinstance(model, dict) and "ensemble_avg" in model:
                    # Handle ensemble case
                    ensemble_preds = np.zeros_like(y_valid)
                    model_count = 0

                    for name, m in model.items():
                        if name != "ensemble_avg" and m is not None:
                            try:
                                ensemble_preds += m.predict(X_valid)
                                model_count += 1
                            except Exception as e:
                                logger.warning(f"Error predicting with {name} model: {e}")

                    if model_count > 0:
                        y_pred = ensemble_preds / model_count
                    else:
                        raise ValueError("No valid models in ensemble")
                else:
                    y_pred = model.predict(X_valid)

            # Check predictions for validity
            valid_pred = np.isfinite(y_pred)
            if not np.all(valid_pred):
                logger.warning(f"Found {(~valid_pred).sum()} invalid predictions, filtering...")
                X_valid = X_valid[valid_pred]
                y_valid = y_valid[valid_pred]
                y_pred = y_pred[valid_pred]

            if len(y_valid) < 10:
                logger.error("Insufficient valid data points after prediction filtering")
                return None

            # Calculate metrics with proper validation
            try:
                # Calculate R²
                if np.var(y_valid) > 1e-10 and np.var(y_pred) > 1e-10:
                    r2 = r2_score(y_valid, y_pred)
                    # Cap extremely negative R² at -1.0 for better display
                    if r2 < -1.0:
                        logger.warning(f"Extremely negative R² detected: {r2:.4f}, capping at -1.0")
                        r2 = -1.0
                    r2_display = f"{r2:.3f}"
                else:
                    logger.warning("Near-zero variance in data, can't calculate meaningful R²")
                    r2 = 0.0
                    r2_display = "0.000 (insufficient variance)"

                rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            except Exception as e:
                logger.exception(f"Error calculating performance metrics: {e}")
                r2 = 0.0
                r2_display = "N/A"
                rmse = np.nan

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create scatter plot with color-coding by error magnitude
            scatter = ax.scatter(
                y_valid,
                y_pred,
                alpha=0.6,
                s=30,
                c=np.abs(y_valid - y_pred),
                cmap="viridis_r",  # Reversed colormap so darker = smaller error
            )

            # Calculate appropriate axis limits
            all_vals = np.concatenate([y_valid, y_pred])
            min_val = np.min(all_vals)
            max_val = np.max(all_vals)

            # Add a margin to the plot
            range_val = max_val - min_val
            margin = range_val * 0.1

            # If range is too small, use a fixed margin
            if range_val < 1e-10:
                margin = 0.1 * (np.abs(min_val) + 1e-10)

            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)

            # Add perfect prediction line
            ax.plot(
                [min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                "k--",
                alpha=0.5,
                label="Perfect prediction"
            )

            # Add metrics in text box
            textstr = f"$R^2 = {r2_display}$\n$RMSE = {rmse:.3f}$\n$n = {len(y_valid)}$"
            props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
            ax.text(
                0.05,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

            # Set labels based on target name
            target_map = {"cv": "Expression Stability (CV)", "mean": "Expression Level", "mean_expression": "Expression Level"}
            target_label = target_map.get(target_name, target_name)

            ax.set_xlabel(f"Actual {target_label}")
            ax.set_ylabel(f"Predicted {target_label}")
            ax.set_title(f"Prediction Performance (R² = {r2:.3f})")

            # Add color bar for error magnitude
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Prediction Error (absolute)")

            # Add grid
            ax.grid(alpha=0.3)

            # Add legend
            ax.legend(loc="lower right")

            # Tight layout
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating prediction performance plot: {e}")
            return None

    def plot_shap_summary(
        self, model: Any, X: pd.DataFrame, figsize: tuple[int, int] = (12, 8)
    ) -> Figure | None:
        """Generate a SHAP value summary plot.

        Args:
            model: Trained prediction model
            X: Feature DataFrame
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if model is None or X is None:
            logger.error("Missing required data for SHAP plot")
            return None

        try:
            import shap

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create a background dataset for SHAP values
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            # Plot SHAP summary
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)

            # Return the current figure
            return plt.gcf()

        except ImportError:
            logger.warning("SHAP package not installed. Cannot create SHAP summary plot.")
            return None
        except Exception as e:
            logger.exception(f"Error creating SHAP summary plot: {e}")
            return None

    def plot_optimization_contour(
        self,
        model: Any,
        feature_ranges: dict[str, tuple[float, float]],
        feature_pair: tuple[str, str],
        optimization_target: str = "cv_minimization",
        figsize: tuple[int, int] = (10, 8),
    ) -> Figure | None:
        """Generate a contour plot showing predicted expression.

        Args:
            model: Trained prediction model
            feature_ranges: Dictionary of feature names to (min, max) ranges
            feature_pair: Pair of features to plot
            optimization_target: Target for optimization ('cv_minimization' or 'mean_maximization')
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if model is None:
            logger.error("No model provided for optimization contour")
            return None

        if not feature_ranges:
            logger.error("No feature ranges provided")
            return None

        if len(feature_pair) != 2:
            logger.error("Feature pair must contain exactly two features")
            return None

        feature1, feature2 = feature_pair

        if feature1 not in feature_ranges or feature2 not in feature_ranges:
            logger.error(f"Features {feature1} and/or {feature2} not in feature ranges")
            return None

        try:
            # Create parameter grid
            x_range = np.linspace(*feature_ranges[feature1], 50)
            y_range = np.linspace(*feature_ranges[feature2], 50)
            X1, X2 = np.meshgrid(x_range, y_range)

            # Get baseline feature values (median)
            baseline_values = {
                feature: np.median(np.linspace(*feature_ranges[feature], 10))
                for feature in feature_ranges
            }

            # Create sorted list of all features
            all_features = sorted(feature_ranges.keys())

            # Create test data
            grid_points = []
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    # Create a data point with baseline values
                    data_point = baseline_values.copy()
                    # Modify the current features
                    data_point[feature1] = X1[i, j]
                    data_point[feature2] = X2[i, j]
                    # Convert to list in the correct order
                    grid_points.append([data_point[f] for f in all_features])

            # Convert to numpy array
            grid_points_np = np.array(grid_points)

            # Predict target values
            Z = model.predict(grid_points_np)
            Z = Z.reshape(X1.shape)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create contour plot
            if optimization_target == "cv_minimization":
                # Lower CV is better (more stable)
                contour = ax.contourf(X1, X2, Z, 50, cmap="viridis_r")
            else:
                # Higher mean expression is better
                contour = ax.contourf(X1, X2, Z, 50, cmap="viridis")

            # Add colorbar
            cbar = fig.colorbar(contour, ax=ax)
            target_label = (
                "Expression Stability (CV)"
                if optimization_target == "cv_minimization"
                else "Expression Level"
            )
            cbar.set_label(target_label)

            # Add optimal point
            if optimization_target == "cv_minimization":
                optimal_idx = np.unravel_index(Z.argmin(), Z.shape)
                optimal_value = Z[optimal_idx]
                optimal_text = f"Min CV: {optimal_value:.3f}"
            else:
                optimal_idx = np.unravel_index(Z.argmax(), Z.shape)
                optimal_value = Z[optimal_idx]
                optimal_text = f"Max Expression: {optimal_value:.3f}"

            optimal_x = X1[optimal_idx]
            optimal_y = X2[optimal_idx]

            ax.plot(optimal_x, optimal_y, "ro", markersize=8)
            ax.annotate(
                optimal_text,
                xy=(float(optimal_x), float(optimal_y)),
                xytext=(10, 10),
                textcoords="offset points",
                color="red",
                fontweight="bold",
            )

            # Add baseline point
            baseline_x = baseline_values[feature1]
            baseline_y = baseline_values[feature2]
            ax.plot(baseline_x, baseline_y, "ko", markersize=8)

            # Set labels
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.set_title(f"Optimization Landscape: {feature1} vs {feature2}")

            # Adjust layout
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating optimization contour plot: {e}")
            return None
