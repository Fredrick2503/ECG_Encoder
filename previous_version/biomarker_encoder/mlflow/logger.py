"""
=========================================================
MLflow Logger

Logs:
1. Experiment
2. Parameters
3. Metrics
4. Models
5. Artifacts

Author : ECG Intelligence System
=========================================================
"""

from pathlib import Path

import mlflow
import mlflow.sklearn


class MLFlowLogger:

    def __init__(
        self,
        experiment_name: str = "Biomarker Encoder"
    ):

        mlflow.set_experiment(experiment_name)

    # --------------------------------------------------
    # Start Run
    # --------------------------------------------------

    def start_run(
        self,
        run_name: str
    ):

        self.run = mlflow.start_run(
            run_name=run_name
        )

        return self.run

    # --------------------------------------------------
    # End Run
    # --------------------------------------------------

    def end_run(self):

        mlflow.end_run()

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------

    def log_params(
        self,
        params: dict
    ):

        if params is None:
            return

        mlflow.log_params(params)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def log_metrics(
        self,
        metrics
    ):
        """
        metrics:
            dict
            OR
            pandas.DataFrame (single row)
        """

        if metrics is None:
            return

        if hasattr(metrics, "to_dict"):

            metrics = metrics.iloc[0].to_dict()

        clean_metrics = {}

        for key, value in metrics.items():

            try:
                clean_metrics[key] = float(value)

            except Exception:
                pass

        mlflow.log_metrics(clean_metrics)

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    def log_models(
        self,
        encoder
    ):
        """
        Logs every internal model.
        """

        models = encoder.get_models()

        for name, model in models.items():

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=name,
            )

    # --------------------------------------------------
    # Artifacts
    # --------------------------------------------------

    def log_artifact(
        self,
        filepath
    ):

        filepath = Path(filepath)

        if filepath.exists():

            mlflow.log_artifact(
                str(filepath)
            )

    # --------------------------------------------------
    # Directory Artifacts
    # --------------------------------------------------

    def log_artifacts(
        self,
        directory
    ):

        directory = Path(directory)

        if directory.exists():

            mlflow.log_artifacts(
                str(directory)
            )

    # --------------------------------------------------
    # Tags
    # --------------------------------------------------

    def set_tags(
        self,
        tags: dict
    ):

        mlflow.set_tags(tags)

    # --------------------------------------------------
    # Complete Logging
    # --------------------------------------------------

    def log_run(
        self,
        run_name,
        encoder,
        parameters=None,
        metrics=None,
        artifacts=None,
        tags=None,
    ):

        self.start_run(run_name)

        if parameters is not None:

            self.log_params(parameters)

        if metrics is not None:

            self.log_metrics(metrics)

        if encoder is not None:

            self.log_models(encoder)

        if artifacts is not None:

            if isinstance(artifacts, list):

                for artifact in artifacts:

                    self.log_artifact(
                        artifact
                    )

            else:

                self.log_artifact(
                    artifacts
                )

        if tags is not None:

            self.set_tags(tags)

        self.end_run()