import wandb

class Logger:
    def __init__(self, project, run_name, config):
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name,
            config=dict(config)
        )

    def log_metrics(self, metrics, step):
        self._run.log(dict(metrics), step=step) # serve a inviare le metriche a WandB, associandole allo step (epoca)

    def log_config(self, config):
        self._run.config.update(dict(config), allow_val_change=True)  # serve ad aggiornare o integrare la config già registrata

    def finish(self):
        self._run.finish() # serve a chiudere la run, forzare l'upload e liberare le risorse


def build_logger(logging_config, config) -> Logger:
    return Logger(
        project=logging_config['project'],
        run_name=logging_config['run_name'],
        config= config
    )