import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from logger import build_logger
from create_dls import create_dataloaders
from train_and_evaluate import MyTrainerEvaluator
import torch.nn as nn

@hydra.main(version_base=None, config_path='config', config_name='train_file')
def main(config): # config è la configurazione finale di Hydra
    print(OmegaConf.to_yaml(config)) # stampa la configurazione
    torch.manual_seed(config.seed)
    device = config.device if torch.cuda.is_available() else 'cpu'

    full_config = OmegaConf.to_container(config, resolve=True) # to_container converte la configurazione in un dict Python e resolve risolve le interpolazioni
    logger = build_logger(config.logging, full_config)

    model = hydra.utils.instantiate(config.model)
    model = model.to(device)

    optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())

    loss_function = nn.CrossEntropyLoss()

    train_dl, val_dl, test_dl = create_dataloaders(training_set_perc=config.training_set_perc, seed=config.seed, batch_size=config.batch_size, num_workers=8)

    trainer_evaluator = MyTrainerEvaluator(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        logger=logger,
        device=device,
        epochs=config.epochs,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl
    )

    trainer_evaluator.train_and_evaluate()
    logger.finish()


if __name__ == '__main__':
    main()