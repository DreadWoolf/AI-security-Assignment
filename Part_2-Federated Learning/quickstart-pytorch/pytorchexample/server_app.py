"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    FedAvg,
    FedMedian,
    FedTrimmedAvg,
    FedAvgM,
    FedYogi
)

import json
from pathlib import Path
from pytorchexample.task import Net, load_centralized_dataset, test_with_metrics

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    #####################################################
    #       Select aggregation strategy                 #
    # (makes it possible to resuse same server code)    #
    #####################################################
    strategy_name = context.run_config["strategy-name"]

    strategies = {
        "fedavg": FedAvg,
        "fedmedian": FedMedian,
        "fedtrimmedavg": FedTrimmedAvg,
        "fedavgm": FedAvgM,
        "fedyogi": FedYogi
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    params = {"fraction_evaluate": fraction_evaluate}

    # Server momentum of 0.9, hur starkt tidigare serveruppdateringar påverkar nästa
    if strategy_name == "fedavgm":
        params["server_momentum"] = 0.9

    # Tar bort högsta och lägsta värdet för varje parameter bland klienterna.
    elif strategy_name == "fedtrimmedavg":
        params["beta"] = 0.2

    # Lokal learning rate-parameter i serverstrategin.
    elif strategy_name == "fedyogi":
        params["eta_l"] = lr

    # Create the strategy
    strategy = strategies[strategy_name](**params)

    print(f"Selected strategy: {strategy_name}")

    #####################################################
    #####################################################
    #####################################################
    



    # Start strategy, run FedAvg for `num_rounds`
    # # Spara centrala mätvärden per runda till en JSON-fil
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )




    # Find Assignment directory from current working directory
    cwd = Path.cwd().resolve()

    assignment_dir = next(
        (p for p in [cwd, *cwd.parents] if p.name == "Assignment"),
        None
    )

    if assignment_dir is None:
        raise RuntimeError(f"Could not find Assignment directory from {cwd}")


    # Create directories
    results_dir = assignment_dir / "Part_2-Federated Learning" / "resultat"
    models_dir = assignment_dir / "Part_2-Federated Learning" / "modeller"

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = (
        f"{context.run_config['run-name']}"
        f"_attack{context.run_config['num-attacked-clients']}"
        f"_rounds{num_rounds}"
    )
    metrics = {rnd: dict(rec) for rnd, rec in result.evaluate_metrics_serverapp.items()}
    with open(results_dir / f"{run_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Resultat sparade i {results_dir / f'{run_name}.json'}")


    # Save final model to disk
    if context.run_config["save-model"]:
        model_path = models_dir / f"{run_name}.pt"

        torch.save(
            result.arrays.to_torch_state_dict(),
            model_path
        )

        print(f"Model saved to {model_path}")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Utvärderar den globala modellen på hela testdatan, inklusive Kappa och F1."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    loss, acc, kappa, f1 = test_with_metrics(model, test_dataloader, device)

    return MetricRecord({"accuracy": acc, "loss": loss, "kappa": kappa, "f1": f1})