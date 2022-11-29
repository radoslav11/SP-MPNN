import torch
from torch_geometric.loader import DataLoader
import numpy as np

# This file is essentially tailor-made for QM9
TASKS = [
    "mu",
    "alpha",
    "HOMO",
    "LUMO",
    "gap",
    "R2",
    "ZPVE",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "Omega",
]
# QM9 y values were normalized, so we need to de-normalize.
CHEMICAL_ACC_NORMALISING_FACTORS = [
    0.066513725,
    0.012235489,
    0.071939046,
    0.033730778,
    0.033486113,
    0.004278493,
    0.001330901,
    0.004165489,
    0.004128926,
    0.00409976,
    0.004527465,
    0.012292586,
    0.037467458,
]
# Dropout is not used. MSE training, MAE for valid and test, with the above de-normalizing factors.


def train(model, loader, optimizer, loss_fun, device="cpu", y_idx=0):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fun(model(data), data.y[:, y_idx : y_idx + 1]).to(device)  #
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    return loss_all / len(loader.dataset)


def val(model, loader, loss_fun, device="cpu", y_idx=0):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += loss_fun(model(data), data.y[:, y_idx : y_idx + 1]).item()

    return loss_all / len(loader.dataset)


def test(model, loader, device="cpu", y_idx=0):
    model.eval()
    total_err = 0

    for data in loader:
        data = data.to(device)
        pred = torch.sum(torch.abs(model(data) - data.y[:, y_idx : y_idx + 1])).item()
        total_err += pred

    return total_err / (
        len(loader.dataset) * CHEMICAL_ACC_NORMALISING_FACTORS[y_idx]
    )  # Introduce norm factors


# Treat every target separately. So you're effectively training 13 times.


def run_model_gr(
    model,
    dataset_tr,
    dataset_val,
    dataset_tst,
    batch_size=32,
    lr=0.0001,
    epochs=300,
    neptune_client=None,
    device="cpu",
    nb_reruns=5,
    specific_task=-1,
):
    loss_fun = torch.nn.MSELoss(
        reduction="sum"
    )  # Mean-Squared Loss is used for regression

    print("---------------- Training on provided split (QM9) ----------------")
    for y_idx, targ in enumerate(TASKS):  # Solve each one at a time...
        if 0 <= specific_task != y_idx:
            continue
        print("----------------- Predicting " + str(targ) + " -----------------")
        all_test_mae = np.zeros(nb_reruns,)
        all_val_mae = np.zeros(nb_reruns,)

        for rerun in range(nb_reruns):  # 5 Reruns for GR
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Made static

            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset_tst, batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(
                dataset_tr, batch_size=batch_size, shuffle=True
            )  # Shuffling is good here

            rerun_str = "QM9/" + str(targ) + "/rerun_" + str(rerun)
            print(
                "---------------- "
                + str(targ)
                + ": Re-run {} ----------------".format(rerun)
            )

            best_val_mse = 100000
            test_mae = 100
            best_val_mae = 100000

            for epoch in range(1, epochs + 1):
                # lr = scheduler.optimizer.param_groups[0]['lr']  # Same as GC
                train_mse = train(
                    model, train_loader, optimizer, loss_fun, device=device, y_idx=y_idx
                )
                val_mse = val(model, val_loader, loss_fun, device=device, y_idx=y_idx)
                # scheduler.step(val_mse_sum)
                if best_val_mse >= val_mse:  # Improvement in validation loss
                    test_mae = test(model, test_loader, device=device, y_idx=y_idx)
                    best_val_mae = test(model, val_loader, device=device, y_idx=y_idx)
                    best_val_mse = val_mse

                if neptune_client is not None:
                    neptune_client[rerun_str + "/params/lr"].log(lr)
                    neptune_client[rerun_str + "/train/loss"].log(train_mse)
                    train_mae = test(model, train_loader, device=device, y_idx=y_idx)
                    neptune_client[rerun_str + "/train/MAE"].log(train_mae)
                    neptune_client[rerun_str + "/validation/loss"].log(val_mse)

                    val_mae = test(model, val_loader, device=device, y_idx=y_idx)
                    neptune_client[rerun_str + "/validation/MAE"].log(val_mae)
                    neptune_client[rerun_str + "/test/MAE"].log(test_mae)

                    model.log_hop_weights(neptune_client, rerun_str)

                print(
                    "Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, "
                    "Val Loss: {:.7f}, Test MAE: {:.7f}".format(
                        epoch, lr, train_mse, val_mse, test_mae
                    )
                )

            all_test_mae[rerun] = test_mae
            all_val_mae[rerun] = best_val_mae

        avg_test_mae = all_test_mae.mean()
        avg_val_mae = all_val_mae.mean()

        std_test_mae = np.std(all_test_mae)
        std_val_mae = np.std(all_val_mae)
        # No need for averaging. This is 1 split anyway.

        if neptune_client is not None:
            neptune_client["QM9/" + str(targ) + "/validation/MAE"].log(avg_val_mae)
            neptune_client["QM9/" + str(targ) + "/test/MAE"].log(avg_test_mae)
            neptune_client["QM9/" + str(targ) + "/test/MAE_std"].log(std_test_mae)
            neptune_client["QM9/" + str(targ) + "/validation/MAE_std"].log(std_val_mae)

            torch.save(model, "../model.pt")
            neptune_client["QM9/" + str(targ) + "/model"].upload("model.pt")

        print("---------------- Final Result ----------------")
        print("Test -- Mean: " + str(avg_test_mae) + ", Std: " + str(std_test_mae))
        print("Validation -- Mean: " + str(avg_val_mae) + ", Std: " + str(std_val_mae))
