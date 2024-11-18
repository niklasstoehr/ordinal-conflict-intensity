
from g1data import dataloading
from g2model.models import model
from g2model import trainer


def compute_random_seed(x, y):
    return x * 100 + y


def run_loop(model_type, df, kwargs = {}, loop_kwargs = {}, print_params = False):
    """
    setting n_c, n_runs and random_seed
    """

    for n_c in loop_kwargs["n_c"]:
        for n_run in range(0, loop_kwargs["n_runs"]):
            print(f"n_run: {n_run}")

            kwargs["n_c"] = n_c
            kwargs["random_seed"] = compute_random_seed(n_c, n_run)

            m = model.get_model(model_type=model_type, n_c=kwargs["n_c"])
            trainer.single_run(m, df, kwargs, print_params)
            print("\n")


if __name__ == "__main__":

    df = dataloading.load_navco(csv_name="navco_full.csv", G_cat=-1, Q_thresh=1.0, Q_log=True, T_rank=0)

    loop_kwargs = {"n_runs": 3, "n_c": [5]}
    kwargs = {"n_samples": 500, "n_warmup": 50, "init_n": 200, "nuts_tree": 8, "accept_prob": 0.8, "post_samples": 200}

    run_loop(model_type = "gqta", df = df, kwargs = kwargs, loop_kwargs = loop_kwargs)