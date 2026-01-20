import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from gene_conservation_network import io
    return io, np, sns


@app.cell
def _(io):
    dir(io)
    return


@app.cell
def _(io):
    extracts = io.list_available_coxpresdb_datasets("data/01_raw/coxpresdb_extracts")
    extracts
    return


@app.cell
def _(io):
    data = io.load_coxpresdb_coexpression(
        data_dir="data/01_raw/coxpresdb_extracts",
        species="Sce",
        modality="microarray"
    )
    data
    return (data,)


@app.cell
def _(data, np, sns):
    sns.histplot(data["association"], bins=np.arange(-7, 10, 0.1))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
