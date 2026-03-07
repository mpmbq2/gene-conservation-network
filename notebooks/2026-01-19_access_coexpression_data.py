import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import duckdb
    import matplotlib.pyplot as plt
    import seaborn as sns

    from gene_conservation_network import io
    return duckdb, pd, pl, sns


@app.cell
def _(duckdb):
    zebra = duckdb.sql(
        """
        SELECT *
        FROM './data/02_transformed/coxpresdb/Dre-m.v21-06.G10112-S1321.combat_pca.subagging.ls.d/*.parquet'
        WHERE association > 4
        """
    ).pl()

    zebra
    return


@app.cell
def _(duckdb):
    fly = duckdb.sql(
        """
        SELECT *
        FROM './data/02_transformed/coxpresdb/Dme-m.v21-06.G12626-S3401.combat_pca.subagging.ls.d/*.parquet'
        WHERE association > 4
        """
    ).pl()

    fly
    return (fly,)


@app.cell
def _(pd):
    dre_dm_ldo = pd.read_csv(
        "data/01_raw/wormhole_extracts/dmdr-WORMHOLE-orthologs.txt",
        delimiter="\t"
    )

    dre_dm_ldo
    return (dre_dm_ldo,)


@app.cell
def _(dre_dm_ldo):
    dre_dm_ldo["Votes"].hist()
    return


@app.cell
def _(dre_dm_ldo):
    dre_dm_ldo["WORMHOLE.Score"].hist()
    return


@app.cell
def _(fly, pl):
    fly.filter(
        pl.col("gene_id_2") == 7227
    )
    return


@app.cell
def _(pl):
    fly_aliases = pl.read_csv(
        "data/01_raw/wormhole_extracts/dm-aliases.txt",
        separator="\t",
        has_header=False,
        new_columns=["TaxID", "GeneID", "alias"]
    )

    fly_aliases
    return (fly_aliases,)


@app.cell
def _(fly_aliases, pl):
    fly_aliases.filter(
        pl.col("alias") == "100000006"
    )
    return


@app.cell
def _(fly, fly_aliases, pl):
    (
        fly.with_columns(
            gene_id_1=pl.col("gene_id_1").cast(pl.String()),
            gene_id_2=pl.col("gene_id_2").cast(pl.String())
        )
        .join(
            fly_aliases,
            left_on="gene_id_1",
            right_on="alias",
            #validate="m:1"
        )
        #.filter(
        #    pl.all_horizontal(pl.col(['GeneID', 'gene_id_2']).is_duplicated())
        #)
        .sort(pl.col(['GeneID', 'gene_id_2']), descending=False)
    )
    return


@app.cell
def _(fly, sns):
    sns.histplot(fly["association"])
    return


@app.cell
def _():
    import networkx as nx
    return (nx,)


@app.cell
def _(fly, nx, pl):
    G = nx.from_pandas_edgelist(
        (
            fly
            .filter(pl.col("association") >= 5)
            .to_pandas()
        ),          # networkx expects pandas
        source="gene_id_1",
        target="gene_id_2",
        edge_attr="association",  # stored as edge weight
    )
    return (G,)


@app.cell
def _(G):
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return


@app.cell
def _(G, nx):
    # Compute a dictionary of features per node
    print("Calc Degree")
    degree = dict(G.degree())
    print("Calc Weighted Degree")
    weighted_degree = dict(G.degree(weight="association"))
    #print("Calc Betweenness")
    #betweenness = nx.betweenness_centrality(G, weight="association")
    #print("Calc Closeness")
    #closeness = nx.closeness_centrality(G, distance="association")
    #print("Calc Eigen")
    #eigenvector = nx.eigenvector_centrality(G, weight="association", max_iter=1000)
    print("Calc Page rank")
    pagerank = nx.pagerank(G, weight="association")
    print("Calc Clustering")
    clustering = nx.clustering(G, weight="association")

    return clustering, degree, pagerank, weighted_degree


@app.cell
def _(G, clustering, degree, pagerank, pl, weighted_degree):
    # Collect all unique gene IDs
    genes = list(G.nodes())

    features_df = pl.DataFrame(
        {
            "gene_id":              genes,
            "degree":               [degree[g] for g in genes],
            "weighted_degree":      [weighted_degree[g] for g in genes],
            #"betweenness":          [betweenness[g] for g in genes],
            #"closeness":            [closeness[g] for g in genes],
            #"eigenvector":          [eigenvector[g] for g in genes],
            "pagerank":             [pagerank[g] for g in genes],
            "clustering_coeff":     [clustering[g] for g in genes],
        },
        strict=False
    )

    features_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
