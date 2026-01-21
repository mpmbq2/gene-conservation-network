import pandera.pandas as pa


class CoexpressionSchema(pa.DataFrameModel):
    gene_id_1: int = pa.Field()
    gene_id_2: int = pa.Field()
    association: float = pa.Field()

    class Config:
        strict = True
