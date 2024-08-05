import geopandas as geo
from utils.database_utils import PostgreSQL


class LoadLayer:
    def __init__(self):
        self.__layers_path = '/Users/diegolopes/Repositories/ms-usp-text-to-sql/data/IBGE_layers'
        self.__layers = {
            'malha_municipal': [
                'BR_Mesorregioes_2022',
                'BR_Microrregioes_2022',
                'BR_Municipios_2022',
                'BR_Pais_2022',
                'BR_RG_Imediatas_2022',
                'BR_RG_Intermediarias_2022',
                'BR_UF_2022'
            ],
            'malha_setores_censitários': [
                'BR_Malha_Preliminar_Distrito_2022',
                'BR_Malha_Preliminar_Subdistrito_2022',
                'BR_Malha_Preliminar_2022'
            ]
        }
        self.__tables_mapping = {
            'BR_Mesorregioes_2022': 'mesorregiao',
            'BR_Microrregioes_2022': 'microrregiao',
            'BR_Municipios_2022': 'municipio',
            'BR_Pais_2022': 'pais',
            'BR_RG_Imediatas_2022': 'rg_imediatas',
            'BR_RG_Intermediarias_2022': 'rg_intermediaria',
            'BR_UF_2022': 'unidade_federativa',
            'BR_Malha_Preliminar_Distrito_2022': 'distrito',
            'BR_Malha_Preliminar_Subdistrito_2022': 'subdistrito',
            'BR_Malha_Preliminar_2022': 'setor_censitario'
        }

    def extract_load_data(self):
        for key, datasets in self.__layers.items():
            for dataset in datasets:
                file = f"{self.__layers_path}/{key}/{dataset}/{dataset}.shp"
                df = geo.read_file(file)

                df.columns = map(str.lower, df.columns)

                postgres = PostgreSQL(user='postgres', password='DoL#7478', database='geo_data')
                engine = postgres.get_engine()
                df.to_postgis(self.__tables_mapping.get(dataset), engine, if_exists='replace', index=False)

layer =LoadLayer()
layer.extract_load_data()
