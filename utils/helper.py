from pathlib import Path
import re

import numpy as np
import pandas as pd

class Const:
    PROJECT_DIR = Path('..').resolve()
    INPUT_DIR = PROJECT_DIR.joinpath('input')
    INTERIM_DIR = PROJECT_DIR.joinpath('interim')
    BLACK_LIST = ["id", "price", "price_m2", "price_median"]
    TARGET_NAME = "price"


def get_data(INPUT_DIR=Const.INPUT_DIR):
    
    df_train = pd.read_hdf(INPUT_DIR.joinpath('train_warsaw_property.h5'))
    df_test = pd.read_hdf(INPUT_DIR.joinpath('test_warsaw_property.h5'))
    
    df_all = pd.concat([df_train, df_test])
    
    assert df_train.price.notna().all()
    assert df_train.shape == df_all[df_all.price.notna()].shape
    assert df_test.shape[0] == df_all[df_all.price.isna()].shape[0]
    assert df_test.shape[1] == df_all[df_all.price.isna()].shape[1] - 1 
    
    return df_train, df_test, df_all

def split_df_all_to_train_test(df_all):
    
    df_train = df_all[df_all.price.notna()]
    df_test = df_all[df_all.price.isna()]
    del df_test['price']
    
    return df_train, df_test

def save_to_interim(
    df_all, 
    INTERIM_DIR = Const.INTERIM_DIR, 
    pickle_name = 'df_all.parquet'
):
    
    df_train, df_test = split_df_all_to_train_test(df_all)
    df_train.to_hdf(INTERIM_DIR.joinpath('train_warsaw_property.h5'), 'df', mode='w')
    df_test.to_hdf(INTERIM_DIR.joinpath('test_warsaw_property.h5'), 'df', mode='w')

    
def get_price_from_text(df):
    
    def find_max_digits_in_text(text):
        number_list = [float(number) for number in (re.findall('[0-9]+', text.replace(' ', '')))]
    
        if number_list:
            max_number_of_list = max(number_list)
            
            if (max_number_of_list >= 1e5) & (max_number_of_list <= 4e6 - 1):
                return max_number_of_list
    
    return df['text'].apply(find_max_digits_in_text)


def append_external_geo_feats(df): 
    
    wiki_stats_miasta_df = pd.read_csv(Const.INTERIM_DIR.joinpath('statystyki-miasta-wiki.csv'))
    wiki_stats_miasta_df = wiki_stats_miasta_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    
    df = df.join(
        df[['loc0', 'loc1', 'loc2']]
            .applymap(
                lambda x: x.lower() if isinstance(x, str) else x
            ).merge(
                wiki_stats_miasta_df,
                how='left',
                left_on=['loc0','loc1','loc2'],
                right_on=['Województwo', 'Powiat', 'Miasto'],
            )[
                ['Powierzchnia', 'Liczba ludności', 'Gęstość zaludnienia']
            ].rename(columns=lambda name: f'city_{name}')
    )
    
   
    wiki_stats_woj_df = pd.read_csv(Const.INTERIM_DIR.joinpath('statystyki-woj.csv'))
    wiki_stats_woj_df = wiki_stats_woj_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    
    
    df = df.join(
        df
        .set_index('loc0')
        .join(
            wiki_stats_woj_df.set_index('Województwo'),
            how='left'
        ).set_index('id')
        [
            ['Ogółem', 'Mężczyźni', 'Kobiety']
        ]
    )
    
    return df

def preprocessing_data(df): 
    
    # Wyciągnięcie informacji z location
    if "location" in df:
        df = df.join(df['location'].apply(pd.Series))
        del df['location']
        df = df.rename(columns=lambda x: f'loc{x}' if x in range(5) else x)
    
    # Wyciągnięcie informacji z stats
    if "stats" in df:
        df = df.join(df['stats'].apply(pd.Series))
        del df['stats']
    
    # Konwersja metrażu
    if df['area'].dtype == object:
        assert df['area'].str.endswith('m²').all()
        df['area_num'] = df['area'].str.replace(',','.').str.split('m²').apply(lambda x: float(x[0].replace(' ','')))
        area_num_99 = np.percentile(df["area_num"], 99)
        df["area_norm"] = df["area_num"].map(lambda x: x if x <= area_num_99 else area_num_99  )
        df["area_num_log"] = np.log(df["area_num"])
    
    for feat in df.select_dtypes("object"):
        try:
            df["{}_cat".format(feat)] = df[feat].factorize()[0]
        except:
            print("can not factorize {}".format(feat))
    
    if not 'price_from_text' in df:
        df['price_from_text'] = get_price_from_text(df)

    if "price_m2" not in df:
        df["price_m2"] = df["price"] / df["area_num"]
    
    if "loc0_price_median" not in df:
        agg_funcs = ["median", "mean"]
        for grp_feat in ["price", "price_m2"]:
            for loc_num in ["loc0", "loc1", "loc2"]:
                loc_grp = df[ [grp_feat, loc_num] ].groupby(loc_num).agg(agg_funcs).to_dict()
                for agg_func in agg_funcs:
                    df["{0}_{1}_{2}".format(loc_num, grp_feat, agg_func)] = df[loc_num].map(loc_grp[ (grp_feat, agg_func) ])
    
    if "price_median" not in df:
        df["price_median"] = df["area_norm"] * df["loc1_price_m2_median"]
    
    if "floor_num" not in df:
        floors_dict = {"parter": 0, "> 10": 11, "poddasze": -2, "suterena": -1 }
        df["floor_num"] = df["floor"].map(lambda x: floors_dict.get(x, x)).fillna(-10).astype("int")
        df["floors_in_building_num"] = (df["floors_in_building"].map(
            lambda x: str(x).split("z")[-1].replace(")", "") if str(x) != "nan" else -1
        ).astype("float"))
        
        df["floors_in_building_num_norm"] = df["floors_in_building_num"].map(lambda x: x if x < 20 else 25)
    
    if "build_year" not in df:
        df["build_year"] = df["rok budowy"].fillna(-1).astype("int")
    
    if "rental" not in df:
        df["rental"] = df["czynsz"].map(lambda x: str(x).split("zł")[0].replace(" ", "").replace(",", ".") if str(x) != "nan" else -1 )
        df["rental"] = df["rental"].map(lambda x: float(str(x).replace("eur", "") * 4) if "eur" in str(x) else x).astype("float")


    if "build_material_cat" not in df:
        cat_feats = {
            "materiał budynku": "build_material_cat",
            "okna": "window_cat",
            "stan wykończenia": "property_completion_cat",
            "rodzaj zabudowy": "property_type_cat",
            "ogrzewanie": "property_heating_cat",
            "forma własności": "own_property_cat"
        }

        for feat_name, feat_new_name in cat_feats.items():
            df[feat_new_name] = df[feat_name].factorize()[0]

            #ohe
            df_dummies = pd.get_dummies(df[feat_name])
            df_dummies.columns = ["{0}_{1}".format(feat_new_name, x) for x in df_dummies.columns]
            df = pd.concat( [df, df_dummies], axis=1)
    
    if "city_Powierzchnia" not in df:
        df = append_external_geo_feats(df)
        
    return df




def get_X_y(df, feats, target="price"):
    
    df_train, df_test = split_df_all_to_train_test(df)
    print(df_train.shape, df_test.shape) #sprawdzamy ile kolumn i wierszy mamy w każdym zbiorze

    X_train, y_train = df_train[feats].values, df_train[target].values
    X_test = df_test[feats].values

    return X_train, X_test, y_train


def overwrite_prediction_by_data_leak(df_test):
    """ 
    Dla 3 mieszkań w próbce test w miejsce powierzchni najprawdopodobniej jest wpisana cena.
    Dla mieszkania o indeksie 29299 ta wartość z area jest podana w tekście jako wartość mieszkania.
    """
    
    df_test.loc[[3260, 29299, 60390], 'price'] = df_test.loc[[3260, 29299, 60390], 'area_num'].astype(float)
    df_test.loc[[29299], 'area'] = 60.2
    
    return df_test


