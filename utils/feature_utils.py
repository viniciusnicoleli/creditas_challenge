import pandas as pd
import numpy as np
import os, sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    OneHotEncoder
)

from typing import Tuple, List, Dict
    
def pipeline_numerical(numerical_scaler, num_cols):
    if num_cols.empty:
        return None
    return ('pipe_num',Pipeline(
        steps=[
            ("selector_numerical", ColumnTransformer([("filter_num_cols", "passthrough", num_cols.columns.values)], remainder='drop')),
            ("num_imputer", SimpleImputer(strategy='median')),
            ("NumScaler", numerical_scaler)
        ]
    ))

def pipeline_one_hot(one_hot_cols, scaler):
    if one_hot_cols.empty:
        return None
    return ('pipe_hot',Pipeline(
        steps=[
            ("selector_one_hot", ColumnTransformer([("filter_one_cols", "passthrough", one_hot_cols.columns.values)], remainder='drop')),
            ("one_imputer", SimpleImputer(strategy='most_frequent')),
            ("OneHotEncoder", scaler)
        ]
    ))

def pipeline_ordinal(ordinal_cols, scaler):
    if ordinal_cols.empty:
        return None
    return ('pipe_ord',Pipeline(
        steps=[
            ("selector_ord_hot", ColumnTransformer([("filter_ord_cols", "passthrough", ordinal_cols.columns.values)], remainder='drop')),
            ("one_imputer", SimpleImputer(strategy='most_frequent')),
            ("OrdinalEncoder", scaler)
        ]
    ))

def pipeline_select_sem_mexer(ignore_features):
    return ('pipe_sem_mexer',Pipeline(
        steps=[
            ("selector_one_hot", ColumnTransformer([("filter_ignore_cols", "passthrough", ignore_features.columns.values)], remainder='drop'))
        ]
    ))



def create_pipeline(df: pd.DataFrame, 
                    columns_ignore: List, 
                    columns_include_without_transformation: List,
                    ordinal_order: Dict[str,List] = None, 
                    numerical_scaler = StandardScaler(),
                    is_eda = False):
    
    #Eliminando do Pipeline colunas que precisam ser ignoradas
    columns_ignore_all = list(ordinal_order.keys()) + columns_ignore if ordinal_order else columns_ignore

    print(f'Ignorando essas colunas tanto para OneHot quanto para Numerical: {columns_ignore_all}')

    # Criando os dataframes que contém as features para transformação
    numerical_features: pd.DataFrame = df[[col for col in df.select_dtypes(include=['float','int']).columns if col not in columns_ignore_all]]
    one_hot_features: pd.DataFrame = df[[col for col in df.select_dtypes(include=['object'], exclude=['datetime']).columns if col not in columns_ignore_all]]
    ordinal_features: pd.DataFrame = df[list(ordinal_order.keys())] if ordinal_order else pd.DataFrame()
    ignore_features: pd.DataFrame = df[columns_include_without_transformation]

    ord_scaler = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    ohe_scaler = OneHotEncoder(handle_unknown='ignore')

    print(numerical_features.columns.tolist())
    print(one_hot_features.columns.tolist())
    print(f'DataFrames criados sendo numericas:{numerical_features.shape[1]}, one_hot:{one_hot_features.shape[1]}, ordinal:{ordinal_features.shape[1]}')

    # # Criando o Pipeline NumScaler
    pipe_num: Tuple = pipeline_numerical(numerical_scaler=numerical_scaler, 
                                         num_cols=numerical_features)
     # Criando o Pipeline OneHotEncoder
    pipe_one_hot: Tuple = pipeline_one_hot(one_hot_cols=one_hot_features, scaler= ohe_scaler if is_eda == False else None)
    
    # Criando o Pipeline Ordinal
    pipe_ordinal: Tuple = pipeline_ordinal(ordinal_cols=ordinal_features, scaler= ord_scaler if is_eda == False else None)

    pipe_sem_mexer = pipeline_select_sem_mexer(ignore_features=ignore_features)

    print(f'Pipelines criados, criando of FeatureUnion')

    return (FeatureUnion(
        transformer_list=[pipe for pipe in [pipe_num, pipe_one_hot, pipe_ordinal, pipe_sem_mexer] if pipe is not None],
        verbose=True
    ), numerical_features.columns, one_hot_features.columns, ordinal_features.columns, ignore_features.columns)

def get_class(brand):
    if brand in [
        'GM - Chevrolet', 
        'Fiat', 
        'VW - VolksWagen', 
        'Renault', 
        'Peugeot',
        'Ford',
        'Citroën',
        'Kia Motors',
        'Hyundai',
    ]:
        return 'popular'
    
    elif brand in [
        'Honda', 
        'Suzuki', 
        'Toyota', 
        'Nissan', 
        'Mitsubishi',
        'Volvo', 
        'Jeep', 
        'Audi', 
        'BMW', 
        'Subaru', 
        'MINI',
        'Dodge'
    ]:
        return 'exclusivo'
    
    elif brand in [
        'SHINERAY',
        'smart', 
        'JAC',
        'LIFAN', 
        'CHERY',
        'RELY', 
        'EFFA',
        'FOTON', 
        'SSANGYONG',
        'GEELY',
        'Seat',
        'Buggy',
        '92', 
        'HAFEI',
        'Cross Lander',
        'Wake', 
        'CHANA',
        '92',
        '91',
        'CHANGAN',
        'Walk', 
        'Mahindra',
        'JINBEI'
    ]:
        return 'incomum'
    
    elif brand in [
        'Dodge',
        'Mercedes-Benz',
        'Land Rover',
        'Chrysler',
        'Porsche',
        'Ferrari',
        'RAM'
    ]:
        return 'super_carros'
    
    else:
        return 'desconhecido'
    
class ModelUtils():
    def __init__(self, df, target, columns_ignore, columns_without):
        self.df = df
        self.target = target
        self.X, self.y = self.splitxy()
        self.columns_ignore = columns_ignore
        self.columns_without = columns_without
        self.df_salarios = pd.read_excel('../data/salario_medio_empresas_UF.xlsx')
        self.df_class_purpose = pd.read_csv('../data/class_purpose.csv').drop(['Unnamed: 0'],axis=1)

    def splitxy(self):
        return self.df.drop(self.target, axis = 1), self.df[self.target]

    def train_test_val(self):
        # Split para casos em que não é um OOT, mas sim um OOS
        # Para isso X_val será o Holdout Set.
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, 
            self.y, 
            test_size=0.20, 
            random_state=42, 
            stratify = self.y)

        # Utilizando a base feita no split anteriormente para encontrar a base X_test que é a OOS
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, 
            y_train, 
            test_size=0.25, 
            random_state=42, 
            stratify = y_train)
        return X_train, y_train, X_test, y_test, X_val, y_val

    def transform_dataframe(self, set, columns):
        set_eda = self.pipe_prep[0].transform(set)
        return pd.DataFrame(set_eda, columns=columns)
    
    @staticmethod
    def change_types(df):
        for col in df.columns:
            if col in ['monthly_income', 'collateral_value', 'loan_amount', 'collateral_debt', 'monthly_payment']:
                df[col] = df[col].astype('float64')
            elif col in ['age','auto_year']:
                df[col] = df[col].astype('int64') 
        return df
    
    def get_train_set_fe(self, X_train_transformed, y_train):
        df_get_info = X_train_transformed
        df_get_info['target'] = y_train
        df_get_info['IdadeCarro'] = df_get_info.apply(lambda x: 2025 - x['auto_year'], axis=1)
        df_get_info['ValorGarantiaRealAuto'] = df_get_info.apply(lambda x: x['collateral_value'] - x['collateral_debt'],axis=1)
        df_get_info['PropLoanReal'] = df_get_info.apply(lambda x: x['loan_amount']/x['ValorGarantiaRealAuto'] 
                                                                  if x['ValorGarantiaRealAuto'] > 0 else -1,axis=1)
        df_get_info['PropLoanMonthlyIncome'] = df_get_info.apply(lambda x: 
                                                                 x['loan_amount']/x['monthly_income'] 
                                                                 if x['monthly_income'] > 0 else -1,axis=1
        )
        df_get_info['PropLoanGarantiaVeiculo'] = df_get_info.apply(lambda x: 
                                                                   x['loan_amount']/x['collateral_value'] 
                                                                   if x['collateral_value'] > 0 else -1,axis=1
        )

        return df_get_info
    
    def type_change(self, df):
        for col in [
        'IdadeCarro', 
        'QtdSalariosMinimos', 
        'ValorGarantiaRealAuto',
        'PropLoanGarantiaVeiculo', 
        'PropLoanReal', 
        'PropLoanMonthlyIncome',
        # 'MeanPropLoanReal_Brand', 
        # 'MeanPropIdadeCarro_Brand',
        # 'MeanValorGarantiaRealAuto_Brand', 
        # 'MeanPropLoanMonthly_ZipCode',
        # 'MeanPropLoanGarantiaVeiculo_ZipCode', 
        'numero_de_empresas_atuantes',
        'PropIncomeSalarioMedioMensal'
        ]:
            if col not in ['IdadeCarro', 'numero_de_empresas_atuantes']:
                df[col] = df[col].astype('float64')
            else:
                df[col] = df[col].astype('int64')
        
        return df
            
    def start_pipeline(self):
        X_train, y_train, X_test, y_test, X_val, y_val = self.train_test_val()

        self.df_get_info = self.get_train_set_fe(
            X_train_transformed=X_train.copy(),
            y_train=y_train
        )

        X_train = self.starter_features(df=X_train)
        X_val = self.starter_features(df=X_val)
        X_test = self.starter_features(df=X_test)

        self.pipe_prep = create_pipeline(
            df=X_train,
            columns_ignore=self.columns_ignore,
            columns_include_without_transformation=self.columns_without,
            ordinal_order=None,
            numerical_scaler=None,
            is_eda=True
        )

        # Fitando o Pipeline
        self.pipe_prep[0].fit(X_train)
        
        columns = self.pipe_prep[1].tolist() + \
                  self.pipe_prep[2].tolist() + \
                  self.pipe_prep[3].tolist() + \
                  self.pipe_prep[4].tolist()
        
        self.X_train_transformed = self.change_types(self.transform_dataframe(X_train,columns=columns))
        self.X_val_transformed = self.change_types(self.transform_dataframe(X_val,columns=columns))
        self.X_test_transformed = self.change_types(self.transform_dataframe(X_test,columns=columns))

        self.X_train_transformed = self.complete_features(df=self.X_train_transformed)
        self.X_val_transformed = self.complete_features(df=self.X_val_transformed)
        self.X_test_transformed = self.complete_features(df=self.X_test_transformed)

        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        self.X_train_transformed = self.type_change(df=self.X_train_transformed)
        self.X_val_transformed = self.type_change(df=self.X_val_transformed)
        self.X_test_transformed = self.type_change(df=self.X_test_transformed)

        return (self.X_train_transformed, self.X_val_transformed, self.X_test_transformed, y_train, y_val, y_test)
    
    def starter_features(self, df):
        # Classificando a marca
        df['ClassMarca'] = df.apply(lambda x: get_class(x['auto_brand']),axis=1)
        df['IdadeCarro'] = df.apply(lambda x: 2025 - x['auto_year'], axis=1) 

        # Diminuindo a cardinalidade de state
        uf_para_regiao = {
            'PR': 'Sul', 
            'SC': 'Sul', 
            'RS': 'Sul',
            'SP': 'Sudeste', 
            'RJ': 'Sudeste', 
            'MG': 'Sudeste', 
            'ES': 'Sudeste',
            'DF': 'Centro-Oeste', 
            'GO': 'Centro-Oeste', 
            'MT': 'Centro-Oeste', 
            'MS': 'Centro-Oeste',
            'BA': 'Nordeste', 
            'SE': 'Nordeste', 
            'AL': 'Nordeste', 
            'PE': 'Nordeste',
            'PB': 'Nordeste', 
            'RN': 'Nordeste', 
            'CE': 'Nordeste', 
            'PI': 'Nordeste', 
            'MA': 'Nordeste',
            'PA': 'Norte', 
            'AP': 'Norte', 
            'RR': 'Norte', 
            'AM': 'Norte',
            'RO': 'Norte', 
            'AC': 'Norte', 
            'TO': 'Norte'
        }

        df['regiao'] = df['state'].map(uf_para_regiao)
        df['isSP'] = df['state'].apply(lambda x: 1 if x == 'SP' else 0)
        df['QtdSalariosMinimos'] = df.apply(lambda x: x['monthly_income']/1512, axis=1)
        df['ValorGarantiaRealAuto'] = df.apply(lambda x: x['collateral_value'] - x['collateral_debt'],axis=1)
        df['PropLoanGarantiaVeiculo'] = df.apply(lambda x: x['loan_amount']/x['collateral_value'] if x['collateral_value'] > 0 else -1,axis=1)
        df['PropLoanReal'] = df.apply(lambda x: x['loan_amount']/x['ValorGarantiaRealAuto'] if x['ValorGarantiaRealAuto'] > 0 else -1,axis=1)
        df['PropLoanMonthlyIncome'] = df.apply(lambda x: x['loan_amount']/x['monthly_income'] if x['monthly_income'] > 0 else -1,axis=1)

        # df = self.advanced_features_auto_brand(df=df)
        # df = self.advanced_features_zip_code(df=df)

        df.rename({
            'state':'UF'
        },axis=1, inplace=True)

        df = df.merge(
                    self.df_salarios[['UF','salario_medio_mensal ','numero_de_empresas_atuantes']],
                    on='UF',
                    how='left'
        )
        df['PropIncomeSalarioMedioMensal'] = df.apply(lambda x: x['monthly_income']/x['salario_medio_mensal '],axis=1)

        # Adicionando a classe do proposito do empréstimo
        df['id'] = df['id'].astype('int64')
        df = df.merge(self.df_class_purpose[['id', 'purpose_normalized', 'ClassPurpose']], on=['id'], how='left')
        df['ClassPurpose'] = df.apply(lambda x: 
                'nothing' if x['purpose_normalized'] == 'nada a declarar' else x['ClassPurpose'],axis=1
        )
        df['ClassPurpose'] = df['ClassPurpose'].map({'debt':1, 'investment':0})

        df = df.drop([
            'auto_year',
            'auto_brand',
            'UF',
            'zip_code',
            'salario_medio_mensal ',
            'purpose_normalized'
        ],axis=1)

        return df
    
    def get_train_set_information(self):
        loan_percentile_75 = np.percentile(self.X_train_transformed['loan_amount'], 75)
        collateral_debt_percentile_50 = np.percentile(self.X_train_transformed['collateral_debt'], 50)
        collateral_value_percentile_50 = np.percentile(self.X_train_transformed['collateral_value'], 50)
        collateral_value_debt_percentile_50 = np.percentile(self.X_train_transformed['collateral_value'] + 
                                                            self.X_train_transformed['collateral_debt'], 50)
        print(
            loan_percentile_75,
            collateral_debt_percentile_50,
            collateral_value_percentile_50,
            collateral_value_debt_percentile_50
        )
        return (loan_percentile_75, 
                collateral_debt_percentile_50, 
                collateral_value_percentile_50 ,
                collateral_value_debt_percentile_50)

    def get_train_set_info_auto_brand(self):
        df_get_info_apply = self.df_get_info.query("target == 1").groupby('auto_brand').agg(
            mean_prop_loan_real=('PropLoanReal', 'mean'),
            mean_idade_carro=('IdadeCarro', 'mean'),
            mean_garantia_real=('ValorGarantiaRealAuto','mean')
        ).reset_index()

        mean_prop_loan_real = dict(df_get_info_apply[['auto_brand','mean_prop_loan_real']].values)
        mean_idade_carro = dict(df_get_info_apply[['auto_brand','mean_idade_carro']].values)
        mean_garantia_real = dict(df_get_info_apply[['auto_brand','mean_garantia_real']].values)

        return (mean_prop_loan_real, mean_idade_carro, mean_garantia_real)
    
    def get_train_set_info_zip_code(self):
        df_get_info_zip_code_grouped = self.df_get_info.groupby('zip_code').agg(
            count_total_cases=('zip_code','count'),
            count_credit_cases=('target','sum'),
            mean_loan_income_target=('PropLoanMonthlyIncome', lambda x: x[self.df_get_info.loc[x.index, 'target'] == 1].mean()),
            mean_loan_collateral_target=('PropLoanGarantiaVeiculo', lambda x: x[self.df_get_info.loc[x.index, 'target'] == 1].mean())
        ).reset_index()

        df_get_info_zip_code_grouped['mean_loan_income_target'] = df_get_info_zip_code_grouped['mean_loan_income_target'].fillna(-1)
        df_get_info_zip_code_grouped['mean_loan_collateral_target'] = df_get_info_zip_code_grouped['mean_loan_collateral_target'].fillna(-1)
        df_get_info_zip_code_grouped['PropHitRate'] = df_get_info_zip_code_grouped.apply(lambda x: x['count_credit_cases']/x['count_total_cases'],axis=1)

        mean_loan_income = dict(df_get_info_zip_code_grouped[['zip_code','mean_loan_income_target']].values)
        mean_loan_collateral_target = dict(df_get_info_zip_code_grouped[['zip_code','mean_loan_collateral_target']].values)
        hit_rate = dict(df_get_info_zip_code_grouped[['zip_code','PropHitRate']].values)

        return (mean_loan_income, mean_loan_collateral_target, hit_rate)

    def complete_features(self, df):
        loan_percentile_75, collateral_debt_percentile_50, collateral_value_percentile_50, collateral_value_debt_percentile_50 = self.get_train_set_information()

        # Monthly_payment
        df['PropVlrParcelaMensal'] = df.apply(lambda x: x['monthly_payment']/x['monthly_income'] if x['monthly_income'] > 0 else -1,axis=1)
        df['PropMonthyPaymentLoan'] = df.apply(lambda x: x['monthly_payment']/x['loan_amount'] if x['loan_amount'] > 0 else -1,axis=1)
        df['PropLoanCollateral'] = df.apply(
            lambda x: ((x['loan_amount'] + x['collateral_debt'])/(x['monthly_income']*12)) if x['monthly_income'] > 0 else -1,axis=1
        )

        df['PropDebtIncome'] = df.apply(lambda x: x['collateral_debt']/x['monthly_income'] if x['monthly_income'] > 0 else -1,axis=1)
        df['PropDebtValue'] = df.apply(lambda x: x['collateral_debt']/x['collateral_value'] if x['collateral_value'] > 0 else -1,axis=1)
        df['PropValueIncome'] = df.apply(lambda x: x['collateral_value']/x['monthly_income'] if x['monthly_income'] > 0 else -1,axis=1)
        df['isLoanAbove75Percent'] = df.apply(lambda x: x['loan_amount']/loan_percentile_75, axis=1)
        df['isValueAbove50Percent'] = df.apply(lambda x: x['collateral_value']/collateral_value_percentile_50, axis=1)
        df['isRealAbove50Percent'] = df.apply(lambda x: (x['collateral_value']-x['collateral_debt'])/collateral_value_debt_percentile_50, axis=1)

        # Diminuindo a dimensionalidade do dado
        for col in ['monthly_income', 'collateral_value', 'loan_amount', 'collateral_debt', 'monthly_payment']:
            df[col] = np.arcsinh(df[col])

        return df
    
    def advanced_features_auto_brand(self, df):
        mean_prop_loan_real, mean_idade_carro, mean_garantia_real = self.get_train_set_info_auto_brand()

        df['Brand_MeanPropLoanReal_Credit'] = df.apply(
            lambda x: -1 if not mean_prop_loan_real.get(x['auto_brand']) else mean_prop_loan_real.get(x['auto_brand']), axis=1
        )

        df['Brand_MeanIdadeCarro_Credit'] = df.apply(
            lambda x: -1 if not mean_idade_carro.get(x['auto_brand']) else mean_idade_carro.get(x['auto_brand']), axis=1
        )

        df['Brand_MeanGarantiaReal_Credit'] = df.apply(
            lambda x: -1 if not mean_garantia_real.get(x['auto_brand']) else mean_garantia_real.get(x['auto_brand']), axis=1
        )

        df['MeanPropLoanReal_Brand'] = df.apply(lambda x: x['PropLoanReal']/x['Brand_MeanPropLoanReal_Credit'],axis=1)
        df['MeanPropIdadeCarro_Brand'] = df.apply(lambda x: x['IdadeCarro']/x['Brand_MeanIdadeCarro_Credit'],axis=1)
        df['MeanValorGarantiaRealAuto_Brand'] = df.apply(lambda x: x['ValorGarantiaRealAuto']/x['Brand_MeanGarantiaReal_Credit'],axis=1)

        return df
    
    def advanced_features_zip_code(self, df):
        mean_loan_income, mean_loan_collateral_target, hit_rate = self.get_train_set_info_zip_code()

        df['ZipCode_MeanPropLoanIncome_Credit'] = df.apply(
            lambda x: -1 if not mean_loan_income.get(x['zip_code']) else mean_loan_income.get(x['zip_code']), axis=1
        )

        df['ZipCode_MeanLoanCollateralTarget_Credit'] = df.apply(
            lambda x: -1 if not mean_loan_collateral_target.get(x['zip_code']) else mean_loan_collateral_target.get(x['zip_code']), axis=1
        )

        df['ZipCode_HitRate_Credit'] = df.apply(
            lambda x: -1 if not hit_rate.get(x['zip_code']) else hit_rate.get(x['zip_code']), axis=1
        )

        df['MeanPropLoanMonthly_ZipCode'] = df.apply(lambda x: x['PropLoanMonthlyIncome']/x['ZipCode_MeanPropLoanIncome_Credit'],axis=1)
        df['MeanPropLoanGarantiaVeiculo_ZipCode'] = df.apply(lambda x: x['PropLoanGarantiaVeiculo']/x['ZipCode_MeanLoanCollateralTarget_Credit'],axis=1)

        return df
        

