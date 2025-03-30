import pandas as pd
import numpy as np
import os, sys

from scipy.stats import ks_2samp
from scipy.stats import pearsonr
from statistics import mode

def definition():
    return {
    'id':'Chave Ãºnica de uma solicitaÃ§Ã£o de cliente',
    'age':'Idade do cliente',
    'monthly_income':'Renda mensal informada pelo cliente no momento do cadastro',
    'collateral_value':'Valor do automÃ³vel que serÃ¡ dado em garantia',
    'loan_amount':'Valor solicitado pelo cliente para emprÃ©stimo',
    'city':'Cidade do cliente',
    'state':'Estado do cliente',
    'collateral_debt':'Valor que o automovel do cliente tem de dÃ­vida (ex. Valor que ainda estÃ¡ financiado)',
    'verified_restriction':'Indica se o cliente possui alguma restriÃ§Ã£o/pendÃªncia verificada. (NA significa que nÃ£o houve consulta da situaÃ§Ã£o do cliente)',
    'dishonored_checks':'Indica se o cliente possui cheques sem fundo',
    'expired_debts':'Indica se o cliente possui dÃ­vidas vencidas',
    'banking_debts':'Indica se o cliente possui divÃ­das bancÃ¡rias',
    'commercial_debts':'Indica se o cliente possui dividas comerciais',
    'protests':'Indica se o cliente possui protestos',
    'marital_status':'Estado civil',
    'informed_restriction':'RestriÃ§Ã£o informada pelo cliente',
    'loan_term':'Prazo do emprÃ©stimo',
    'monthly_payment':'Pagamento mensal do emprÃ©stimo',
    'informed_purpose':'Motivo pelo qual o cliente deseja o emprÃ©stimo',
    'auto_brand':'Marca do carro',
    'auto_model':'Modelo do carro',
    'auto_year':'Ano do carro',
    'pre_approved':'Lead prÃ©-aprovado (apenas leads prÃ©-aprovados sÃ£o atendidos)',
    'form_completed':'Ficha cadastral completa pelo cliente',
    'sent_to_analysis':'Enviado para anÃ¡lise de crÃ©dito',
    'channel':'Canal de entrada do lead',
    'zip_code':'CEP anonimizado do cliente',
    'landing_page':'PÃ¡gina inicial que o cliente acessou no site',
    'landing_page_product':'Produto da pÃ¡gina inicial do cliente',
    'gender':'GÃªnero do cliente',
    'education_level':'Grau de instruÃ§Ã£o do cliente',
    'utm_term':'Tipo de dispositivo do cliente (c = computer, m = mobile, t = tablet)'
    }

def get_all_information(df, column_filter, column_target):
    # Conteudo relacionado aos casos pré aprovados
    pre_approved = df[df[column_filter] == 1].shape[0]
    not_pre_approved = df[df[column_filter] == 0].shape[0]

    # Conteudo da target do modelo para análise da problemática
    credit_approved = df[
        (df[column_filter] == 1) &
        (df[column_target] == 1)
    ].shape[0]

    not_credit_approved = df[
        (df[column_filter] == 1) &
        (df[column_target] == 0)
    ].shape[0]

    print('# Analisando a problematica dos dados #')
    print(f'De todos os casos nessa base, reprovados temos {not_pre_approved}')
    print(f'De todos os casos nessa base, pré-aprovado temos {pre_approved}')
    print(f'Que realmente aprovamos para a analise de crédito foram {credit_approved}')
    print(f'Que realmente reprovamos para a analise de crédito foram {not_credit_approved}')

def summary_data(df):
    print('Essa função apenas analisa os casos int e float')
    lista_ = []

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:  
            linha = {
                'Coluna': col,
                'num_missings': df[col].isna().sum(),
                'minimo': df[col].min(),
                'primeiro_quartil': df[col].quantile(0.25),
                'mediana': df[col].median(),
                'media': df[col].mean(),
                'terceiro_quartil': df[col].quantile(0.75),
                'valores_distintos': df[col].nunique(),
                'tamanho_base': df.shape[0],
            }
            lista_.append(linha)

    aux_estatisticas = pd.DataFrame(lista_)
    aux_estatisticas['perc_missing'] = aux_estatisticas['num_missings']/aux_estatisticas['tamanho_base']

    return aux_estatisticas

def summary_data_cat(df):
    lista_ = []

    for col in df.columns:
        if df[col].dtype in ['object']:
            linha = {
                'coluna': col,
                'num_missings': df[col].isna().sum(),
                'valores_distintos': df[col].nunique(),
                'tamanho_base': df.shape[0]
            }  
            lista_.append(linha)
    
    aux_estatisticas = pd.DataFrame(lista_)
    aux_estatisticas['perc_missing'] = aux_estatisticas['num_missings']/aux_estatisticas['tamanho_base']

    return aux_estatisticas

def ks_test(df, y, cols):
    df_ = df.copy()
    resultados = []
    
    for var in cols:    
        df_[var] = df_[var].fillna(-1)
        group_0 = df_[df_[y] == 0][var]
        group_1 = df_[df_[y] == 1][var]
        
        ks_stat, ks_p_value = ks_2samp(group_0, group_1)
        
        resultados.append({
            'explicativa': var,
            'ks': ks_stat,
            'p_valor_ks': ks_p_value,
        }) 
    
    return pd.DataFrame(resultados)

def ks_test_cat(df, y, cols):
    df_ = df.copy()
    resultados = []
    
    for var in cols:    
        inputer = mode(df[var])
        df_[var] = df_[var].fillna(inputer)
        group_0 = df_[df_[y] == 0][var]
        group_1 = df_[df_[y] == 1][var]
        
        ks_stat, ks_p_value = ks_2samp(group_0, group_1)
        
        resultados.append({
            'explicativa': var,
            'ks': ks_stat,
            'p_valor_ks': ks_p_value,
        }) 
    
    return pd.DataFrame(resultados) 

def check_proportion(df, x, target):
    listing = []
    for cat in df[x].unique().tolist():
        if str(cat) != 'nan':
            df_query = df[df[x] == cat]
        else:
            df_query = df[df[x].isna()]
        value_one = df_query[df_query[target] == 1].shape[0]
        value_zero = df_query[df_query[target] == 0].shape[0]

        listing.append({
            'cat': cat,
            'target':value_one,
            'nao_target':value_zero,
            'prop': value_one/(value_one + value_zero)
        })

    return pd.DataFrame(listing)

class StartEda():
    def __init__(self, df, direct_input_cols):
        self.df = df
        self.direct_input_cols = direct_input_cols
        self.direct_input()

    def direct_input(self):
        for col in self.direct_input_cols:
            self.df[col] = self.df[col].fillna(-1)

        self.df['channel'] = self.df['channel'].fillna('nao_conhecido') 

        lista_fill = [
            '/emprestimos/garantia-veiculo/solicitar',
            '/emprestimos/garantia-veiculo/guia-bolso',
            '/blog/artigos/emprestimos/emprestimo-com-garantia-de-veiculo-alienado'
        ]

        # Adicionando as alterações para landing_page_product, são dois maps pois depende de 2 regras distintas
        self.df['landing_page_product'] = list(
            map(
                lambda x,y: 'GarantiaVeiculo' if y in lista_fill else x,
                self.df['landing_page_product'], self.df['landing_page']
            )
        )

        self.df['landing_page_product'] = list(
            map(
                lambda x, y: 'ProdutoDesconhecido' if x == 'nao_conhecido' else y,
                self.df['channel'], self.df['landing_page_product']
            )
        )

        self.df['gender'] = self.df['gender'].map({'male':1, 'female':0})
        self.df['gender'] = self.df['gender'].fillna('-1')

        self.df.drop(['Unnamed: 0','utm_term','landing_page'],axis=1,inplace=True)

                    



