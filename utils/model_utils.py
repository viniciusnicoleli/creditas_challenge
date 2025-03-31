from sklearn.preprocessing import (OneHotEncoder, 
                                   OrdinalEncoder, 
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.metrics import (precision_recall_curve, 
                             PrecisionRecallDisplay, 
                             roc_curve, 
                             RocCurveDisplay, 
                             auc)
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, 
                             average_precision_score, 
                             classification_report, 
                             confusion_matrix,
                             recall_score,
                             precision_score)
from sklearn.pipeline import Pipeline, FeatureUnion
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from typing import List, Dict, Tuple                   
from matplotlib import pyplot as plt
from IPython.display import display
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shap

from lightgbm import LGBMClassifier


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
            ("selector_ord", ColumnTransformer([("filter_one_cols", "passthrough", ordinal_cols.columns.values)], remainder='drop')),
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

    ord_scaler = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1, categories=list(ordinal_order.values()) if ordinal_order else None)
    ohe_scaler = OneHotEncoder(handle_unknown='ignore')

    print(numerical_features.columns.tolist())
    print(one_hot_features.columns.tolist())
    print(f'DataFrames criados sendo numericas:{numerical_features.shape[1]}, one_hot:{one_hot_features.shape[1]}, ordinal:{ordinal_features.shape[1]}')

    # # Criando o Pipeline NumScaler
    pipe_num: Tuple = pipeline_numerical(numerical_scaler=numerical_scaler, 
                                         num_cols=numerical_features)
    
    # Criando o Pipeline OneHotEncoder
    for col in one_hot_features.columns:
        one_hot_features[col] = one_hot_features.apply(lambda x: str(x[col]), axis=1) 

    pipe_one_hot: Tuple = pipeline_one_hot(one_hot_cols=one_hot_features, scaler= ohe_scaler)
    
    # Criando o Pipeline Ordinal
    pipe_ordinal: Tuple = pipeline_ordinal(ordinal_cols=ordinal_features, scaler= ord_scaler)

    for col in ignore_features.columns:
        ignore_features[col] = ignore_features.apply(lambda x: int(x[col]), axis=1) 

    pipe_sem_mexer = pipeline_select_sem_mexer(ignore_features=ignore_features)

    print(f'Pipelines criados, criando of FeatureUnion')
    
    return (FeatureUnion(
        transformer_list=[pipe for pipe in [pipe_num, pipe_one_hot, pipe_ordinal, pipe_sem_mexer] if pipe is not None],
        verbose=False
    ), 
    numerical_features.columns, 
    one_hot_features.columns, 
    ordinal_features.columns, 
    ignore_features.columns, 
    pipe_num[1].named_steps['selector_numerical'].transformers[0][2] if pipe_num is not None else None,
    pipe_one_hot[1].named_steps['selector_one_hot'].transformers[0][2] if pipe_one_hot is not None else None,
    pipe_ordinal[1].named_steps['selector_ord'].transformers[0][2] if pipe_ordinal is not None else None,
    pipe_sem_mexer[1].named_steps['selector_one_hot'].transformers[0][2] if pipe_sem_mexer is not None else None)


def create_training_pipeline(pipe_features, columns, model):
    return Pipeline([
        ('transformer_prep', pipe_features),
        ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = columns))),
        ('estimator', model)
    ])

def plot_rfe_train_scores(pipe) -> Dict:
    rfecv = pipe['estimator']

    print("Número ótimo de features: %d" % rfecv.n_features_)
    print(f"Ranking das features {rfecv.ranking_}")

    explaining = {
        'rows/index':'As rows são as features sendo excluidas recursivamente',
        'colunas':'Possuimos o valor do teste médio assim o resultado individual de cada teste, quanto mais cv mais colunas'
    }

    plt.figure(figsize=(12,6)) 
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], label="Cross-validation score")
    plt.xlabel("Número de features selecionadas")
    plt.ylabel("Cross validation score")
    plt.legend(loc="best")
    plt.show()

    display(pd.DataFrame(rfecv.cv_results_))

    transformed_features = pipe.named_steps['transformer_prep'].get_feature_names_out()
    selected_features = transformed_features[pipe.named_steps['estimator'].support_]

    print('Features que o RFECV recomenda:')
    print(selected_features)

    return explaining, selected_features

def create_columns_listing(columns_selected: list):
    print(columns_selected)
    one_hot_features = list(set([name.split('pipe_hot__filter_one_cols__')[1].split('_')[0] for name in columns_selected if name.startswith('pipe_hot__filter_one_cols__')]))
    num_cols = list(set([name.split('pipe_num__filter_num_cols__')[1] for name in columns_selected if name.startswith('pipe_num__filter_num_cols__')]))
    ord_features = list(set([name.split('pipe_ord__filter_one_cols__')[1].split('_')[0] for name in columns_selected if name.startswith('pipe_ord__filter_one_cols__')]))

    [result for result in [num_cols, one_hot_features, ord_features] if len(result) > 0]
    return list(set(num_cols + one_hot_features + ord_features))

def plot_precision_recall_and_roc(y_true, y_pred, estimator_name="Estimator"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator_name)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    pr_display.plot(ax=axes[0])
    axes[0].set_title('Precision-Recall Curve')
    axes[0].grid(True)
    
    roc_display.plot(ax=axes[1])
    axes[1].set_title('ROC Curve')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_dist(y_train, pred_proba_train, y_val, pred_proba_val, split):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,6))
    plt.subplots_adjust(left = None, right = None, top = None, bottom = None, wspace = 0.2, hspace = 0.4)

    vis = pd.DataFrame()
    vis['target'] = y_train
    vis['proba'] = pred_proba_train

    list_1 = vis[vis.target == 1].proba
    list_2 = vis[vis.target == 0].proba

    sns.distplot(list_1, kde = True, ax = axs[0], hist = True, bins = 100, color = 'blue')
    sns.distplot(list_2, kde = True, ax = axs[0], hist = True, bins = 100, color = 'red')

    axs[0].set_title('Score Train Threshold Curve')

    vis = pd.DataFrame()
    vis['target'] = y_val
    vis['proba'] = pred_proba_val

    list_1 = vis[vis.target == 1].proba
    list_2 = vis[vis.target == 0].proba

    sns.distplot(list_1, kde = True, ax = axs[1], hist = True, bins = 100, color = 'blue')
    sns.distplot(list_2, kde = True, ax = axs[1], hist = True, bins = 100, color = 'red')

    axs[1].set_title(f'Score {split} Threshold Curve')  

    plt.show(fig)
    plt.close(fig) # Ajuda a evitar plotar o gráfico 2 vezes

def get_feature_union_output_columns(df, pipeline_union):
    feature_names = []

    for name, pipe in pipeline_union.transformer_list:
        pipe.fit(df)

        if 'selector_numerical' in pipe.named_steps:
            feature_names += list(pipe.named_steps['selector_numerical'].transformers[0][2])

        elif 'selector_one_hot' in pipe.named_steps and 'OneHotEncoder' in pipe.named_steps:
            original_cols = pipe.named_steps['selector_one_hot'].transformers[0][2]
            ohe_names = pipe.named_steps['OneHotEncoder'].get_feature_names_out(original_cols)
            feature_names += list(ohe_names)

        elif 'selector_ord' in pipe.named_steps:
            feature_names += list(pipe.named_steps['selector_ord'].transformers[0][2])

        elif 'selector_one_hot' in pipe.named_steps:
            feature_names += list(pipe.named_steps['selector_one_hot'].transformers[0][2])

    return feature_names

def score_interval_evaluation(target,predicted_probabilities, X_val, y_val) -> pd.DataFrame:
    """
    Função que tabula os resultados do modelo e tabula os cortes mais ideais
    """
    # Criando o DataFrame principal
    temp = pd.DataFrame(target).rename({'status_final_model':'target'},axis=1)
    print(temp.columns)
    predicted_probabilities = [float(format(round(x,2),'.2f')) for x in predicted_probabilities]

    # Adicionando a coluna score
    temp['score'] = predicted_probabilities
    temp['score'] = temp['score'].astype(float)
    temp.sort_values(by='score',ascending=False,inplace=True) # Ordenando os valores do score

    # Aqui contamos a quantidade de casos totais tanto 0 quanto 1 para calculo da volumetria
    grouped_score = temp.groupby('score').count() # Quantidade target tanto 0 quanto 1 para cada score individualmente [0.86 tem 6 casos]
    grouped_score = grouped_score.sort_values(by='score',ascending=False)
    grouped_score['volumetria_acc'] = grouped_score['target'].cumsum() # Acumulando os casos de scores anteriores [0.86 tem 6 casos, 0.83 tem (6) + 13 casos desse score]

    # Retrabalhando os DataFrames criados
    grouped_score.reset_index(inplace=True)
    grouped_score.rename({'target':'volumetria'},axis=1,inplace=True) # Renomeando a contagem para volumetria do score individual

    # Agrupando score e calculando o somatório da target [Aqui avaliamos apenas a target 1]
    grouped_score_target = temp.groupby('score')['target'].sum().reset_index().sort_values(by='score',ascending=False)
    grouped_score_target.rename({'target':'acertos'},axis=1,inplace=True)
    grouped_score_target['acertos_acc'] = grouped_score_target['acertos'].cumsum()

    # Fazendo o merge final dos dataframes [Juntamos as contagens totais com os casos que apenas é 1] e calculando erros
    grouped_final = grouped_score.merge(grouped_score_target, on='score', how='left')
    grouped_final['erros'] = grouped_final['volumetria'] - grouped_final['acertos']
    grouped_final['erros_acc'] = grouped_final['volumetria_acc'] - grouped_final['acertos_acc']
    grouped_final.sort_values(by='score',ascending=False,inplace=True)

    # Calculando a precisão por score individual [0.99, 0.98 etc]
    prec_linha = [
        precision_score(temp['target'], np.where(temp['score'] >= value, 1, 0))
        for value in grouped_final['score'].values
    ]

    # Calculando a recall por score individual [0.95, 0.94 etc]
    recall_linha = [
        recall_score(temp['target'], np.where(temp['score'] >= value, 1, 0))
        for value in grouped_final['score'].values

    ]
    grouped_final['prec_linha'] = prec_linha
    grouped_final['recall_linha'] = recall_linha

    bins = [min(grouped_final['score']) - 0.01] + [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                                   0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                                   0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    bins.sort()
    labels = [f'({bins[i]:.2f},{bins[i+1]:.2f}]' for i in range(len(bins) - 1)]

    # Criando os intervalos do score e fazendo as agregações
    grouped_final['score_interval'] = pd.cut(grouped_final['score'], bins=bins, right=True, labels=labels)
    grouped = grouped_final.groupby('score_interval').agg({
        'volumetria': 'sum',
        'acertos': 'sum',
        'volumetria_acc': 'max',
        'acertos_acc': 'max',
        'erros': 'sum',
        'erros_acc': 'max',
        'prec_linha': 'mean',
        'recall_linha': 'mean'
    }).reset_index().sort_values(by='score_interval',ascending=False)

    # Calculando o quanto que realmente estamos impactando da base, quanto de acertos estamos tb influenciando
    grouped['percent_impact'] = grouped.apply(lambda x: (x['volumetria_acc']/(X_val.shape[0]))*100,axis=1)
    grouped['percent_error'] = grouped.apply(lambda x: (x['erros_acc']/(y_val.value_counts()[0]))*100,axis=1)
    grouped['percent_acertos'] = grouped.apply(lambda x: (x['acertos_acc']/(y_val.value_counts()[1]))*100,axis=1)

    # Calculando a precision e a recall acumulada naquele score
    grouped['prec_acc'] = grouped.apply(lambda x: x['acertos_acc']/x['volumetria_acc'] ,axis=1)
    grouped['recall_acc'] = grouped.apply(lambda x: x['acertos_acc']/(y_val.value_counts()[1]) ,axis=1)

    return grouped    

def plot_auc(y_true, predicted_score):
    """
    Plota os gráficos de ROC-auc e PR-auc
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.2)

    # Plotando os gráficos de ROC-auc e PR-auc
    RocCurveDisplay.from_predictions(y_true, predicted_score, ax=ax1, color='darkblue', linewidth=1)
    ax1.set_title('ROC-auc curve')
    PrecisionRecallDisplay.from_predictions(y_true, predicted_score, ax=ax2, color='darkblue', linewidth=1)
    ax2.set_title('PR-auc curve')

    plt.close(fig)
    return fig

def plot_confusion_matrix(y_true, y_pred, split, position_title=0.475):
    """
    Plota a matriz de confusão do modelo utilizando o ConfusionMatrixDisplay
    """
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, values_format='d')
    for i in range(len(disp.confusion_matrix)):
        for j in range(len(disp.confusion_matrix)):
            if i == j:
                disp.im_.get_array()[i, i] = np.max(disp.confusion_matrix)
                disp.text_[i, i].set_color('white')
                disp.text_[i, i].set_fontsize(16)
                disp.text_[i, i].set_fontweight('bold')

            else:
                disp.text_[i, j].set_color('red')
                disp.text_[i, j].set_fontsize(16)
                disp.text_[i, j].set_fontweight('bold')

                if i == 0 and j == 1:
                    disp.ax_.text(j, i + 0.1, 'False Positive', ha='center', va='center', color='red')

                elif i == 1 and j == 0:
                    disp.ax_.text(j, i + 0.1, 'False Negative', ha='center', va='center', color='red')

    # Add percentages based on the vertical line
    total = disp.confusion_matrix.sum(axis=0)
    for i in range(len(disp.confusion_matrix)):
        for j in range(len(disp.confusion_matrix)):
            percentage = disp.confusion_matrix[i, j] / total[j] * 100
            color = 'white' if i == j else 'red'
            disp.ax_.text(j, i + 0.2, f'{percentage:.1f}%', ha='center', va='center', color=color)

    disp.im_.set_cmap('Blues')
    plt.suptitle('Matriz de confusão | Resultados', fontsize=17, x=position_title)
    disp.ax_.set_title(f'Analisando a base de {split}', loc='center')
    disp.ax_.set_xlabel('Predicted label')  

def plot_feature_importance(model_hyperopt, model_step, preprocessing_step, RFE=True, HyperTuner=False):
    if HyperTuner:
        model_hyperopt = model_hyperopt.best_estimator_

    if RFE:
        feature_importances = model_hyperopt.named_steps[model_step].estimator_.feature_importances_
        features = model_hyperopt.named_steps[preprocessing_step].get_feature_names_out()
        support_mask = model_hyperopt.named_steps['estimator'].support_
        features = features[support_mask]
    else:
        feature_importances = model_hyperopt.named_steps[model_step].feature_importances_
        features = model_hyperopt.named_steps[preprocessing_step].get_feature_names_out()
        features = [feature.split('cols__')[1] for feature in features]

    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df['Importance'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    count_above_1_5 = (importance_df['Importance'] > 1.5).sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Importance (%)')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importance do modelo')
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=8)

    print(f'Você possui {count_above_1_5} features com importância acima de 1.5%')
    print(f'Isso representa {round(count_above_1_5/features.__len__(),2)*100}%')
    print(f'Você possui no total {features.__len__()} features')

    for bar in bars:
        color = 'darkorange' if bar.get_width() < 1.5 else 'darkblue'
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}%', va='center', ha='left', color=color, fontsize=7, fontweight='bold')
        bar.set_color(color)

    plt.show(fig)
    plt.tight_layout()
    plt.close(fig)
    return importance_df[importance_df['Importance'] <= 1.5]['Feature'].to_list()
 

def prec_safra(df, values):
    """
    Calcula a precisão, recall e Average precision em formato mes_ano
    """
    metrics = [
        (
            dataz,
            precision_score(y_true=df[df['month_year'] == dataz]['y_true'], y_pred=df[df['month_year'] == dataz]['y_pred'], pos_label=1),
            recall_score(y_true=df[df['month_year'] == dataz]['y_true'], y_pred=df[df['month_year'] == dataz]['y_pred'], pos_label=1),
            average_precision_score(y_true=df[df['month_year'] == dataz]['y_true'], y_score=df[df['month_year'] == dataz]['y_pred'])
        )
        for dataz in values
    ]
    return pd.DataFrame(metrics, columns=['data', 'precision', 'recall', 'average_precision'])

def plot_shapley(model, df: pd.DataFrame, frac_set: int = 0.7):
    """
    Plota o shapley de acordo com a base de dados
    """
    print('########################################################')
    print(f'Usando frac {frac_set} para treinamento do shap e beeswarm')
    print(f'Usando o TreeExplainer para o shapley.')
    print('########################################################')

    selected_index = df.sample(frac=frac_set, random_state=42).index.to_list()
    df_sampled = df.loc[selected_index]

    # Ensure the data matrix is the same shape as the model was trained on
    explainer = shap.TreeExplainer(
                model,
                data=df_sampled,
                feature_perturbation="interventional",
                model_output="probability"
    )
    shap_values = explainer(df_sampled)
    fig, ax = plt.subplots(layout="constrained")
    shap.plots.beeswarm(shap_values, max_display=25, order=shap_values.abs.max(0), show=False)
    plt.show(fig)
    plt.close(fig)

def transform_feature_pipe(df):
    for col in [
        'dishonored_checks',
        'expired_debts', 
        'banking_debts', 
        'commercial_debts', 
        'protests',
        'informed_restriction', 
        'form_completed', 
        'verified_restriction',
        'isSP',
        'ClassPurpose'
    ]:
        df[col] = df.apply(lambda x: int(x[col]), axis=1) 

    for col in [
        'channel',
        'landing_page_product',
        'gender',
        'regiao',
        'gender',
        'education_level', 
        'ClassMarca',
    ]:
        df[col] = df.apply(lambda x: str(x[col]), axis=1)

    return df 

def vif(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure only numeric columns are considered
    df_numeric = df.select_dtypes(include=[float, int])
    
    vif_data = pd.DataFrame()
    vif_data['features'] = df_numeric.columns
    vif_data['VIF_Value'] = [
        variance_inflation_factor(df_numeric.values, i)
        for i in range(df_numeric.shape[1])
    ]
    
    return vif_data

def training_fased(
        hyper_parameter='n_estimators',
        dict_parameters={
            'max_depth': [2,3,4,5,6,7,8,10,12],
            'class_weight': 'balanced'
        },
        limit_down_range=100,
        limit_up_range=500,
        step=100,
        model=LGBMClassifier(random_state=42),
        metric=average_precision_score,
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None
):
    result_metric=[]
    model['estimator'].set_params(**dict_parameters)

    for value in np.arange(limit_down_range, limit_up_range, step):
        model['estimator'].set_params(**{hyper_parameter:value})
        model.fit(X_train, y_train)
        
        score_val = model.predict_proba(X_val)[:, 1]
        score_train = model.predict_proba(X_train)[:, 1]
        
        for score in [{'val': score_val}, {'train': score_train}]:
            scores = list(score.values())[0]
            set = list(score.keys())[0]
            target = y_val if set == 'val' else y_train

            result_metric.append({
                'value': value,
                'metric': metric(y_true=target, y_score=scores),
                'set': set
            })
    
    return pd.DataFrame(result_metric).reset_index()

def check_leakage(
        X_train_transformed, 
        X_val_transformed,
        columns_ignore, 
        columns_without
    ):

    pipe = create_pipeline(
    df=X_train_transformed,
    columns_ignore=columns_ignore,
    columns_include_without_transformation=columns_without,
    ordinal_order=None,
    numerical_scaler=None,
    )

    pipe_prep_leakage = pipe[0]

    X_train_transformed['label'] = 'train'
    X_val_transformed['label'] = 'val'
    df_check_leakage = pd.concat([X_train_transformed, X_val_transformed],axis=0)

    # Definindo sets
    X_leakage = df_check_leakage.drop(['label'],axis=1)
    y_leakage = df_check_leakage['label']
    y_leakage = np.where(y_leakage == 'train', 1, 0)

    pipe = Pipeline([
        ('transformer_prep', pipe_prep_leakage),
        ('estimator', LGBMClassifier(random_state=42, class_weight='balanced'))
    ])

    pipe.fit(
        X_leakage,
        y_leakage
    )

    return plot_feature_importance(
        model_hyperopt=pipe,
        model_step='estimator',
        preprocessing_step='transformer_prep',
        RFE=False
    )

def calculate_gains(
        df,
        target,
        predicted_label,
        time_spent_hours
):
    df['target'] = target
    df['predicted_label'] = predicted_label
    df['time_spent_hours'] = time_spent_hours

    tempo_economizado = df[
        (df['target'] == 0) &
        (df['predicted_label'] == 0)
    ]['time_spent_hours'].sum()

    qtde_direcionado = df[
        (df['target'] == 1) &
        (df['predicted_label'] == 1)
    ].shape[0]

    tempo_falso_positivo = df[
        (df['target'] == 0) &
        (df['predicted_label'] == 1)
    ]['time_spent_hours'].sum()

    print('#'*15)
    print(f'> O modelo conseguiu identificar casos em que não avançariam para a analise de crédito poupando {tempo_economizado} horas de trabalho')
    print(f'> O modelo gerou uma quantidade de Falso Positivo gerando {tempo_falso_positivo} mais horas de trabalho')
    print(f'> O modelo conseguiu identificar casos em que avançariam para a analise de crédito em quantidade {qtde_direcionado}')
    print('#'*15)

