# 🚀 Desafio Creditas para a posição de Cientista de Dados PL 

💠 Como foi estruturado o desafio

* Analisando o Dataframe buscando solucionar os problemas de duplicatas e compreender features.
* Analisando os nulos e preparando estratégias de input.
* Analisando os dados e realizando a normalização dos dados.
* Construção de features inclusive utilizando informações do IBGE e do facebook/bart.
* Construção dos modelos de ML, realizando também um treinamento faseado e avaliação de métricas.
* Avaliação do resultado em comparação com o Dummy Classifier.
* Sugestões e como monitoraria o modelo.
___
💠 **O que existe dentro desse repositório?**

Apresenta todos os códigos para a construção da analise e do modelo. Cada task é apresentada por um número sequencial e realizando o story telling de todo o processo de conclusão do desafio. Cada notebook utiliza de uma classe/utils diferente para realizar corretamente o processo de modularização da analise.

💠 **Como rodar o código ?**

O código está modularizado basta possuir as bibliotecas no requirements.txt e rodar os notebooks dentro da pasta src, que possui todos os steps corretamente ordenados pelo numero no inicio sequencial.

> Para gerar a base de dados para o informed_purpose e rodar o step 4 corretamente, basta rodar o notebook neste caminho ../eda/infer_class_bart.ipynb. A run dele dura 50 minutos mas infere para todos os casos corretamente a classe do propósito do empréstimo. **Ela já está gerada por padrão**

A parte de documentation do codigo, descreve exatamente a minha linha de raciocínio entre todos os processos, até como um TODO list mesmo.

💠 **Porque o nome do model é "primeira versão"?**

O modelo foi ajustado com treinamento faseado e utilização de Feature selection, mesmo assim utilizei de um pipeline de Hypertunning para evidenciar meu conhecimento ao redor dessas habilidades. Esse modelo salvo na pasta models, não é exatamente a primeira versão do hypertunning mas sim a que foi salva.
___
💠 **Notas sobre a renderização dos notebooks**

> Devido a presença de HTML nos códigos, pode ser que via github acabe por não renderizar o notebook <br>
> Nesse caso recomendo dar um F5 na página e aguardar a renderização.

💠 **Quais seriam os processos naturais de evolução desse case?**

> Devido ao score da classe positiva não atingir 1.0 em nenhum caso da base de validação, inicialmente se nota a necessidade da inclusão de mais features ao modelo.

Minhas sugestões de análise futura incluem:
    * Utilizar de features sobre a rentabilização do empréstimo
    * Características de inadimplência do CPF
    * Características mais profundas sobre a região geográfica do CPF
    * Características do estado atual do carro, análise de sinistros e etc.