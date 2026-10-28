# Decisões de modelagem e premissas

## Dados
- Os arquivos oficiais vêm **exclusivamente** do notebook do professor (`manifesto_dados.json` com SHA-256). O código recusa arquivos alterados (`comum/dados.py::verificar_integridade`) e recusa manifestos marcados como `provisorio`.
- A instância usada no desenvolvimento foi reconstruída dos prints do notebook; ela serve apenas para testar o pipeline. Troque-a pelos arquivos oficiais e rode `python run_all.py`.

## Questão 1
- **Caixa operacional** = Σ unidades × (preço·(1 − taxa_marketplace) − custo_unitário − custo_logístico): mesma definição de “caixa base” do gerador, para a meta e a dívida serem comparáveis.
- **Demanda** com elasticidade constante: D(p) = demanda_base · (p/preço_atual)^(−elasticidade); unidades = min(estoque, D). Modelo de valor esperado (unidades fracionárias).
- **Preços permitidos**: os `niveis_preco_relativos` aplicados ao `preco_atual`, filtrados pelo **piso de margem por SKU** (margem/preço ≥ `margem_minima_percentual`). SKU sem nenhum preço permitido é **suspenso** (não vende), em vez de ser liquidado abaixo do piso → evita desconto indiscriminado.
- **Separabilidade:** o objetivo é aditivo por SKU, então o ótimo exato também sai por decomposição (Σ máx por SKU). Usamos isso como oráculo extra (`gap_vs_otimo_exato_pct`). O GA continua obrigatório e é o método que absorve restrições acopladas se o professor acrescentar alguma.
- **Subconjunto do Brute Force:** os `m` SKUs (padrão 8, mínimo exigido 6) com mais preços permitidos, desempate por sorteio com seed.
- **GA:** cromossomo = posição do preço na lista de permitidos (piso respeitado por construção); fitness = caixa; torneio k=3 (pressão seletiva moderada); cruzamento uniforme (genes independentes por SKU); mutação por gene com passo ±1 (os níveis são ordinais) ou sorteio; elitismo de 4; parada por 40 gerações sem melhoria. Configuração B (população 40, mutação 1%, elitismo 1) serve de comparação.
- **Incentivos (premissa — os dados não trazem curva de resposta):** demanda × (1 + lift_max·(1 − e^(−x/x_sat))), lift_max = min(40%, 0,15·elasticidade), x_sat = 4% da receita do SKU; o gasto é descontado do caixa; o Greedy investe lotes de orçamento/200 no maior benefício marginal por real e para quando o benefício < R$ 1,00 por R$ 1,00. Parâmetros em `comum/incentivos.py::ParamsIncentivo`.

## Questão 2
- **Choques** lognormais de média 1 (demanda, preço do concorrente relativo à base, custo logístico), i.i.d. por semana e SKU, com os sigmas do `q2_config`. Demanda = base · nível^(−ε) · ζ^ε · ξ. Validação extra com choque **persistente** do concorrente (passeio aleatório).
- **Cenários**: estimativa com ≥ 2.000 (`mc-estimativa`), calibração do θ com outros 2.000 (`calibracao`), validação com ≥ 5.000 (`validacao`): fluxos de seed distintos → independentes. As políticas comparadas usam os mesmos números aleatórios (variância reduzida na comparação).
- **DP por SKU** (estado exato semana × estoque; transição pelos 40 quantis da demanda vindos do Monte Carlo; terminal 0). O objetivo é o caixa esperado ponderado por urgência: peso da semana = 1 + θ·(prazos restantes − 1)/2. θ ∈ {0; 0,5; 1; 2} é escolhido pela maior P(pagar tudo) nos cenários de construção (empate → maior caixa).
- **Política** adaptativa (consulta a tabela da DP com o estoque observado). A trajetória ótima esperada é reconstruída seguindo a política com a venda esperada; a trajetória fixa (malha aberta) é validada para comparação.
- **Greedy**: usa preço e estoque esperado do caminho da DP e o valor futuro do estoque V(w+1, ·) da DP (um incentivo que vende hoje consome estoque que valeria mais depois). Orçamento semanal respeitado; para quando o benefício < custo.
- **Pagamentos:** cascata nas semanas 4, 8 e 12: paga se o saldo ≥ dívida; senão falha no prazo (dívida não paga permanece). “Saldo líquido” = caixa inicial + caixa de 12 semanas − dívidas; risco reportado no percentil 10 (alinhado ao limite de falha de 10% da instância) e CVaR.
- **Estresse:** o plano (calculado com as premissas originais) é reavaliado sem replanejar sob concorrente −8%/−15%, logística +15%/+30% e dívidas +20%…+100%; o ponto de ruptura (maior multiplicador de dívida com P(pagar tudo) ≥ 90%) é obtido por bisseção.

## Limitações assumidas
- Parâmetros de resposta a incentivo são premissas justificadas, não dados.
- Choques independentes entre semanas na DP; o caso persistente só é testado na validação.
- Unidades esperadas fracionárias na Q1; unidades inteiras (arredondadas) na simulação da Q2.
