# Guia para apresentar o SiteGuard em entrevistas

## Pitch de 30 segundos

> "Treinei um YOLO11 para detetar equipamento de proteção em estaleiros e construí por cima a lógica que decide, trabalhador a trabalhador, se está conforme. Avaliei com a métrica que interessa a um responsável de segurança, e não só com mAP: dos trabalhadores sem capacete, o sistema assinala 72%, e só 4% são dados como seguros por engano. Exportei para ONNX com pré e pós-processamento escritos por mim, verifiquei que dá exatamente as mesmas deteções do PyTorch e servi-o com FastAPI numa imagem Docker sem PyTorch."

## Como complementa o Vigia

| Vigia | SiteGuard |
|---|---|
| Usa um modelo pré-treinado | **Treina** um modelo próprio |
| Tracking + analytics em vídeo | Avaliação rigorosa (split de teste intocado, ajuste em validação) |
| Streamlit | API REST + ONNX + Docker leve (deploy) |

Juntos mostram o ciclo completo: dados → treino → avaliação → otimização → serving.

## Demo (3 minutos)

1. `uvicorn api.main:app` e abre `localhost:8000`.
2. Arrasta `tests/assets/site.jpg` e mostra o trabalhador sem colete a vermelho.
3. Liga **boots** e **Strict**: os trabalhadores com botas não visíveis passam a violação. Explica a diferença entre "missing" e "unknown".
4. Abre `localhost:8000/docs` para mostrar a API documentada automaticamente.
5. Mostra a tabela de resultados do README e a linha "Missing helmet reported as OK".

## Perguntas prováveis e respostas

**Porque não chega o mAP?**
O mAP mede caixas. O utilizador quer saber se cada pessoa está protegida. Por isso associo os EPI às pessoas e meço ao nível do trabalhador: recall de violações, precisão e, sobretudo, a taxa de "sem EPI mas dado como OK", que é o erro perigoso.

**Como associas o capacete à pessoa certa?**
- **Contenção:** pelo menos 50% da caixa do capacete tem de estar dentro da caixa da pessoa.
- **Região do corpo:** o centro do capacete tem de estar nos 30% de cima da pessoa.
- **Calibração:** validei as regiões nas anotações de treino (98,3% dos capacetes usados caem lá).
- **Multidões:** a região evita dar o capacete do trabalhador da frente ao de trás. Tenho um teste unitário exatamente para esse caso.

**Porque três estados (ok / missing / unknown)?**
Se tudo o que não se vê for violação, há fadiga de alertas e as pessoas deixam de ligar. Se for conforme, escondes risco. O "não verificado" é honesto, e o modo strict deixa o cliente escolher.

**Como escolheste os limiares?**
- **Critério:** F2 no conjunto de validação, porque falhar uma violação é pior do que um falso alarme.
- **Validação:** os resultados finais vêm do conjunto de teste, que nunca foi usado para nada.
- **Ruído:** abaixo de 0,05 a curva deixou de ser monotónica com apenas 77 exemplos, por isso não desci mais. É ruído, não sinal.

**Porque inferir a 960 se treinaste a 640?**
Na análise exploratória vi que o `no_helmet` mediano ocupa 0,06% da imagem, cerca de 16 px a 640. Aumentar a resolução na inferência subiu o recall de violações de capacete de 60% para 72% com a mesma precisão, à custa de cerca de 2× mais tempo. Num sistema de segurança vale a pena.

**Que erros encontraste na tua própria avaliação?**
- **Pessoas fantasma:** a primeira versão da métrica ignorava "pessoas" detetadas que não existem (um banco de empilhador vazio). Corrigi, e a precisão do colete desceu de 0,90 para 0,85. Prefiro reportar o número verdadeiro.
- **Critério de paridade:** a paridade ONNX parecia ser de 90%, mas era o critério (IoU ≥ 0,95) que era demasiado severo para caixas de 15 px. As deteções eram idênticas.

**Porque ONNX e não PyTorch em produção?**
- **Dependências e tamanho:** sem PyTorch, o ambiente fica com ~325 MB em vez de vários GB e arranca mais depressa.
- **Velocidade:** com shapes dinâmicas e letterbox retangular é 12% mais rápido do que o PyTorch.
- **Implementação própria:** escrevi o letterbox, a descodificação da saída e o NMS por classe. O NMS por classe é necessário porque, senão, a caixa do capacete eliminava a caixa da pessoa.

**Como garantes que o ONNX dá o mesmo resultado?**
Comparo nas 40 imagens de teste: 707 contra 707 caixas, todas correspondidas, confianças iguais e desvio máximo de 1 px. Há também um teste automático para isso.

**Limitações?**
- Dataset pequeno (1000 imagens), por isso espero números piores em câmaras reais.
- Só 20 épocas do modelo nano em CPU.
- `no_goggle` inutilizável (AP 0,22).
- Não medi se o modelo funciona igualmente bem para todos (tons de pele, turbantes, hijabs). Isso teria de ser feito antes de qualquer uso real.

**O que farias a seguir?**
- Recolher 200 frames das câmaras do cliente e fazer fine-tuning.
- `yolo11s` com 100 épocas em GPU.
- Tracking em vídeo com alerta só se a violação persistir N frames.
- Quantização INT8 para edge.
- Monitorizar em produção a distribuição de confianças.

## Conceitos a rever antes

- Transfer learning e fine-tuning; porque o pré-treino em COCO ajuda (época 1 já com mAP50 0,46).
- mAP50 vs mAP50-95; precisão, recall, F1/F2; ajustar limiares em validação e não em teste.
- NMS (agnóstico vs por classe), letterbox, formato de saída do YOLO (4 + nc, 8400 âncoras a 640).
- ONNX, ONNX Runtime, shapes estáticas vs dinâmicas, quantização.
- FastAPI: lifespan (carregar o modelo uma vez), validação de parâmetros, códigos 400/413/422.
