# /commit

Gere uma mensagem de commit para as alterações atualmente em stage e faça o commit
com essa mensagem.

Regras:
- Use apenas o conteúdo já staged como fonte de verdade.
- Analise o diff staged completo antes de escrever a mensagem.
- Produza uma mensagem curta, clara e específica, seguindo boas práticas de commit.
- Prefira o formato Conventional Commits quando fizer sentido.
- Se houver uma mudança principal e uma secundária, destaque a principal no título e deixe os detalhes para o corpo.
- Não invente escopo, efeitos ou intenção que não estejam evidentes no diff.

Saída esperada:
- Uma única sugestão de mensagem de commit.
- Se necessário, inclua uma linha de corpo logo abaixo do título, com no máximo 72 caracteres por linha.
- Após gerar a mensagem, execute `git commit` com ela.
- Não inclua arquivos não staged.

Formato preferido:
- `tipo(escopo): resumo`

Tipos sugeridos:
- `feat` para nova funcionalidade
- `fix` para correção de bug
- `refactor` para refatoração sem mudança de comportamento
- `docs` para documentação
- `test` para testes
- `chore` para manutenção, scripts ou ajustes de tooling

Critérios de qualidade:
- Resumo em imperativo e no presente.
- Primeira linha com até 50 caracteres, quando possível.
- Sem ponto final no título.
- Seja específico o suficiente para identificar a alteração sem ler o diff.
