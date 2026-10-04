# Decisões

- Quantidades contínuas em todo o projeto; corrigir o PSO em vez de converter soluções contínuas para inteiras.
- TSP com limite e rota válida, usando CPLEX existente; sem dependência Concorde.
- Etapa 3 minimiza F4+F5; seleção entre etapas usa a função configurada.
- Parquets complementares preservam consumidores legados e identificam etapas.
- Orçamentos separados; deadline PSO opcional.
- Nenhuma avaliação experimental durante a implementação.
- Incumbentes rejeitadas pela auditoria conservam `eta` e snapshots de debug para diagnóstico; o resumo distingue existência de incumbente de solução factível aceita.
- Gaps/bounds do incumbente exato descartado não são atribuídos a uma solução heurística conservada. Eventos do exato permanecem disponíveis para inspecionar sua execução.
- Defaults e rejeições de configuração ficam em `Config`; `--unit-only` impede resolução acidental durante a validação autorizada.
