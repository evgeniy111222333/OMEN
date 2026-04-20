# Deep Scenario Modeling Report

Дата: `2026-04-19`

Режим запуску: CPU, `torch.manual_seed(0)`.

Цей звіт спеціально розділяє два шари:

- `raw readiness`: що система робить на сирому українському вводі як є.
- `canonical execution`: що відбувається після мінімальної нормалізації в той формат, який поточні модулі реально вміють формально обробляти.

Головний висновок: reasoning core уже місцями сильний, але end-to-end anchoring сирого природного вводу ще не дотягує до повністю стабільного режиму.

## Загальний підсумок

```json
{
  "overall": {
    "n_scenarios": 5,
    "raw_trace_semantic_hits": 0,
    "verdict_counts": {
      "partial": 5
    }
  },
  "scenario_verdicts": {
    "1": {
      "title": "Виведення нового правила з абстрактного тексту",
      "verdict": "partial"
    },
    "2": {
      "title": "Аналіз та виправлення синтаксичного алгоритму",
      "verdict": "partial"
    },
    "3": {
      "title": "Вирішення логічної суперечності",
      "verdict": "partial"
    },
    "4": {
      "title": "Розпізнавання та згортання патернів",
      "verdict": "partial"
    },
    "5": {
      "title": "Створення симуляційної пісочниці (\"Що якби...\")",
      "verdict": "partial"
    }
  },
  "total_measured_phase_time_ms": 401.391
}
```

## Що працює добре

- Після canonical symbolic anchoring core вміє синтезувати bridge-rules і доводити ціль дедуктивно.
- AST/trace layer добре бачить фактичну поведінку реального Python-коду і різницю між buggy та fixed execution.
- Ontology/RuleHypothesis path уже invent-ить внутрішні предикати для стискання повторюваних патернів.
- Counterfactual consequence reasoning уже здатний відрізняти досяжну підціль від логічно недосяжної.

## Де межа системи зараз

- Сирий український natural-language input майже не дає повноцінних relation/state/goal facts у symbolic trace layer.
- Rule-centric abduction поки сильніша за hidden-entity/hidden-event abduction з конкретними новими акторами або подіями.
- Human-readable naming для invented predicates поки не native; зараз це overlay над внутрішнім предикатом.
- Деструктивне planning/state transition reasoning ще не є повноцінним native шаром symbolic core.

## Деталізація по сценаріях

## Сценарій 1: Виведення нового правила з абстрактного тексту

**Мета:** Перевірити, чи може ядро синтезувати нове bridge-rule транзитивності без жорстко заданого правила.

**Raw input**

```text
Уяви вигаданий всесвіт.
Факт 1: Об'єкти типу "Зірки" генерують об'єкти типу "Планети".
Факт 2: Об'єкти типу "Планети" генерують об'єкти типу "Супутники".
Завдання: Зроби висновок. Який зв'язок між "Зірками" і "Супутниками"?
```

**Raw routing**

```json
{
  "confidence": 0.55,
  "domain": "text",
  "evidence": {
    "code_score": 0.0,
    "mixed_score": 0.0,
    "natural_text_score": 1.25,
    "observation_score": 1.25,
    "parser_agreement": 0.0,
    "second_language_score": 0.0,
    "structured_score": 0.0,
    "top_language_score": 0.0
  },
  "language": "text",
  "modality": "natural_text",
  "profile": {
    "code": 0.0,
    "mixed": 0.0,
    "natural_text": 1.0,
    "structured_text": 0.0,
    "unknown": 0.0
  },
  "subtype": "generic_text",
  "verification_path": "natural_language_claim_verification"
}
```

**Raw trace readiness**

```json
{
  "binop_events": 0,
  "counterexamples": 0,
  "observed_facts": 17,
  "present": true,
  "return_values": [],
  "target_facts": 2,
  "text_goals": 0,
  "text_negations": 0,
  "text_relations": 0,
  "text_states": 0,
  "transitions": 4
}
```

**Canonical execution summary**

```json
{
  "bridge_rules": [
    "p44(?G0,?G1) :- p33(?G0,?G2), p33(?G2,?G1)"
  ],
  "canonical_representation": "manual_symbolic_anchor",
  "goal": "p44(1,3)",
  "observed_facts": [
    "p33(1,2)",
    "p33(2,3)"
  ],
  "rule_statuses": {
    "p44(?G0,?G1) :- p33(?G0,?G2), p33(?G2,?G1)": "verified"
  }
}
```

**Фазовий таймлайн**

| Фаза | Час (мс) | Ключовий підсумок |
| --- | ---: | --- |
| symbolic_bridge_rule_synthesis | 180.294 | checked=1.0, accepted=1.0, added=1.0, verified=1.0, goal_derived_verified=1.0 |

**Внутрішні метрики**

```json
{
  "bridge_rule_count": 1,
  "cycle_stats": {
    "accepted": 1.0,
    "active": 1.0,
    "added": 1.0,
    "candidate_budget": 10.0,
    "checked": 1.0,
    "contextual_candidates": 1.0,
    "contradicted": 0.0,
    "eval_active": 0.0,
    "learning_active": 1.0,
    "loss": 0.7391087412834167,
    "mean_counterexample_error": 0.0,
    "mean_error": 0.4669415354728699,
    "mean_graph_energy": 25.918420791625977,
    "mean_relaxed_body_error": 0.2409428358078003,
    "mean_relaxed_head_error": 0.44461822509765625,
    "mean_soft_symbolic_error": 0.5017056167125702,
    "mean_symbolic_error": 0.4265367388725281,
    "mean_token_error": 0.5,
    "mean_trace_error": 0.0,
    "mean_utility": 0.5330584645271301,
    "mean_world_error": 0.5,
    "neural_candidates": 0.0,
    "policy_loss": 0.0,
    "repaired": 0.0,
    "retained": 0.0,
    "trace_candidates": 0.0,
    "verified": 1.0
  },
  "goal_derived_verified": true
}
```

**Зовнішня поведінка**

```json
{
  "core_interpretation": "Після canonical symbolic anchoring система синтезувала bridge-rule транзитивності.",
  "surface_answer": "Зірки непрямим шляхом генерують Супутники."
}
```

**Verdict:** `partial`

**Нотатки**

- Сирий український текст не дав relation-facts: observation parser поки не покриває цей лексикон.
- Raw router не розпізнав абстрактний логічний текст як claim/scientific path; він пішов у загальний natural-text path.

## Сценарій 2: Аналіз та виправлення синтаксичного алгоритму

**Мета:** Перевірити, чи бачить система причинно-наслідкову помилку в алгоритмі через AST/trace path, а не лише токени.

**Raw input**

```text
Ось код на Python для підрахунку суми парних чисел від 1 до 10:
сума = 0
для і від 1 до 10:
якщо і ділиться на 2 без остачі:
сума = сума - і
повернути сума
Очікуваний результат дорівнює 30, але код повертає -30.
Де в цьому коді помилка відносно очікуваного результату і як її виправити?
```

**Raw routing**

```json
{
  "confidence": 0.51,
  "domain": "text",
  "evidence": {
    "code_score": 0.0,
    "mixed_score": 0.0,
    "natural_text_score": 0.6,
    "observation_score": 0.5,
    "parser_agreement": 0.0,
    "second_language_score": 0.0,
    "structured_score": 1.0,
    "top_language_score": 0.0
  },
  "language": "text",
  "modality": "natural_text",
  "profile": {
    "code": 0.0,
    "mixed": 0.0,
    "natural_text": 0.25,
    "structured_text": 0.4167,
    "unknown": 0.3333
  },
  "subtype": "generic_text",
  "verification_path": "natural_language_claim_verification"
}
```

**Raw trace readiness**

```json
{
  "binop_events": 0,
  "counterexamples": 0,
  "observed_facts": 47,
  "present": true,
  "return_values": [],
  "target_facts": 2,
  "text_goals": 0,
  "text_negations": 0,
  "text_relations": 0,
  "text_states": 0,
  "transitions": 8
}
```

**Canonical execution summary**

```json
{
  "buggy_code": "def solve():\n    total = 0\n    for i in range(1, 11):\n        if i % 2 == 0:\n            total = total - i\n    return total",
  "buggy_routing": {
    "confidence": 0.99,
    "domain": "code",
    "evidence": {
      "code_score": 10.6,
      "mixed_score": 2.6,
      "natural_text_score": 0.0,
      "observation_score": 0.0,
      "parser_agreement": 0.0,
      "second_language_score": 0.0,
      "structured_score": 2.6,
      "top_language_score": 8.3
    },
    "language": "python",
    "modality": "mixed",
    "profile": {
      "code": 0.6709,
      "mixed": 0.1646,
      "natural_text": 0.0,
      "structured_text": 0.1646,
      "unknown": 0.0
    },
    "subtype": "mixed_code_structured",
    "verification_path": "mixed_hybrid_verification"
  },
  "buggy_trace": {
    "binop_events": 15,
    "counterexamples": 0,
    "observed_facts": 250,
    "present": true,
    "return_values": [
      -30
    ],
    "target_facts": 65,
    "text_goals": 0,
    "text_negations": 0,
    "text_relations": 0,
    "text_states": 0,
    "transitions": 28
  },
  "canonical_representation": "normalized_python_code",
  "fixed_code": "def solve():\n    total = 0\n    for i in range(1, 11):\n        if i % 2 == 0:\n            total = total + i\n    return total",
  "fixed_trace": {
    "binop_events": 15,
    "counterexamples": 0,
    "observed_facts": 250,
    "present": true,
    "return_values": [
      30
    ],
    "target_facts": 65,
    "text_goals": 0,
    "text_negations": 0,
    "text_relations": 0,
    "text_states": 0,
    "transitions": 28
  },
  "overlay_localization": {
    "actual_result": -30,
    "callable_found": true,
    "expected_result": 30,
    "mismatch": true,
    "observed_operator": "Sub",
    "suggested_line": "            total = total + i",
    "suggested_operator": "Add",
    "suspect_line": 5
  }
}
```

**Фазовий таймлайн**

| Фаза | Час (мс) | Ключовий підсумок |
| --- | ---: | --- |
| canonical_python_routing_buggy | 0.362 | modality=mixed, verification_path=mixed_hybrid_verification, confidence=0.99 |
| buggy_ast_trace_build | 8.098 | transitions=28, binop_events=15, return_values=[-30] |
| fixed_ast_trace_build | 3.910 | transitions=28, binop_events=15, return_values=[30] |
| overlay_ast_mismatch_localization | 0.188 | actual_result=-30, expected_result=30, suspect_line=5, observed_operator=Sub, suggested_operator=Add |

**Внутрішні метрики**

```json
{
  "buggy_return_values": [
    -30
  ],
  "fixed_return_values": [
    30
  ],
  "native_ast_ready": false,
  "native_repair_agent_present": false,
  "trace_transition_delta": 0
}
```

**Зовнішня поведінка**

```json
{
  "core_interpretation": "Core trace layer правильно бачить execution mismatch, але самостійний patch synthesis тут поки робить overlay-діагностика, а не native prover.",
  "surface_answer": "Помилка в акумулюванні суми: рядок `total = total - i` має бути `total = total + i`."
}
```

**Verdict:** `partial`

**Нотатки**

- Сирий український псевдокод не був піднятий у code/AST path без нормалізації до справжнього Python.
- Native symbolic core дає спостережуваність execution trace, але не має окремого повноцінного program-repair planner.

## Сценарій 3: Вирішення логічної суперечності

**Мета:** Перевірити, чи може система побудувати пояснення, яке знімає суперечність, не руйнуючи базове правило доступу.

**Raw input**

```text
Правило: Всі двері на космічній станції відчиняються виключно зеленою карткою.
Факт 1: Боб стоїть перед Дверима номер 5.
Факт 2: У Боба немає зеленої картки.
Факт 3: Через хвилину Двері номер 5 відчинилися.
Чому відчинилися Двері номер 5, якщо Боб не мав картки? Запропонуй логічне пояснення, що не ламає початкове правило.
```

**Raw routing**

```json
{
  "confidence": 0.63,
  "domain": "structured_observation",
  "evidence": {
    "code_score": 0.0,
    "mixed_score": 0.0,
    "natural_text_score": 1.5,
    "observation_score": 1.5,
    "parser_agreement": 0.0,
    "second_language_score": 0.0,
    "structured_score": 2.5,
    "top_language_score": 0.0
  },
  "language": "text",
  "modality": "structured_text",
  "profile": {
    "code": 0.0,
    "mixed": 0.0,
    "natural_text": 0.375,
    "structured_text": 0.625,
    "unknown": 0.0
  },
  "subtype": "config_text",
  "verification_path": "config_schema_verification"
}
```

**Raw trace readiness**

```json
{
  "binop_events": 0,
  "counterexamples": 0,
  "observed_facts": 32,
  "present": true,
  "return_values": [],
  "target_facts": 5,
  "text_goals": 0,
  "text_negations": 0,
  "text_relations": 0,
  "text_states": 0,
  "transitions": 5
}
```

**Canonical execution summary**

```json
{
  "canonical_representation": "manual_symbolic_contradiction_cluster",
  "explanation_goal": "p13(5)",
  "explanation_rules": [
    "p13(?G0) :- p12(?G0)",
    "p13(?G0) :- p10(?G1,?G0)",
    "p13(?G0) :- p12(?G0), p10(?G1,?G0)",
    "p13(?G0) :- p11(?G1), p10(?G1,?G0)",
    "p13(?G0) :- p11(?G1), p12(?G0), p10(?G1,?G0)"
  ],
  "invented_bridge_candidates": [
    {
      "clause": "p900000(?X0,?X1) :- p10(?X0,?X1)",
      "metadata": {
        "invented_predicate": 900000.0,
        "neural_guided": 0.0,
        "template": "heuristic_anchor"
      }
    },
    {
      "clause": "p12(?X0) :- p900000(?X0,?X1)",
      "metadata": {
        "invented_predicate": 900000.0,
        "neural_guided": 0.0,
        "template": "heuristic_bridge"
      }
    }
  ],
  "observed_facts": [
    "p10(1,5)",
    "p11(1)",
    "p12(5)"
  ],
  "rule_statuses": {
    "p12(?D) :- p10(?P,?D), p14(?P)": "verified",
    "p13(?G0) :- p10(?G1,?G0)": "verified",
    "p13(?G0) :- p11(?G1), p10(?G1,?G0)": "verified",
    "p13(?G0) :- p11(?G1), p12(?G0), p10(?G1,?G0)": "verified",
    "p13(?G0) :- p12(?G0)": "verified",
    "p13(?G0) :- p12(?G0), p10(?G1,?G0)": "verified"
  }
}
```

**Фазовий таймлайн**

| Фаза | Час (мс) | Ключовий підсумок |
| --- | ---: | --- |
| contextual_explanation_abduction | 97.113 | contextual_candidates=5.0, accepted=5.0, verified=5.0, explanation_rule_count=5 |
| ontology_bridge_invention | 0.112 | candidate_count=2, invented_predicates=[900000] |

**Внутрішні метрики**

```json
{
  "explanation_rule_count": 5,
  "invented_predicates": [
    900000
  ],
  "rule_preserving_specific_explanation": false,
  "strict_hidden_cause_generated": false
}
```

**Зовнішня поведінка**

```json
{
  "core_interpretation": "Rule-centric abduction працює на bridge-rules, але не на явному invent hidden actor/fact under exclusivity constraint.",
  "surface_answer": "Система підняла latent explanatory bridge навколо суперечності, але не сформулювала конкретну зовнішню причину на кшталт дистанційного відкриття чи іншого актора."
}
```

**Verdict:** `partial`

**Нотатки**

- Цей сценарій виявляє архітектурну межу: поточна абдукція набагато сильніша в rule synthesis, ніж у генерації конкретних прихованих entity-level фактів.
- Сирий текст не дав explicit negation-fact навіть попри наявність фрази про відсутність зеленої картки.

## Сценарій 4: Розпізнавання та згортання патернів

**Мета:** Перевірити, чи може система стиснути повторювані події в новий внутрішній концепт і використати його для короткого опису стану безпеки.

**Raw input**

```text
Лог системи за поточну хвилину:
Користувач Адмін увійшов успішно о 10:00. IP внутрішній. Збоїв немає.
Користувач Гість увів неправильний пароль о 10:01. IP зовнішній. Тривога.
Користувач Адмін увійшов успішно о 10:02. IP внутрішній. Збоїв немає.
Користувач Невідомий увів неправильний пароль о 10:03. IP зовнішній. Тривога.
Користувач Хакер увів неправильний пароль о 10:04. IP зовнішній. Тривога.
Опиши стан безпеки максимально стисло. Згрупуй схожі події одним новим терміном, який ти маєш придумати.
```

**Raw routing**

```json
{
  "confidence": 0.59,
  "domain": "observation_text",
  "evidence": {
    "code_score": 0.0,
    "mixed_score": 0.0,
    "natural_text_score": 1.5,
    "observation_score": 1.5,
    "parser_agreement": 0.0,
    "second_language_score": 0.0,
    "structured_score": 0.0,
    "top_language_score": 0.0
  },
  "language": "text",
  "modality": "natural_text",
  "profile": {
    "code": 0.0,
    "mixed": 0.0,
    "natural_text": 1.0,
    "structured_text": 0.0,
    "unknown": 0.0
  },
  "subtype": "generic_text",
  "verification_path": "natural_language_claim_verification"
}
```

**Raw trace readiness**

```json
{
  "binop_events": 0,
  "counterexamples": 0,
  "observed_facts": 65,
  "present": true,
  "return_values": [],
  "target_facts": 2,
  "text_goals": 0,
  "text_negations": 0,
  "text_relations": 0,
  "text_states": 0,
  "transitions": 7
}
```

**Canonical execution summary**

```json
{
  "canonical_representation": "normalized_log_records + invented_predicate",
  "invented_label_overlay": "external_failed_login_cluster",
  "invented_users": [
    "guest",
    "unknown",
    "hacker"
  ],
  "normalized_log_routing": {
    "confidence": 0.99,
    "domain": "structured_observation",
    "evidence": {
      "code_score": 0.0,
      "mixed_score": 0.0,
      "natural_text_score": 0.0,
      "observation_score": 0.0,
      "parser_agreement": 0.0,
      "second_language_score": 0.0,
      "structured_score": 5.4,
      "top_language_score": 0.0
    },
    "language": "text",
    "modality": "structured_text",
    "profile": {
      "code": 0.0,
      "mixed": 0.0,
      "natural_text": 0.0,
      "structured_text": 1.0,
      "unknown": 0.0
    },
    "subtype": "log_text",
    "verification_path": "log_trace_verification"
  },
  "normalized_log_trace": {
    "binop_events": 0,
    "counterexamples": 0,
    "observed_facts": 184,
    "present": true,
    "return_values": [],
    "target_facts": 50,
    "text_goals": 0,
    "text_negations": 0,
    "text_relations": 0,
    "text_states": 25,
    "transitions": 5
  },
  "risk_users": [
    "guest",
    "unknown",
    "hacker"
  ],
  "selected_rules": [
    "p900001(?V0) :- p20(?V0), p21(?V0)",
    "p26(?V0) :- p900001(?V0)"
  ]
}
```

**Фазовий таймлайн**

| Фаза | Час (мс) | Ключовий підсумок |
| --- | ---: | --- |
| pattern_invention_and_grouping | 104.318 | hypothesis_count=16, invented_instances=3, risk_instances=3 |

**Внутрішні метрики**

```json
{
  "compression_ratio_suspicious_facts_to_invented_instances": 3.0,
  "hypothesis_count": 16,
  "invented_instances": 3,
  "native_human_name_generation": false,
  "risk_instances": 3
}
```

**Зовнішня поведінка**

```json
{
  "core_interpretation": "OEE/RuleHypothesisSampler invent-ить внутрішній predicate добре; людська назва поки накладається зверху як label overlay.",
  "surface_answer": "Зафіксовано 2 рутинні успішні входи адміна. Виявлено 3 події типу `external_failed_login_cluster` (guest, unknown, hacker)."
}
```

**Verdict:** `partial`

**Нотатки**

- Сирий український лог у формі речень не routed як log_text; після нормалізації в key=value path спрацьовує стабільно.
- Система invent-ить внутрішній предикат, але не має окремого native name-synthesis шару для людиночитного терміна.

## Сценарій 5: Створення симуляційної пісочниці ("Що якби...")

**Мета:** Перевірити наслідкове reasoning у зміненому світі та виявити, чи може система відрізнити досяжну ціль від недосяжної.

**Raw input**

```text
Існуючі правила світу: Вогонь знищує Дерево. Вода знищує Вогонь. Камінь і Вода не взаємодіють.
Наявні ресурси: Вогонь, Дерево, Вода.
Нова умовно-фантастична ввідна: тепер припустімо, що Вогонь примножує Дерево, створюючи з нього Камінь,
а Вода миттєво знищує будь-яке Дерево. Використовуючи лише ці нові правила і доступні ресурси, як мені
створити Камінь, якщо в мене ще немає Каменя, і як після цього зберегти лише його?
```

**Raw routing**

```json
{
  "confidence": 0.774,
  "domain": "structured_observation",
  "evidence": {
    "code_score": 0.0,
    "mixed_score": 0.0,
    "natural_text_score": 1.5,
    "observation_score": 1.5,
    "parser_agreement": 0.0,
    "second_language_score": 0.0,
    "structured_score": 3.4,
    "top_language_score": 0.0
  },
  "language": "text",
  "modality": "structured_text",
  "profile": {
    "code": 0.0,
    "mixed": 0.0,
    "natural_text": 0.3061,
    "structured_text": 0.6939,
    "unknown": 0.0
  },
  "subtype": "table_text",
  "verification_path": "table_consistency_verification"
}
```

**Raw trace readiness**

```json
{
  "binop_events": 0,
  "counterexamples": 0,
  "observed_facts": 14,
  "present": true,
  "return_values": [],
  "target_facts": 2,
  "text_goals": 0,
  "text_negations": 0,
  "text_relations": 0,
  "text_states": 0,
  "transitions": 5
}
```

**Canonical execution summary**

```json
{
  "canonical_representation": "explicit_counterfactual_rule_set + state_overlay",
  "derived_facts": [
    "p30(1)",
    "p31(1)",
    "p32(1)",
    "p33(1)",
    "p34(1)"
  ],
  "observed_facts": [
    "p30(1)",
    "p31(1)",
    "p32(1)"
  ],
  "reachable_states": [
    {
      "path": [],
      "state": [
        "fire",
        "tree",
        "water"
      ]
    },
    {
      "path": [
        "fire_multiplies_tree_into_stone"
      ],
      "state": [
        "fire",
        "stone",
        "water"
      ]
    },
    {
      "path": [
        "water_destroys_tree"
      ],
      "state": [
        "fire",
        "water"
      ]
    }
  ]
}
```

**Фазовий таймлайн**

| Фаза | Час (мс) | Ключовий підсумок |
| --- | ---: | --- |
| counterfactual_rule_sandbox | 6.973 | stone_derived=1.0, tree_gone_derived=1.0, derived_fact_count=5 |
| overlay_state_reachability | 0.023 | reachable_state_count=3, only_stone_reachable=0.0, best_with_stone={'path': ['fire_multiplies_tree_into_stone'], 'state': ['fire', 'stone', 'water']} |

**Внутрішні метрики**

```json
{
  "best_with_stone": {
    "path": [
      "fire_multiplies_tree_into_stone"
    ],
    "state": [
      "fire",
      "stone",
      "water"
    ]
  },
  "only_stone_reachable": false,
  "stone_derived": true,
  "tree_gone_derived": true
}
```

**Зовнішня поведінка**

```json
{
  "core_interpretation": "Counterfactual sandbox добре виводить наслідки нових правил, але destructive planning з ресурсним споживанням тут ще wrapper-level, а не native core.",
  "surface_answer": "Камінь створюється одразу з пари `fire + tree`. Найкращий досяжний стан за цими правилами: `fire + water + stone`. Стан `only stone` недосяжний, бо нові правила не дають способу прибрати fire або water."
}
```

**Verdict:** `partial`

**Нотатки**

- Raw router не впізнав prompt як звичайний counterfactual claim bundle; path вийшов не тим, що потрібен для reasoning.
- Цей сценарій спеціально показує корисний негативний результат: 'зберегти лише камінь' логічно неможливо за даним набором правил.

## Підсумкова оцінка архітектури

Поточна архітектура вже виглядає як вузький neuro-symbolic AI runtime, а не просто текстовий класифікатор або шаблонний parser, бо вона реально має:

- typed routing;
- symbolic facts/rules;
- abduction/induction/deduction;
- ontology-like predicate invention;
- counterfactual consequence checking.

Але для рівня `працює стабільно на будь-який сирий ввід` їй ще бракує:

- сильнішого multilingual observation grounding;
- явного hidden-cause fact synthesis;
- native surface verbalization of invented concepts;
- native stateful planner з destructive transitions і reachability proofs.
