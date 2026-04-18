# Оновлений аудит відповідності `concept.md` і кодової бази OMEN

Дата: 2026-04-18

База порівняння:

- попередній аудит у `concept_codebase_audit.md` від 2026-04-17;
- поточний стан коду в canonical runtime (`omen_scale.py`, `omen_prolog.py`, `omen_symbolic/*`, `omen_train_code.py`);
- поточний тестовий стан репозиторію.

Фактична перевірка:

- локальний перегляд `concept.md` і проблемних секцій старого аудиту;
- перевірка реалізації в коді;
- прогін `python -m pytest -q`.

Результат тестів на момент цього refresh:

- `89 passed, 2 warnings`

## 1. Короткий висновок

Відносно `concept_codebase_audit.md` стан репозиторію реально покращився.

Головне:

- пункт `3.1` / `4.1` про слабкий `SymbolicTaskContext` можна вважати закритим на рівні концептуального контракту;
- пункт `4.2` про non-code world-grounded path більше не виглядає як "не реалізовано", але все ще є лише частково завершеним;
- пункт `4.3` про внутрішню документацію більше не є повністю червоним, але ще далекий від завершення;
- пункт `3.4` про creative/intrinsic stack став сильнішим і менш декоративним, але за зрілістю все ще не дорівнює world/symbolic core;
- пункт `3.3` про benchmark proof все ще лишається частково закритим: evaluation story став кращим, але ще не є фінальним доказом non-decorative gain по всіх осях концепту.

## 2. Що саме змінилося відносно старого аудиту

| Пункт старого аудиту | Старий статус | Новий статус | Вердикт |
| --- | --- | --- | --- |
| `3.1 SymbolicTaskContext слабший за канонічний контракт` | Частково реалізовано | Реалізовано | Закрито |
| `3.2 Execution-grounded supervision сильно прив'язане до коду` | Частково реалізовано | Частково реалізовано | Покращено, але не закрито |
| `3.3 Доказ канонічності тримається переважно на telemetry/protocol tests` | Частково реалізовано | Частково реалізовано | Суттєво не закрито |
| `3.4 Creative/intrinsic stack зрілістю нижчий за core` | Частково реалізовано | Частково реалізовано | Покращено, але не закрито |
| `4.1 First-class контракт джерел symbolic context` | Не завершено | Реалізовано | Закрито |
| `4.2 Єдиний world-grounded path для не-code доменів` | Не завершено | Частково реалізовано | Переведено з open у partial |
| `4.3 Внутрішня документація не приведена до нового канону` | Не завершено | Частково реалізовано | Переведено з open у partial |

## 3. Що тепер реально виправлено

### 3.1 `SymbolicTaskContext` тепер відповідає канону значно краще

Це найбільш явно закритий борг.

Що є в коді зараз:

- у `omen_prolog.py` `SymbolicTaskContext` має окремі first-class поля:
  - `observed_now_facts`
  - `memory_derived_facts`
  - `saliency_derived_facts`
  - `net_derived_facts`
  - `world_context_facts`
  - `abduced_support_facts`
  - `world_context_summary`
- є `source_fact_records(...)` і `source_counts()`, тобто джерела факту більше не розчиняються повністю в одному flat pool;
- у `omen_scale.py` `_compose_symbolic_task_context(...)` вже збирає ці buckets окремо, а не лише через `observed_facts + metadata`;
- `OMENScale.forward()` також виносить ці counts у telemetry:
  - `sym_observed_now_facts`
  - `sym_memory_derived_facts`
  - `sym_saliency_derived_facts`
  - `sym_net_derived_facts`
  - `sym_world_context_facts`
  - `sym_abduced_support_facts`
  - `sym_world_context_summary_entries`

Висновок:

- вимога `concept.md` §7.5 і кроку 13 canonical `forward`-циклу тепер реалізована не лише номінально, а через реальний структурний контракт;
- старі пункти `3.1` і `4.1` можна вважати закритими.

### 3.2 Creative/intrinsic outputs більше не виглядають декоративними

Тут не повне закриття, але є реальний прогрес.

Що є в коді зараз:

- creative stack підключений у prover через `CreativeCycleCoordinator`;
- результати creative cycle materialize-яться назад у task context, а не висять окремим side-report;
- `omen_symbolic/creative_cycle.py` додає:
  - selected-rule support у context;
  - counterfactual novel facts у world context;
  - intrinsic goals у world context / target shaping;
  - `creative_*` summary metrics у `world_context_summary`;
- у `omen_prolog.py` creative metrics прокидаються у `last_forward_info`;
- у `omen_scale.py` creative metrics стають частиною загального `out[...]`;
- creative runtime state тепер входить у checkpoint/runtime persistence.

Що це змінює концептуально:

- creative/intrinsic layer уже не виглядає як "мертвий support stack";
- він реально впливає на symbolic episode, world context, telemetry, persistence і benchmark surface.

Але:

- за обсягом доказу, test depth і benchmark maturity цей шар усе ще слабший за NET/world graph/prover/memory core;
- отже пункт `3.4` не закритий повністю, але став суттєво слабшим як критика.

## 4. Що стало кращим, але все ще лише частково закрито

### 4.1 Non-code world-grounded path: вже не open, але ще не фінальний

Старий аудит був правий, що execution-grounded path був надто code-centric.

Що з'явилося зараз:

- у `omen_symbolic/execution_trace.py` є `_ObservationTraceBuilder`;
- `build_symbolic_trace_bundle(...)` тепер має fallback не лише для Python/code path, а й для plain-text / observation-style traces;
- для тексту будуються:
  - step transitions;
  - observed facts;
  - target facts;
  - counterexamples / negation markers;
  - relation facts між спостереженнями;
- є окремий regression test:
  - `tests/test_omen_world_graph.py::test_observation_trace_builder_supports_plain_text_sequences`

Чому це ще не "реалізовано повністю":

- observation path усе ще сильно heuristic:
  - regex-based relation extraction;
  - sentence segmentation;
  - lexical token facts;
  - shallow negation markers;
- це вже world-grounded path для non-code input, але ще не універсальний substrate для багатих не-програмних trace-доменів;
- отже `4.2` переходить із `не завершено` у `частково реалізовано`, але не більше.

### 4.2 Evaluation story покращився, але не став остаточним benchmark proof

У репозиторії зараз вже є не лише protocol tests, а й benchmark/reporting infrastructure:

- `benchmarks/benchmark_omen_scale_eval.py`
- `tests/test_benchmark_protocol.py`
- `tests/test_transfer_suite_protocol.py`
- `benchmarks/benchmark_creative_cycle.py`

Це означає:

- evaluation story уже не тримається тільки на route/contract assertions;
- є benchmark surface, corpus protocol, transfer summaries, weighted reports, transfer deltas.

Але:

- це все ще переважно protocol-scale і small-sample proof;
- у репозиторії ще немає великого, завершеного, ablation-backed empirical story, який би закривав концептуальні осі `concept.md` §27 як production-grade proof.

Висновок:

- пункт `3.3` лишається `частково реалізовано`.

### 4.3 Internal docs: top-level canonicalization є, але doc debt лишився

Що реально покращено:

- верхні docstring-и в canonical files вже приведені до англомовного канонічного framing:
  - `omen_scale.py`
  - `omen_scale_config.py`
  - `omen_perceiver.py`
  - `omen_net_tokenizer.py`
  - `omen_osf.py`
  - `omen_prolog.py`

Чому пункт ще не можна закривати:

- далі всередині цих файлів усе ще багато mojibake;
- inline comments та частина secondary docstrings залишаються змішаними, локально застарілими або пошкодженими кодуванням;
- тобто верхній narrative уже canonical, але внутрішній explanatory layer ще неоднорідний.

Висновок:

- `4.3` більше не є повністю open;
- але статус повинен бути `частково реалізовано`, а не `реалізовано`.

## 5. Що лишається частковим відносно `concept.md`

Поточні незакриті речі:

- non-code world grounding все ще heuristic і текстоцентричний, а не truly universal observation substrate;
- creative/intrinsic stack уже інтегрований, але ще не має тієї ж доказової щільності, що canonical core;
- evaluation story ще не доведений великими benchmark/ablation результатами по всіх axes;
- внутрішня документація репозиторію ще не очищена від mojibake і змішаного pre-canonical narration.

## 6. Оновлений практичний вердикт

Відносно старого аудиту найбільший прогрес такий:

1. `SymbolicTaskContext` і first-class source contract більше не є conceptual gap.
2. Non-code world path уже існує і тестується, тому ця зона більше не "не реалізована".
3. Creative/intrinsic stack уже інтегрований в operational context, persistence і metrics, тобто він значно менш декоративний.
4. Internal docs частково canonicalized, але cleanup ще не завершено.

Поточний стан репозиторію відносно `concept.md` я б описав так:

- canonical core: реалізовано;
- symbolic context contract: реалізовано;
- non-code world-grounded path: частково реалізовано;
- creative/intrinsic maturity: частково реалізовано;
- benchmark proof of non-decorative gain: частково реалізовано;
- internal documentation canonicalization: частково реалізовано.

## 7. Підсумкова матриця стану після refresh

| Зона | Поточний статус |
| --- | --- |
| Єдиний canonical runtime | Реалізовано |
| Byte-first / NET-first substrate | Реалізовано |
| Graph-primary world state | Реалізовано |
| World-graph-grounded perception and rollout | Реалізовано |
| Memory as operational substrate | Реалізовано |
| FOL-capable symbolic core | Реалізовано |
| First-class symbolic context source contract | Реалізовано |
| Execution-grounded code path | Реалізовано |
| Non-code world-grounded observation path | Частково реалізовано |
| Creative / intrinsic integration | Частково реалізовано |
| Benchmark proof / large empirical evaluation story | Частково реалізовано |
| Internal documentation canonicalization | Частково реалізовано |
