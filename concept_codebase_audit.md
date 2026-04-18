# Аудит відповідності `concept.md` і кодової бази OMEN

Дата: 2026-04-17

## 1. Висновок коротко

Поточний репозиторій вже виглядає не як "ескіз під концепт", а як реально прошитий канонічний стек OMEN.

Головне:

- канонічний runtime справді зведений до `omen_scale.py`, а публічний surface винесений у `omen.py`;
- graph-primary `z` справді реалізований через `CanonicalWorldState`, а не через один щільний latent;
- NET-first byte-level шлях, world graph, graph-conditioned `WorldRNN`, M-Core, exact symbolic memory, FOL-capable symbolic core, VeM, EMC, saliency, program anchoring, graph-centered decode і eval-time online adaptation у коді присутні та інтегровані;
- тестове покриття під це вже є: `python -m pytest -q` пройшов успішно, `77 passed`.

Але є і суттєві зауваження:

- частина концепту реалізована не як чистий канонічний контракт, а як "працюючий runtime + телеметрія + евристики";
- `SymbolicTaskContext` у коді слабший за концептуальний контракт: багато важливих джерел сигналу зливаються в `observed_facts` і `metadata`, а не мають власних полів;
- execution-grounded path добре реалізований насамперед для коду й AST/trace-задач, але не як загальний world-grounded substrate для довільних типів спостережень;
- внутрішня документація й частина docstring/коментарів відстають від нового канону і місцями прямо йому суперечать.

## 2. Що вже реалізовано і відповідає концепту

### 2.1 Канонічний surface і єдиний runtime

Статус: реалізовано.

Факти:

- `omen.py` експортує саме `OMEN = OMENScale`, `OMENConfig = OMENScaleConfig`, `build_omen(...)`.
- `omen_canonical.py` фіксує `omen_scale.OMENScale` як єдиний canonical entrypoint і маркує `omen_v2.py`, `omen_tensor_unify.py` як legacy.
- `OMENScale` примусово вмикає канонічний стек, якщо не дозволено `allow_noncanonical_ablation`.

Що це означає відносно `concept.md`:

- твердження "OMEN не має кількох рівноправних runtime" у коді вже зафіксоване не лише текстом, а й API/metadata-політикою.

Підтвердження:

- `omen.py`
- `omen_canonical.py`
- `omen_scale.py`
- `tests/test_canonical_stack_protocol.py`

### 2.2 Byte-level input і NET-first tokenizer/compressor

Статус: реалізовано.

Факти:

- `OMENScaleConfig` канонічно використовує `vocab_size = 256`.
- `NeuralEpistemicTokenizer` є штатним шляхом у `OMENScale`.
- `ByteContextEncoder` працює по сирих байтах, використовує bidirectional attention і segment-aware pooling.
- `EpistemicQuantizer` підтримує EMA, динамічний ріст словника, adaptive tau, restart dead codes, anti-collapse сигнали.
- `ByteDecoder` використовується як canonical decode-path для NET.
- NET інтегрований із symbolic core через `attach_kb(...)` і semantic feedback pairs.

Що це означає:

- концептова вимога "NET-first, byte-first, а classic token path тільки як compatibility/ablation" виконана.
- classic `TokenEncoder/TokenDecoder` у репозиторії ще існують, але це вже справді ablation/legacy path, а не canonical default.

Підтвердження:

- `omen_net_tokenizer.py`
- `omen_scale.py`
- `tests/test_net_symbolic_protocol.py`

### 2.3 Graph-primary world state і канонічний `z`

Статус: реалізовано.

Факти:

- `omen_symbolic/world_graph.py` містить `WorldGraphState`, `WorldGraphBatch`, `CanonicalWorldState`.
- `out["z"]` у `OMENScale.forward()` повертає саме `CanonicalWorldState`.
- `out["z_dense"]`, `out["world_state"]`, `out["z_graph_struct"]`, `out["z_world"]` віддаються окремо і в логіці відповідають концепту.
- `CanonicalWorldState` містить `graphs`, `neural_state`, `graph_grounded_state`, `graph_projection`, `graph_readout_state`, `grounded_state`, `symbolic_state`, `memory_state`, `program_state`, `symbolic_facts`, `target_facts`, `metadata`.

Що це означає:

- центральний концептуальний зсув від "dense latent first" до "graph/world state first" реально відбувся.

Підтвердження:

- `omen_symbolic/world_graph.py`
- `omen_scale.py`
- `tests/test_omen_world_graph.py`
- `tests/test_online_symbolic_learning_eval.py`

### 2.4 World graph як справжній grounding substrate

Статус: реалізовано.

Факти:

- `WorldGraphEncoder` будує вузли з фактів, saliency, trace-targets і context facts.
- Є канонічні типи ребер: `shared_term`, `same_pred`, `trace_step`, `counterfactual`, `saliency`, `cooccurs`.
- Є signature encoding для термів/атомів, message passing, pooled graph state.
- Trace transitions і counterexamples реально переносяться у граф.
- Граф використовується двічі: для perception-stage posterior і для reasoning/decode-stage enriched graph.

Що це означає:

- world graph у коді не декоративний. Він впливає і на posterior, і на world rollout, і на decoder-facing state.

Підтвердження:

- `omen_symbolic/world_graph.py`
- `omen_scale.py`
- `tests/test_omen_world_graph.py`
- `tests/test_generation_saliency_protocol.py`

### 2.5 Graph-native posterior і graph-centered decoder state

Статус: реалізовано.

Факти:

- якщо graph already available, `_sample_variational_latent(...)` будує posterior через `_graph_posterior_state(...)`, а не через Perceiver fallback;
- `_ground_world_state(...)` повертає graph-derived grounded state без домішки neural residual у canonical path;
- `_graph_centered_decoder_state(...)` будує readout через attention до node states;
- program state може бути вплетений у graph-centered projection.

Що це означає:

- концептова вимога "graph-native posterior whenever available" і "generation is synthesis over state" тут уже реалізована саме як runtime-поведінка.

Підтвердження:

- `omen_scale.py`
- `omen_symbolic/integration.py`
- `tests/test_omen_world_graph.py`

### 2.6 WorldRNN як graph-conditioned world transition model

Статус: реалізовано.

Факти:

- `omen_world_model.py` містить `WorldRNN`, `EpistemicGapDetector`, `CuriosityModule`.
- `WorldRNN.transition(...)` працює з `graph_context`, `target_state`, `action` або `action_probs`.
- Повертаються `causal_error`, `graph_alignment`, `state_residual`.
- `OMENScale` будує rollout через execution/world-graph-driven targets, а не просто через hidden fallback.

Що це означає:

- у репозиторії world model реально зміщено від "GRU над latent" до "graph-conditioned transition model".

Підтвердження:

- `omen_world_model.py`
- `omen_scale.py`
- `tests/test_omen_world_graph.py`

### 2.7 Epistemic gap, curiosity, memory-grounded gap

Статус: реалізовано, причому сильніше, ніж просто базовий gap detector.

Факти:

- є `EpistemicGapDetector` з exact-grad і approximate режимами;
- `CuriosityModule` використовує memory read, episodic recall і counterfactual rollouts;
- `OMENScale` має окремий `_memory_grounded_epistemic_state(...)`, де gap рахується вже після урахування memory explanation;
- це ж прокидається в generation та EMC.

Що це означає:

- реалізовано не лише базову "епістемічну прогалину", а й її memory-grounded варіант, який ближчий до концептуальної економіки системи.

Підтвердження:

- `omen_world_model.py`
- `omen_scale.py`
- `tests/test_memory_gap_protocol.py`
- `tests/test_emc_gap_protocol.py`

### 2.8 M-Core і exact symbolic memory path

Статус: реалізовано.

Факти:

- `AsyncTensorProductMemory` дає neural long-term memory з `read(...)`, `episodic_recall(...)`, buffered writes і `flush()`;
- всередині нього живе `SymbolicMemoryIndex` для exact symbolic recall;
- symbolic path підтримує `write(...)`, `recall(...)`, `recall_by_pattern(...)`, predicate hints, anchor values;
- exact symbolic writes пускаються в той самий long-term write path, а не в окремий "мертвий архів".

Що це означає:

- концептова вимога "memory is operational" і "symbolic memory is mandatory" виконана.

Підтвердження:

- `omen_scale.py`
- `omen_symbolic/memory_index.py`
- `tests/test_memory_gap_protocol.py`
- `tests/test_net_symbolic_protocol.py`

### 2.9 FOL-capable symbolic core

Статус: реалізовано.

Факти:

- у `omen_prolog.py` є `Const`, `Var`, `Compound`, `HornAtom`, `HornClause`, `Substitution`;
- є справжня уніфікація через Martelli-Montanari (`unify_mm`, `unify`, `find_all_substitutions`);
- є `KnowledgeBase` і `TensorKnowledgeBase`;
- правила мають епістемічний статус `proposed` / `verified` / `contradicted`;
- є `consolidate()`, utility-aware penalty, forward chaining, contradiction handling.

Що це означає:

- це вже не "pseudo-symbolic graph layer". Символічний core у коді реально логічний.

Підтвердження:

- `omen_prolog.py`
- `tests/test_symbolic_cycle_eval.py`
- `tests/test_creative_symbolic_engines.py`

### 2.10 VeM, continuous symbolic cycle, repair, eval-time learning

Статус: реалізовано.

Факти:

- `VerificationModule` оцінює utility rule candidates;
- `DifferentiableProver` має `continuous_hypothesis_cycle(...)`;
- є пороги acceptance/verification/contradiction;
- є repair path;
- eval-time learning окремо вмикається і тестується;
- `OMENScale.forward()` і `generate()` можуть робити online symbolic/world updates в eval режимі.

Що це означає:

- дуже важливий концептовий пункт про "без жорсткого розриву між train і eval" у коді вже не декларативний.

Підтвердження:

- `omen_prolog.py`
- `omen_scale.py`
- `tests/test_symbolic_cycle_eval.py`
- `tests/test_online_symbolic_learning_eval.py`

### 2.11 Saliency Trace

Статус: реалізовано.

Факти:

- `SaliencyTraceModule` перетворює attention і token hidden states у Horn facts;
- є фіксована рольова онтологія;
- формуються `sal_raw_facts`, `sal_semantic_facts`, `sal_expected_facts`, `sal_graph_latent`, consistency metrics;
- saliency факти входять у world graph, memory hints і symbolic task context;
- generation також повторно використовує saliency per step.

Що це означає:

- saliency у коді вже працює як bridge, а не як пояснювальна косметика.

Підтвердження:

- `omen_saliency.py`
- `omen_scale.py`
- `tests/test_generation_saliency_protocol.py`
- `tests/test_saliency_semantics.py`

### 2.12 EMC як adaptive reasoning controller

Статус: реалізовано.

Факти:

- `omen_emc.py` містить actor-critic контролер;
- є дії `Stop`, `RecallMCore`, `ForwardChainStep`, `Abduce`, `FocusIntrinsicGoal`;
- у runtime EMC реально викликається і в train, і в eval-generation режимах;
- gap/memory-pressure сигнали йдуть у stopping utility.

Що це означає:

- EMC уже є окремим робочим шаром control logic, а не лише концептуальним placeholder.

Підтвердження:

- `omen_emc.py`
- `omen_scale.py`
- `tests/test_emc_gap_protocol.py`

### 2.13 OSF, creative cycle, intrinsic goals, transfer-oriented support stack

Статус: реалізовано як support stack.

Факти:

- `omen_osf.py` та суміжні модулі реалізують `IntentEncoder -> SymbolicPlanner -> HierarchicalDecoder`, а також `WorldSimulator`, `ReflectionModule`, meta-controller;
- creative/intrinsic/ontology/analogy/counterfactual/aesthetic модулі є в `omen_symbolic/`;
- prover інтегрує `CreativeCycleCoordinator`;
- у generation/path tests перевіряється, що OSF реально може замінити decoder path;
- transfer benchmark path існує і тестується.

Що це означає:

- ці модулі вже не просто "ідеї в каталозі", а частина canonical support stack.

Підтвердження:

- `omen_osf.py` і `omen_osf_*.py`
- `omen_symbolic/creative_cycle.py` та суміжні модулі
- `tests/test_decoder_path_protocol.py`
- `tests/test_creative_symbolic_engines.py`
- `tests/test_transfer_suite_protocol.py`

### 2.14 Unified MDL / VFE / reasoning-cost objective

Статус: реалізовано.

Факти:

- `OMENScaleLoss` рахує все в єдиній валюті bits;
- явно виділені `observation_bits`, `local_complexity_bits`, `global_model_bits`, `bits_per_token`, `free_energy`;
- rule complexity і utility-adjusted penalty входять у loss;
- memory read likelihood, VFE KL terms, NET loss, VeM penalty, meta loss, reasoning cost, program losses інтегровані в один outer objective.

Що це означає:

- концептовий перехід від "набору майже несумісних loss-термів" до єдиної економіки в коді вже відбувся.

Підтвердження:

- `omen_scale.py`
- `omen_symbolic/universal_bits.py`
- `omen_prolog.py`

## 3. Що реалізовано частково або не в тій чистоті, як у `concept.md`

### 3.1 `SymbolicTaskContext` слабший за канонічний контракт документа

Статус: частково реалізовано.

Проблема:

- концепт описує `SymbolicTaskContext` як окрему операційну сцену з явними категоріями: `observed_facts`, `target_facts`, `goal`, `execution_trace`, saliency-derived facts, memory-derived facts, NET-derived concept facts, world graph summaries;
- у коді `SymbolicTaskContext` має лише `observed_facts`, `goal`, `target_facts`, `execution_trace`, `provenance`, `trigger_abduction`, `hot_dims`, `metadata`;
- memory/net/saliency/AST/decoder-surprise сигнали змішуються в один фактологічний пул і в `metadata`.

Наслідок:

- runtime працює, але концептуальна прозорість нижча, ніж заявлено в `concept.md`;
- складніше перевіряти локально, який саме підшар приніс який факт.

Де видно:

- `omen_prolog.py`
- `omen_scale.py`

### 3.2 Execution-grounded supervision сильно прив'язане до коду

Статус: частково реалізовано.

Проблема:

- execution-trace-first supervision добре зроблена для коду через AST/trace pipeline;
- для довільного тексту немає рівноцінного world-execution substrate;
- поза кодом система здебільшого спирається на token edges, AST-if-possible, saliency, NET facts і heuristic task contexts.

Наслідок:

- для code/program-like tasks відповідність концепту сильна;
- для broader "world-grounded observations" реалізація ще не рівня фінального концепту.

Де видно:

- `omen_symbolic/execution_trace.py`
- `omen_ast_multilang.py`
- `omen_scale.py`
- `benchmarks/benchmark_omen_scale_eval.py`

### 3.3 Частина "доказу канонічності" тримається на telemetry/protocol tests, а не на великому benchmark proof

Статус: частково реалізовано.

Проблема:

- у коді багато хороших protocol tests, що перевіряють наявність правильних шляхів і контрактів;
- але концепт вимагає довести, що saliency, world graph, NET, symbolic reasoning, OSF реально дають non-decorative gain;
- зараз у репозиторії це доведено сильніше на рівні route/contract/telemetry, ніж на рівні великих порівняльних benchmark results.

Наслідок:

- архітектурно стек уже зібраний;
- емпіричний "evaluation story" є, але ще не виглядає остаточно замкненим доказом по всіх осях концепту.

### 3.4 Creative/intrinsic stack існує, але його зрілість нижча, ніж у world/symbolic core

Статус: частково реалізовано.

Проблема:

- creative engines, ontology, intrinsic goals, analogy/counterfactual modules присутні і wired into prover/EMC;
- але основна доказова база репозиторію зараз сильніша для canonical core, ніж для цих "верхніх" шарів.

Наслідок:

- це вже не порожні заглушки;
- але за зрілістю це все ще support stack, а не настільки ж доведений production-grade core, як NET/world graph/prover/memory.

## 4. Що поки не виглядає реалізованим повністю

### 4.1 Чистий first-class контракт для джерел symbolic context

Статус: не завершено.

Що бракує:

- окремих полів у `SymbolicTaskContext` під memory-derived facts;
- окремих полів під saliency-derived facts;
- окремих полів під NET-derived concept facts;
- окремих полів під world graph summaries / context slices;
- явного розмежування "observed now" vs "recalled" vs "abduced support".

Це не ламає runtime, але це ще не той рівень концептуальної чистоти, який описано в документі.

### 4.2 Єдиний world-grounded path для не-code доменів

Статус: не завершено.

Що бракує:

- execution-grounded supervision поза кодом;
- більш універсального world-state ingestion для non-program traces;
- менш heuristic fallback для чисто текстових прикладів.

### 4.3 Внутрішня документація коду не приведена до нового канону

Статус: не завершено.

Проблема:

- верхні docstring і коментарі в частині файлів досі описують старі шари, старі vocabulary assumptions або pre-canonical framing;
- є mojibake/поламане кодування в багатьох docstring-ах;
- це створює локальну суперечність між `concept.md` і тим, що сам код "розповідає" про себе.

Найпомітніше:

- `omen_scale.py`
- `omen_scale_config.py`
- `omen_perceiver.py`
- частково `omen_net_tokenizer.py`, `omen_osf.py`, `omen_prolog.py`

Це не runtime-баг, але це концептуальний борг репозиторію.

## 5. Що зроблено добре відносно `concept.md`

- Канонічна вісь репозиторію вже зафіксована не лише словами, а кодом, metadata і тестами.
- `z` реально став структурованим world state.
- Symbolic layer не симулякр: є терми, змінні, уніфікація, Horn clauses, rule status, consolidation.
- Exact symbolic memory path реально існує і включений у загальний memory loop.
- World graph реально впливає на posterior, rollout і decoding.
- Eval-time online learning і world self-update реально присутні.
- Generation уже не "голий decoder": per-step world/symbolic/memory update справді відбувається.
- Objective уже близький до тієї unified economy, яку описано в концепті.

## 6. Що зроблено погано або потребує виправлення

- Внутрішні docstring/коментарі відстають від канонічного концепту і місцями йому суперечать.
- Частина канонічного контракту живе в `metadata`, а не в строгих типах/структурах.
- `SymbolicTaskContext` занадто "сплющений" порівняно з тим, як його описано в `concept.md`.
- Реалізація execution-grounded reasoning зараз набагато сильніша для code/runtime traces, ніж для загальніших world-grounded даних.
- Для advanced support stack є good path coverage, але менше доказу реальної незамінності на evaluation-рівні.

## 7. Підсумкова матриця відповідності

| Концептуальна вимога | Статус |
| --- | --- |
| Єдиний canonical runtime | Реалізовано |
| Public surface `omen.OMEN`, `omen.OMENConfig`, `build_omen(...)` | Реалізовано |
| Byte-level input | Реалізовано |
| NET-first tokenizer/compressor | Реалізовано |
| Graph-primary `z` | Реалізовано |
| `CanonicalWorldState` як primary output | Реалізовано |
| Graph-native posterior | Реалізовано |
| Graph-conditioned `WorldRNN` | Реалізовано |
| Epistemic gap + curiosity | Реалізовано |
| Long-term neural memory | Реалізовано |
| Exact symbolic memory | Реалізовано |
| FOL-capable symbolic substrate | Реалізовано |
| Rule epistemic status + consolidation | Реалізовано |
| VeM | Реалізовано |
| Execution-trace-first symbolic supervision | Реалізовано частково, сильно для code tasks |
| Continuous symbolic cycle | Реалізовано |
| Eval-time online symbolic learning | Реалізовано |
| Program anchoring | Реалізовано |
| EMC | Реалізовано |
| Saliency Trace | Реалізовано |
| OSF | Реалізовано |
| Creative/intrinsic support stack | Реалізовано частково за зрілістю |
| Unified MDL/VFE objective | Реалізовано |
| First-class rich `SymbolicTaskContext` contract | Частково реалізовано |
| Загальний world-grounded substrate поза code traces | Не завершено |
| Внутрішня документація узгоджена з новим каноном | Не завершено |

## 8. Перевірка, яку я реально виконав

Локально було перевірено:

- огляд канонічних модулів і support stack;
- звірка `concept.md` проти `omen.py`, `omen_canonical.py`, `omen_scale.py`, `omen_scale_config.py`, `omen_world_model.py`, `omen_prolog.py`, `omen_net_tokenizer.py`, `omen_saliency.py`, `omen_symbolic/*`;
- запуск тестів: `python -m pytest -q`.

Результат:

- `77 passed in 84.30s`

Найважливіші тести для цього аудиту:

- `tests/test_canonical_stack_protocol.py`
- `tests/test_omen_world_graph.py`
- `tests/test_online_symbolic_learning_eval.py`
- `tests/test_symbolic_cycle_eval.py`
- `tests/test_net_symbolic_protocol.py`
- `tests/test_generation_saliency_protocol.py`
- `tests/test_memory_gap_protocol.py`
- `tests/test_emc_gap_protocol.py`
- `tests/test_decoder_path_protocol.py`
- `tests/test_transfer_suite_protocol.py`

## 9. Практичний вердикт

Якщо оцінювати жорстко відносно `concept.md`, то репозиторій уже можна назвати канонічним OMEN runtime, а не лише "research prototype зі схожими словами".

Але до повної концептуальної чистоти ще треба дотиснути три речі:

1. зробити `SymbolicTaskContext` і споріднені контракти більш first-class, а не metadata-heavy;
2. довести execution-grounded / world-grounded story не лише для code traces;
3. синхронізувати внутрішню документацію коду з новим каноном, прибравши старі формулювання і mojibake.

У поточному стані найчесніше формулювання таке:

OMEN у цьому репозиторії вже переважно відповідає `concept.md` по ядру і runtime-інтеграції, але ще не повністю відповідає йому по чистоті внутрішніх контрактів, універсальності world-grounding і дисципліні внутрішньої документації.
