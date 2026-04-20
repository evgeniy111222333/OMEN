# Аудит Реалізації `GROUNDING_MASTERPLAN.md`

Дата аудиту: 2026-04-20

## 1. Мета аудиту

Цей документ відповідає на два питання:

1. Що з `GROUNDING_MASTERPLAN.md` уже реально імплементовано в репозиторії.
2. Як grounding-підсистема працює зараз насправді в runtime, а не лише на рівні концепту.

Аудит зроблено по таких шарах:

- `omen_grounding/*`
- `omen_symbolic/execution_trace.py`
- інтеграція в `omen_scale.py`
- пов’язані тести в `tests/*`

## 2. Короткий висновок

`GROUNDING_MASTERPLAN.md` уже не є просто “майбутньою концепцією”. У репозиторії є реальний working slice такого ланцюга:

`text/bytes -> routing -> semantic text grounding -> semantic scene graph -> canonical interlingua -> symbolic compilation -> verification -> world-state writeback -> symbolic/world-graph/memory/planner/EMC integration`

Але критично важливий нюанс такий:

- цей pipeline поки не є окремим top-level orchestrator для всієї системи;
- зараз він вбудований насамперед у `omen_symbolic/execution_trace.py` через `_ObservationTraceBuilder`;
- тобто фактичний runtime grounding живе всередині trace-побудови і далі заходить у `SymbolicTaskContext`, `WorldGraph`, `EMC`, `memory` і decoder через `trace_bundle`.

Отже, статус не “не реалізовано”, а:

- backbone уже є;
- інтеграція вже серйозна;
- найслабше місце зараз не world-state/writeback, а ранні semantic layers L3-L5;
- природномовний grounding усе ще в основному heuristic-first, хоч уже і проходить повний багатошаровий pipeline.

## 3. Головний фактичний висновок

Найточніше формулювання поточного стану таке:

- `masterplan` реалізований частково, але по наскрізному шляху;
- код уже має staged grounding pipeline;
- цей pipeline уже збагачує trace, symbolic context, world graph, memory, EMC і planner state;
- але він ще не став головним learned semantic engine;
- головний “semantic burden” досі несуть евристики, trace builder, source router і downstream symbolic/world-state machinery.

Іншими словами:

- нижні та середні шари L0/L1/L6/L7/L8/L9 уже добре відчутні;
- L3/L4/L5 існують, але поки ще легкі й евристичні;
- окремого сильного learned `SemanticGroundingBackbone` поки немає.

## 4. Які модулі реально існують

### 4.1 `omen_grounding` уже містить окремий grounding stack

Реально присутні такі модулі:

- `text_semantics.py`
- `semantic_scene.py`
- `interlingua.py`
- `symbolic_compiler.py`
- `verification.py`
- `world_state_writeback.py`
- `world_state_atoms.py`
- `world_graph_records.py`
- `memory_hints.py`
- `planner_state.py`
- `emc_signals.py`
- `pipeline.py`

Тобто це вже не один helper-файл, а окремий пакет із внутрішніми контрактами.

### 4.2 Повний pipeline уже існує як функція

У `omen_grounding/pipeline.py` функція `ground_text_to_symbolic(...)` робить повний прохід:

- `ground_text_document(...)`
- `build_semantic_scene_graph(...)`
- `build_canonical_interlingua(...)`
- `compile_canonical_interlingua(...)`
- `verify_symbolic_hypotheses(...)`
- `build_grounding_world_state_writeback(...)`

Це вже пряме втілення ядра masterplan.

### 4.3 Але top-level runtime користується ним не напряму

Основний runtime не викликає `ground_text_to_symbolic(...)` як окремий canonical stage зверху.

Замість цього:

- `omen_scale.py` викликає `build_symbolic_trace_bundle(...)`;
- для observation/text path всередині `omen_symbolic/execution_trace.py` використовується `_ObservationTraceBuilder`;
- уже `_ObservationTraceBuilder.build(...)` запускає `ground_text_to_symbolic(...)`.

Це дуже важлива реальність:

- grounding stack уже живий;
- але його первинна точка входу зараз це trace-layer, а не окремий “grounding orchestrator”.

## 5. Як система працює зараз насправді

## 5.1 Реальний шлях для natural text / observation text

Для природномовного або observation input фактичний шлях такий:

1. `src` подається як UTF-8 bytes / byte tokens.
2. `omen_scale.forward()` будує `perception_world_graph`.
3. Далі формується `saliency_out`, якщо `saliency_enabled=True`.
4. `omen_scale.py` звертається до `_ast_trace_from_bytes(...)`.
5. Для natural/structured text trace builder переходить у `_ObservationTraceBuilder`.
6. `_ObservationTraceBuilder.build(...)` запускає `ground_text_to_symbolic(...)`.
7. Цей pipeline створює:
   - `GroundedTextDocument`
   - `SemanticSceneGraph`
   - `CanonicalInterlingua`
   - `SymbolicCompilationResult`
   - `GroundingVerificationReport`
   - `GroundingWorldStateWriteback`
8. Далі trace bundle насичується:
   - `grounding_facts`
   - `grounding_target_facts`
   - `grounding_hypotheses`
   - `grounding_verification_records`
   - `grounding_world_state_records`
   - `grounding_world_state_active_facts`
   - `grounding_world_state_hypothetical_facts`
   - `grounding_world_state_contradicted_facts`
   - `grounding_graph_records`
9. Уже ці об’єкти заходять у `SymbolicTaskContext`.
10. Далі вони йдуть у `WorldGraph`, `memory`, `EMC`, `planner_state`, decoder/OSF.

Тобто сьогодні grounding path реально проходить усю систему.

## 5.2 Реальний шлях у `forward()`

У `omen_scale.forward()` послідовність виглядає так:

1. `NET` або token encoder кодує `src`.
2. Будується `perception_world_graph`.
3. Із нього або з perceiver береться latent `z`.
4. За потреби рахується `saliency_out`.
5. Будується `world_graph_batch` через `_build_world_graph_batch(...)`.
   У graph потрапляють:
   - sequence edges
   - AST facts
   - trace targets
   - saliency facts
   - context / grounding records
6. Після цього `z` додатково ground-иться world graph’ом.
7. Далі:
   - memory retrieval
   - world rollout
   - epistemic gap
   - curiosity / counterfactual actions
8. Після цього робиться:
   - `_seed_symbolic_memory_facts(...)`
   - `_seed_grounding_memory_records(...)`
   - `_recall_symbolic_memory_facts(...)`
   - `_recall_grounding_memory_records(...)`
9. Далі збирається `SymbolicTaskContext`:
   - observed now
   - memory facts
   - memory grounding records
   - grounding world-state records
   - grounding verification
   - grounding hypotheses
   - saliency facts
   - net facts
   - grounding derived facts
10. Далі world graph ще раз збагачується через `_enrich_world_graph_batch(...)`.
11. Проливається symbolic reasoning batch.
12. Формується `CanonicalWorldState`, у який входить `planner_state`.
13. Decoder/OSF отримує:
   - `planner_goal`
   - `planner_symbolic_facts`

Отже, grounding реально дотягується аж до generation stack.

## 5.3 Реальний шлях у `generate()`

Генерація не відокремлена від grounding:

- `_build_generation_task_context(...)` робить той самий тип збирання context;
- бере `trace_bundle`, grounding records, saliency, net facts, memory grounding records;
- з цього формує `SymbolicTaskContext`;
- далі на кожному кроці generation може знову будувати `planner_state`;
- `planner_state` використовується для `symbolic_goal` і `symbolic_facts`.

Тобто grounding path реально живе і в generate-time, а не лише у train/forward.

## 6. Шарова карта: masterplan -> фактичний статус

## 6.1 L0: Unified Carrier

### Статус

Частково реалізовано.

### Що є

- система реально byte-first;
- `src` і `prompt` проходять як byte-level тензори;
- routing і trace будуються з байтового представлення.

### Що відповідає masterplan

- unified carrier реально є;
- canonical шлях справді йде від bytes.

### Чого ще немає

- немає окремого first-class byte-span provenance object на всіх шарах;
- `GroundingSpan` існує в типах, але реальна span-трасованість використовується слабо;
- повного “every object traceable to byte spans” ще немає.

## 6.2 L1: Typed Perception and Segmentation

### Статус

Реалізовано добре, але евристично.

### Що є

- `SourceRoutingDecision`
- `_infer_source_routing(...)`
- `modality`, `subtype`, `verification_path`
- `profile`
- routing для:
  - code
  - natural text
  - structured text
  - mixed
  - fallback

### Що реально працює

Router уже вміє:

- розрізняти `code`, `structured_observation`, `observation_text`, `text`;
- мапити їх у `modality`;
- обирати `verification_path`:
  - `ast_program_verification`
  - `mixed_hybrid_verification`
  - `structured_state_verification`
  - `log_trace_verification`
  - `config_schema_verification`
  - `table_consistency_verification`
  - `natural_language_claim_verification`
  - `scientific_claim_verification`
  - `dialogue_state_verification`
  - `fallback_verification`

### Висновок

Section 3.2 masterplan тут реально підтверджується кодом.

### Чого ще немає

- це все ще heuristic router;
- немає learned soft router з багатими parser candidates;
- немає окремих multilingual parser families.

## 6.3 L2: Structural Grounding

### Статус

Реалізовано частково.

### Для code

Сильніше реалізовано:

- AST parsing
- AST facts
- AST rules
- symbolic trace bundle
- execution-like supervision

### Для structured text

Є:

- `extract_structured_pairs(...)`
- JSON-like / key-value / record extraction
- source routing для structured inputs

### Для natural text

Є лише легкий structural slice:

- `split_text_segments(...)`
- sentence/line splitting
- counterexample flagging

### Чого немає

- повноцінного clause parser;
- dependency trees;
- discourse segmentation;
- robust table/log/config parsers як окремих сильних модулів.

## 6.4 L3: Linguistic Grounding

### Статус

Слабка часткова реалізація.

### Що є

- `normalize_symbol_text(...)`
- `tokenize_semantic_words(...)`
- multilingual normalization через `unicodedata.normalize("NFKC", ...)`
- relation marker extraction
- goal marker extraction
- negation / counterexample detection

### Що реально відбувається

Natural-language understanding сьогодні тримається в основному на:

- regex/marker patterns;
- quoted-focus extraction;
- token normalization;
- heuristic phrase slicing.

### Що masterplan тут хотів, але поки немає

- morphology
- lemma normalization на рівні мовної моделі
- POS
- dependency structure
- clause decomposition як окремий шар
- named entity candidates як окремий сильний модуль
- coreference hypotheses
- discourse relations

### Висновок

L3 існує, але поки це heuristic linguistic grounding, а не серйозний multilingual semantic parser.

## 6.5 L4: Semantic Scene Graph

### Статус

Реалізовано частково і вже корисно.

### Що є

`build_semantic_scene_graph(...)` реально будує:

- `SemanticEntity`
- `SemanticState`
- `SemanticEvent`
- `SemanticGoal`
- `SemanticClaim`
- `SemanticSceneGraph`

### Що реально в scene graph присутнє

- entities
- states
- relations/events
- goals
- claims
- confidence
- status
- source segment linkage

### Що masterplan хотів, але поки слабко або відсутнє

- time
- location
- modality
- negation як окремий семантичний об’єкт
- quantifiers
- obligations
- explanations
- role structure високої точності

### Висновок

Scene graph уже є, але він поки значно компактніший за target L4 із masterplan.

## 6.6 L5: Canonical Semantic Interlingua

### Статус

Реалізовано частково.

### Що є

`build_canonical_interlingua(...)` створює:

- canonical entities
- canonical state claims
- canonical relation claims
- canonical goal claims
- canonical keys

### Що вже добре

- є canonical key normalization;
- scene graph нормалізується в окремий інтерфейс;
- interlingua має свої metadata та окремі graph records.

### Що ще відсутнє

- справжні semantic equivalence classes;
- серйозна cross-language convergence;
- повна role inventory;
- polarity/modality/tense-aspect як багатий canonical contract;
- багатоваріантна канонізація значення.

### Висновок

Interlingua вже існує, але це light canonical layer, а не фінальна language-invariant semantic core із masterplan.

## 6.7 L6: Probabilistic Symbolic Compiler

### Статус

Реалізовано частково.

### Що є

`compile_canonical_interlingua(...)` створює:

- `CompiledSymbolicSegment`
- `CompiledSymbolicHypothesis`
- `SymbolicCompilationResult`

Гіпотези вже мають:

- `confidence`
- `status`
- `deferred`
- `conflict_tag`
- `provenance`

### Що вже добре

- є multi-hypothesis flavor;
- є відкладені гіпотези;
- є provenance;
- є segment-level structure;
- negative/counterexample context уже впливає на compiler output.

### Що ще відсутнє

- повноцінні support sets;
- explicit contradiction links між гіпотезами;
- richer epistemic graph;
- learned probabilistic compiler;
- справжній multi-interpretation survival beyond simple `deferred`.

### Висновок

L6 не “відсутній”. Він уже є. Але поки ще це lightweight symbolic compiler, а не фінальний probabilistic semantic compiler із masterplan.

## 6.8 L7: Verification and Repair

### Статус

Реалізовано частково, але реально працює.

### Що є

`verify_symbolic_hypotheses(...)` обчислює:

- `support`
- `conflict`
- `verification_status`
- `repair_action`
- `hidden_cause_candidate`

### Які repair actions реально є

- `accept_to_world_state`
- `keep_multiple_hypotheses_alive`
- `preserve_conflict_scope`
- `trigger_hidden_cause_abduction`

### Що вже добре

- contradiction не знищується мовчки;
- deferred hypotheses підтримуються;
- hidden-cause path існує;
- verification metadata і pressure вже присутні.

### Чого ще немає

- parser agreement checks як окремий реальний module family;
- world-model compatibility checks як сильний verifier;
- memory support checks як окрема verification stage;
- rich repair scheduler;
- strong learned verification layer.

### Висновок

L7 реально вже працює, але поки в основному heuristic scoring + epistemic triage.

## 6.9 L8: Persistent World State and Memory Writeback

### Статус

Сильна часткова реалізація.

### Що є

`build_grounding_world_state_writeback(...)` створює:

- `GroundingWorldStateRecord`
- `GroundingWorldStateWriteback`

із status buckets:

- `active`
- `hypothetical`
- `contradicted`

Плюс є:

- `compile_world_state_symbolic_atoms(...)`
- `compile_interlingua_graph_records(...)`
- grounding memory seeding / recall / writeback

### Що реально працює

- world-state records потрапляють у trace bundle;
- symbolic atoms компілюються окремо для active/hypothetical/contradicted;
- records пишуться в memory;
- records потрапляють у world graph;
- `SymbolicTaskContext` не втрачає buckets.

### Висновок

L8 один із найсильніших уже реалізованих шарів.

## 6.10 L9: Reasoning, Planning, EMC, Generation

### Статус

Частково реалізовано і вже інтегровано.

### EMC

`grounding_emc_features(...)` уже будує control signals:

- `grounding_uncertainty`
- `grounding_support`
- `grounding_ambiguity`
- `grounding_memory_signal`
- `grounding_recall_readiness`
- `grounding_verification_pressure`
- `grounding_abduction_pressure`
- `grounding_control_pressure`
- `grounding_world_state_branching_pressure`
- `grounding_world_state_contradiction_pressure`

### Planner

`build_planner_world_state(...)` already projects:

- active/hypothetical/contradicted records
- active/hypothetical/contradicted facts
- goal facts
- target facts
- symbolic facts
- resource symbols
- world rules
- hypothetical rules
- destructive/persistent effects
- uncertainty / branching / contradiction / hidden cause pressure

### Generation / decode

Decoder/OSF отримує:

- `planner_goal`
- `planner_symbolic_facts`

### Обмеження

- planner ще не є окремим сильним world-object planner у сенсі masterplan;
- `planner_state` зараз радше bridge/summary layer між grounding і decode/planning logic;
- повного sandbox-world planning поверх rich world objects поки немає.

## 7. Що вже реалізовано особливо добре

## 7.1 Provenance discipline

Це один із найсильніших аспектів поточного коду.

`SymbolicTaskContext` уже розділяє:

- `observed_now_facts`
- `memory_derived_facts`
- `memory_grounding_records`
- `grounding_world_state_records`
- `grounding_world_state_*_facts`
- `grounding_hypotheses`
- `grounding_verification_records`
- `saliency_derived_facts`
- `net_derived_facts`
- `grounding_derived_facts`
- `world_context_facts`
- `abduced_support_facts`

Це дуже близько до духу masterplan.

## 7.2 World graph as integration substrate

`omen_symbolic/world_graph.py` уже реально вміє тримати:

- observed
- trace_target
- saliency
- memory
- memory_grounding
- grounding
- interlingua
- grounding world-state records/facts
- verification
- hypotheses
- goal
- target
- world_context
- abduced

Тобто world graph уже є реальною mid-level integration plane.

## 7.3 Memory integration

Grounding не лише генерує локальні objects, а й:

- сіє records у memory;
- робить recall по graph terms / graph families / statuses;
- впливає на generation context;
- впливає на EMC pressure.

Це сильна ознака того, що grounding уже не “helper utility”.

## 7.4 Planner contract

`planner_state.py` дуже прямо закриває planner contract із masterplan:

- resources
- goals
- world rules
- hypothetical rules
- destructive effects
- persistent effects
- uncertainty

Тобто planner-facing projection уже існує.

## 8. Що ще не дотягує до masterplan

## 8.1 Немає сильного learned semantic backbone

`SemanticGroundingBackbone` зараз існує лише як protocol/slot.

У `omen_scale.py` є:

- `self.semantic_grounding_backbone = None`
- `set_semantic_grounding_backbone(...)`

але реального стандартного implementation в репозиторії наразі немає.

Отже:

- hook для learned semantic grounding є;
- основний production path поки fallback/heuristic.

## 8.2 L3-L5 усе ще занадто heuristic

Поточний natural-language grounding спирається на:

- marker rules
- regexes
- token normalization
- lightweight segment parsing

Немає:

- coreference
- role labeling високої якості
- multilingual morphology
- robust entity persistence
- semantic equivalence normalization між різними формулюваннями на сильному рівні

## 8.3 Pipeline живе через trace, а не як окремий top-level orchestrator

Це найважливіший архітектурний gap відносно masterplan.

Зараз:

- grounding pipeline переважно входить через `build_symbolic_trace_bundle(...)`;
- далі trace bundle несе grounding artifacts вниз.

Тобто runtime уже використовує grounding stack, але не як окремий canonical subsystem coordinator.

## 8.4 Verification ще не є повноцінним verifier stack

Verification уже є, але поки не має:

- parser agreement families
- world-model validator
- memory corroboration validator
- temporal consistency engines
- explicit cross-module repair scheduler

## 8.5 Planner ще не читає повний rich world-object layer

`planner_state` корисний і вже інтегрований, але:

- це ще projection/summarization layer;
- не видно повного planner, який працює напряму з rich semantic scene / interlingua / world-state objects як із first-class planning substrate.

## 8.6 Ontology growth майже відсутній

У masterplan це великий блок.

У поточному коді:

- canonical keys є;
- normalized symbols є;
- але немає реального ontology growth engine.

## 8.7 Mention objects / discourse / explanations / temporal layer

Masterplan хоче значно багатші semantic objects.

Зараз у коді ще не видно повної реалізації:

- mention objects
- discourse relations
- time/location graph
- explanations
- obligations
- quantifiers

## 9. Що реалізовано неочевидно, але важливо

## 9.1 Grounding уже впливає на EMC

Це не косметика.

Світові branching/contradiction signals уже піднімають:

- `grounding_verification_pressure`
- `grounding_abduction_pressure`
- `grounding_control_pressure`

Тобто grounding уже реально впливає на meta-control.

## 9.2 Grounding уже потрапляє в memory не лише як facts

Пишуться саме records:

- world-state records
- verification records
- hypotheses
- interlingua graph records

Це означає, що memory already stores epistemic grounding artifacts, а не просто flat facts.

## 9.3 Grounding уже потрапляє в world graph як typed records

Через `extra_records` і `interlingua` labels world graph отримує не тільки прості Horn facts, а й richer typed graph records.

Це дуже близько до того, що masterplan описує як graph-centered substrate.

## 10. Де masterplan уже майже буквально виконується

Нижче те, що вже дуже близьке до тексту masterplan:

- source routing з verification paths
- symbolic ingress через `SymbolicTaskContext`
- graph-centered internal integration
- interlingua layer
- symbolic compiler layer
- verification records
- active/hypothetical/contradicted world-state writeback
- grounding-aware memory
- grounding-aware EMC pressure
- planner-facing world-state projection

## 11. Де masterplan поки лише частково виконано

- multilingual semantics beyond normalization
- real linguistic grounding
- scene graph richness
- language-invariant interlingua
- true probabilistic symbolic compiler
- robust verifier families
- ontology growth
- strong planner from world-state objects

## 12. Що masterplan поки явно випереджає код

- mention-level object system
- discourse/coreference stack
- semantic equivalence classes
- full role inventory
- hidden-cause explanation stack як окремий сильний subsystem
- direct planner over rich grounded objects
- mature external integration contracts

## 13. Реальний статус по секціях masterplan

## 13.1 Section 3: “What Exists in the Current Code”

Ця секція загалом відповідає реальності.

Особливо точно підтверджуються:

- source routing already exists
- symbolic task context is rich
- world graph is right mid-level substrate
- symbolic core is not the main weakness

## 13.2 Section 4: “What Is Good Right Now”

Також загалом відповідає коду.

Сильні підтвердження:

- correct macro-architecture
- good provenance discipline
- good verification direction
- good integration points

## 13.3 Section 5: “What Is Weak Right Now”

Ця секція теж дуже влучна.

Найбільш точні пункти:

- text grounding is mostly hint extraction
- English/ASCII bias ще не подолано системно
- weak entity persistence
- weak compositional semantics
- premature collapse only partially fixed
- no fully native semantic interlingua yet

## 13.4 Section 7: Final Target Architecture

Можна сказати так:

- L0/L1/L6/L7/L8/L9 уже частково живі;
- L3/L4/L5 присутні, але ще слабкі;
- структура реально формується в бік masterplan.

## 13.5 Section 9: Contracts Between Major Modules

Це одна з найреалістичніших частин masterplan щодо поточного коду.

Найближче реалізовані контракти:

- source router
- symbolic compiler
- verification
- memory
- EMC
- planner
- generator

## 13.6 Section 17: Metrics

Це теж уже досить добре реалізовано.

У metadata реально накопичуються:

- routing metrics
- scene metrics
- interlingua metrics
- compiled hypothesis metrics
- verification metrics
- world-state metrics
- grounding quality metrics
- planner_state metrics
- world_graph metrics

## 14. Тести, які були перевірені під час аудиту

Під час цього аудиту були реально прогнані:

- `tests/test_grounding_text_semantics.py`
- `tests/test_grounding_verification.py`
- `tests/test_grounding_world_state_writeback.py`
- `tests/test_grounding_planner_state.py`
- `tests/test_symbolic_task_context_world_state.py`
- вибрані grounding/EMC тести з `tests/test_emc_gap_protocol.py`
- вибрані інтеграційні тести з `tests/test_omen_world_graph.py`
- вибрані router/observation-trace тести з `tests/test_omen_world_graph.py`

Результат під час аудиту:

- 10 grounding-oriented tests: `passed`
- 3 EMC grounding tests: `passed`
- 5 інтеграційних world-graph tests: `passed`
- 4 router / observation-trace tests: `passed`

Сумарно в аудиті підтверджено:

- grounding pipeline реально виконується;
- verification/writeback реально працюють;
- planner_state реально збирається;
- grounding metrics реально піднімаються в context і output;
- world graph реально збагачується grounding records;
- EMC реально читає grounding pressure.

## 15. Практичний висновок

Якщо говорити жорстко й без прикрас:

- `GROUNDING_MASTERPLAN.md` уже має серйозну кодову базу під собою;
- але реалізація поки не доросла до “extreme grounding stack” із документа;
- найсильніше вже реалізовані downstream epistemic layers;
- найслабше поки реалізовані ранні semantic interpretation layers.

Тобто сьогоднішня система вже не “без grounding”.

Сьогоднішня система скоріше така:

- сильний routing + trace backbone;
- робочий but lightweight semantic grounding pipeline;
- хороший symbolic / verification / world-state downstream;
- сильна integration story;
- ще недостатньо сильний front-end semantic understanding.

## 16. Найточніший one-line status

`GROUNDING_MASTERPLAN` уже реалізований як working partial stack, інтегрований через trace/world-state/context/world-graph/memory/EMC/planner path, але його semantic front-end поки ще heuristic-first і не є тим сильним learned multilingual grounding engine, який masterplan описує як фінальну форму.

## 17. Що логічно робити далі

Найвищий ROI зараз дадуть такі напрями:

1. Винести grounding pipeline з trace-only входу в окремий first-class runtime orchestrator.
2. Додати реальний learned `SemanticGroundingBackbone`.
3. Посилити L3:
   - morphology
   - clause structure
   - coreference
   - multilingual normalization beyond marker rules
4. Посилити L4/L5:
   - richer scene graph objects
   - better event-role modeling
   - stronger interlingua equivalence
5. Посилити verification:
   - parser agreement
   - memory corroboration
   - world-model compatibility
6. Доробити planner так, щоб він працював не тільки через summary/projection, а поверх richer world-state objects.

