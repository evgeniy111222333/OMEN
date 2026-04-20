# Детальний Worklist По Grounding L0-L9 Відносно `concept.md`

Дата: 2026-04-20

## 1. Для чого цей документ

Цей документ відповідає на практичне питання:

- що вже реально існує в репозиторії по grounding;
- як це співвідноситься з цільовою архітектурою з `concept.md` і `GROUNDING_MASTERPLAN.md`;
- по чому саме треба працювати зараз, щоб дійти до target state.

Це не абстрактний wishlist. Це backlog, прив'язаний до поточного коду.

## 2. Нормативна ціль

Нормативний target для репозиторію задають:

- `concept.md`
- `GROUNDING_MASTERPLAN.md`

Цільовий цикл там такий:

`carrier bytes -> typed perception -> structural grounding -> linguistic grounding -> semantic scene graph -> canonical interlingua -> probabilistic symbolic compilation -> verification/repair -> world state -> reasoning/planning/generation`

Ключова ідея:

- OMEN не має бути просто heuristic engine;
- OMEN не має бути uncontrolled black-box learner;
- semantic burden має переходити в learned grounding components;
- truth burden має сидіти у symbolic / verification / world-state шарах;
- planners / EMC / generator мають читати grounded state, а не сирий текст.

## 3. На що я спирався

Нижче все зіставлено по реально наявних модулях:

- `omen_grounding/text_semantics.py`
- `omen_grounding/source_routing.py`
- `omen_grounding/semantic_scene.py`
- `omen_grounding/heuristic_backbone.py`
- `omen_grounding/semantic_context.py`
- `omen_grounding/interlingua.py`
- `omen_grounding/symbolic_compiler.py`
- `omen_grounding/verification.py`
- `omen_grounding/verifier_stack.py`
- `omen_grounding/world_state_writeback.py`
- `omen_grounding/memory_hints.py`
- `omen_grounding/emc_signals.py`
- `omen_grounding/planner_state.py`
- `omen_grounding/ontology_growth.py`
- `omen_grounding/pipeline.py`
- `omen_grounding/orchestrator.py`
- `omen_symbolic/execution_trace.py`
- `omen_scale.py`
- поточні grounding-тести в `tests/*`

## 4. Короткий статус одним абзацом

У репозиторії вже є реальний staged grounding stack, а не порожня концепція. Є bytes, routing, structural units, fallback semantic scene graph, canonical interlingua, symbolic compiler, verification, verifier stack, world-state writeback, ontology growth, planner projection, EMC signals, memory hints і наскрізна інтеграція в `omen_scale.py`.

Але головний semantic burden природної мови все ще несе heuristic path:

- document layer тримає `grounding_document_semantic_authority = 0.0`;
- default scene builder це `HeuristicFallbackSemanticBackbone`;
- тести явно очікують `scene_fallback_backbone_active = 1.0`;
- базовий verifier у `verification.py` взагалі не читає `document`, `scene`, `interlingua`;
- `GroundingSpan` зараз побудований на string offsets, а не на canonical byte offsets;
- runtime має дублювання routing logic між `omen_scale.py` і `omen_grounding/source_routing.py`.

Тобто backbone репозиторію правильний, але front-end semantic grounding ще не дотягує до концепту.

## 5. Що вже сильне і це треба зберегти

- Byte-first canonical path уже є.
- Graph-primary world-state path уже є.
- `SymbolicTaskContext` уже вміє нести grounding artifacts далі вниз.
- Є явне розведення `active / hypothetical / contradicted`.
- Є deferred hypotheses замість раннього жорсткого колапсу.
- Є integration story з memory, verifier stack, planner, EMC, world graph, generation.
- Є replaceable hook для `SemanticGroundingBackbone`.
- Є окремий пакет `omen_grounding`, а не один helper-файл.
- Є серйозна кількість metadata/metrics.
- Є тести на pipeline, verifier stack, world-state writeback, planner state, context layer, ontology growth, memory recall.

## 6. Layer-by-Layer Карта Поточного Стану

| Layer | Target | Що вже є | Статус | Головний gap |
| --- | --- | --- | --- | --- |
| `L0` | unified carrier, byte provenance, traceability | byte-first runtime, `GroundingSpan` | partial | spans зараз символьні, не byte-native |
| `L1` | soft typed perception, parser candidates, ambiguity | `omen_scale._infer_source_routing`, `omen_grounding/source_routing.py`, script/profile/parser_candidates | partial-good | два routing paths, слабка калібровка і слабка multilingual ID |
| `L2` | structural objects by subtype | structured pairs, clause-like units, table/log/json/speaker/citation units, AST path for code | partial | downstream слабо споживає structural layer як primary evidence |
| `L3` | serious multilingual linguistic grounding | normalization, token hints, marker rules, lightweight context objects | weak | немає morphology, lemma, POS, dependency, real coreference |
| `L4` | semantic scene graph with event frames and roles | entities, states, events, goals, claims, mentions, discourse, temporal, explanations | partial | це ще fallback scene graph, не full semantic event-frame graph |
| `L5` | language-invariant canonical interlingua | canonical entities, states, relations, goals, modifiers | partial | слабка semantic equivalence, слабка generic/instance/epistemic normalization |
| `L6` | probabilistic symbolic compiler | compiled segments, hypotheses, deferred flags, conflict tags | partial-good | мало evidence bundles, support sets, alternative parses |
| `L7` | verification and repair families | base verification + verifier stack + repair actions | partial-good | verifier families ще heuristic/lightweight, база не використовує upstream context |
| `L8` | persistent world-state + exact memory writeback | world-state records, status buckets, memory hints, ontology growth | good-partial | provenance ще не повністю carrier-grade, naming/revision policy слабкі |
| `L9` | grounded reasoning/planning/EMC/generation | planner projection, EMC grounding signals, trace/world-state integration in runtime | partial-good | consumers ще читають projection/summaries, не full rich grounded object layer |

## 7. Найважливіші Поточні Архітектурні Розриви

### 7.1 Semantic front-end все ще heuristic-first

Симптоми:

- `grounding_document_semantic_authority` тримається на `0.0`;
- default backbone це `HeuristicFallbackSemanticBackbone`;
- scene/interlingua/compiled path живе, але сенс туди вноситься переважно regex/marker rules.

Наслідок:

- pipeline уже є, але головна semantic power ще не там, де її хоче концепт.

### 7.2 Є routing split-brain

Зараз одночасно існують:

- `omen_scale.py::_infer_source_routing(...)`
- `omen_grounding/source_routing.py::infer_source_profile(...)`

Це означає:

- є ризик роз'їзду `modality/subtype/verification_path`;
- є дублювання правил;
- частина runtime читає один routing contract, а grounding document layer уже читає інший.

### 7.3 L2 structural layer існує, але ще не став сильним upstream evidence layer

`text_semantics.py` уже вміє будувати:

- `clause`
- `speaker_turn`
- `citation_region`
- `table_row`
- `log_entry`
- `section_header`
- `key_value_record`
- `json_record`

Але далі система переважно все одно переходить у fallback semantic extraction, а не у subtype-specific structural compilers.

### 7.4 L3-L5 ще не несуть multilingual invariance

Поточний stack уміє:

- normalization;
- script-aware routing;
- alias aggregation;
- simple coreference heuristic через last salient entity;
- context markers;
- relation modifiers.

Але він ще не вміє по-справжньому:

- lemma-level equivalence;
- morphology-sensitive grounding;
- robust paraphrase invariance;
- generic vs instance distinction;
- epistemic stance normalization;
- role labeling на сильному рівні.

### 7.5 Базова verification layer ще занадто thin

Показовий факт:

- `verification.py` приймає `document`, `interlingua`, `scene`, але відразу їх відкидає через `del document, interlingua, scene`.

Це означає:

- verification зараз рахує переважно score-функції по hypotheses;
- upstream scene/interlingua context ще не став справжньою частиною verify logic.

### 7.6 Planner і generation ще не читають full rich grounding object layer

`planner_state.py` вже корисний і інтегрований, але це все ще projection layer:

- `resources`
- `operators`
- `constraints`
- `alternative_worlds`

Потрібен наступний крок:

- planner має читати richer grounded objects і verifier-family outputs, а не лише summary/projection.

### 7.7 Немає повноцінного data/eval flywheel для learned grounding

Щоб дійти до target state, недостатньо просто дописати ще 200 regexes.

Потрібно:

- gold data;
- cross-lingual equivalence data;
- contradiction/repair datasets;
- shadow evaluation;
- learned backbone training loop;
- safe online update regime.

## 8. По Чому Працювати Спочатку

Якщо пріоритезувати тверезо, то зараз highest ROI дають саме ці напрями:

1. Зробити `run_grounding_orchestrator(...)` канонічною top-level grounding stage у runtime, а не побічним trace-consumer path.
2. Прибрати split між `SourceRoutingDecision` і `GroundingSourceProfile`.
3. Перевести provenance на byte-native spans або dual char/byte spans.
4. Посилити L2 structural layer як реальний primary evidence source для structured inputs.
5. Посилити L3 multilingual linguistic grounding.
6. Перебудувати L4 на event-frame-first scene graph.
7. Нормалізувати L5 interlingua до справді language-invariant contract.
8. Розширити L6 на multiple interpretations / evidence bundles / support sets.
9. Перетворити L7 на повноцінний verifier-family stack.
10. Зробити planner/EMC/generation stricter consumers of grounded state.
11. Створити data/eval flywheel для learned grounding backbone.
12. Підключити безпечний цикл online symbolic learning і controlled self-improvement так, як вимагає `concept.md`.

## 9. Детальний Worklist

Нижче вже не “ідеї”, а конкретні задачі.

### A. Архітектура І Контракти

- `G-001 [P0]` Зробити `run_grounding_orchestrator(...)` first-class runtime stage у `omen_scale.py`.
  Файли: `omen_scale.py`, `omen_grounding/orchestrator.py`, `omen_grounding/pipeline.py`, `omen_symbolic/execution_trace.py`.
  Done when: `forward()` і `generate()` отримують один canonical grounding result object, а trace builder стає споживачем grounding, а не точкою координації.

- `G-002 [P0]` Уніфікувати routing contract.
  Файли: `omen_scale.py`, `omen_grounding/source_routing.py`, `omen_grounding/types.py`.
  Done when: у runtime залишається один canonical source profile object з `modality/subtype/verification_path/profile/script_profile/parser_candidates/ambiguity`.

- `G-003 [P0]` Ввести один versioned grounding result schema.
  Файли: `omen_grounding/types.py`, `omen_grounding/pipeline.py`, `omen_grounding/orchestrator.py`, `omen_scale.py`.
  Done when: grounding artifacts мають стабільний contract version і не зшиваються ad hoc через десятки metadata keys.

- `G-004 [P0]` Явно розвести canonical path і fallback path.
  Файли: `omen_grounding/semantic_scene.py`, `omen_grounding/backbone.py`, `omen_grounding/heuristic_backbone.py`.
  Done when: runtime завжди знає, де працює learned backbone, а де fallback; fallback більше не маскується під primary semantic engine.

- `G-005 [P0]` Ввести layer-ownership matrix.
  Файли: новий технічний doc + коментарі в `omen_grounding/*`.
  Done when: чітко зафіксовано, який шар має право робити routing, syntax, semantics, verification, writeback, planning projection.

- `G-006 [P0]` Перенести grounding metadata з “великий плоский dict” у typed summaries.
  Файли: `omen_grounding/*`, `omen_scale.py`.
  Done when: критичні control signals і quality metrics мають typed holders, а не лише безрозмірну плоску мапу.

- `G-007 [P0]` Зробити connectors/source loaders explicit citizens of L1.
  Файли: `omen.py`, `omen_scale.py`, майбутні input loaders.
  Done when: зовнішні інтеграції передають `source metadata`, а не bypass-ять grounding stack.

- `G-008 [P0]` Прибрати приховану залежність між trace metadata і grounding orchestration.
  Файли: `omen_symbolic/execution_trace.py`, `omen_scale.py`.
  Done when: grounding може існувати самостійно і trace bundle лише підключається до нього.

- `G-009 [P1]` Додати canonical config knobs для backbone/router/verifier families.
  Файли: `omen_scale_config.py`, `omen_scale.py`.
  Done when: можна керовано перемикати fallback, hybrid і learned режим без локальних хаків.

- `G-010 [P1]` Підготувати migration plan “heuristics -> learned backbone”.
  Файли: окремий design doc + `omen_grounding/backbone.py`.
  Done when: є поетапний план заміни частин fallback stack без руйнування downstream contracts.

- `G-011 [P1]` Ввести shadow-mode execution для learned grounding.
  Файли: `omen_scale.py`, `omen_grounding/pipeline.py`.
  Done when: learned backbone можна запускати паралельно з fallback і порівнювати на тих самих inputs без руйнування runtime.

- `G-012 [P1]` Логувати grounding outcomes як training/eval artifacts.
  Файли: `omen_scale.py`, `omen_grounding/orchestrator.py`, benchmark tooling.
  Done when: accepted/deferred/conflicted cases можна збирати у replay/eval corpora.

### B. `L0` Unified Carrier

- `G-013 [P0]` Перевести `GroundingSpan` на byte offsets або на dual representation `char_span + byte_span`.
  Файли: `omen_grounding/types.py`, `omen_grounding/text_semantics.py`, всі модулі, що несуть `source_span`.
  Done when: кожен об'єкт можна відтрасувати до сирих UTF-8 bytes, як вимагає концепт.

- `G-014 [P0]` Додати `source_id`, `document_id`, `episode_id` у grounding records.
  Файли: `interlingua_types.py`, `symbolic_compiler.py`, `verification.py`, `verifier_stack.py`, `world_state_writeback.py`, `ontology_growth.py`.
  Done when: world-state і memory writeback мають carrier-grade identity, а не лише segment index.

- `G-015 [P1]` Зберігати normalization map від raw text до normalized text.
  Файли: `text_semantics.py`.
  Done when: normalization не ламає provenance і можна точно реконструювати fragment mapping.

- `G-016 [P1]` Додати encoding/line-ending metadata.
  Файли: `omen_scale.py`, `omen_grounding/orchestrator.py`.
  Done when: carrier layer зберігає достатньо інформації для зовнішньої відтворюваності й audit trail.

- `G-017 [P1]` Ввести source-attachment contract для multi-document episodes.
  Файли: `omen_scale.py`, `omen_grounding/types.py`.
  Done when: grounding artifacts можуть належати до кількох source documents в одному episode.

- `G-018 [P1]` Додати span round-trip tests.
  Файли: нові тести в `tests/test_grounding_text_semantics.py`.
  Done when: кожен span стабільно мапиться назад на оригінальні bytes.

### C. `L1` Typed Perception and Segmentation

- `G-019 [P0]` Зробити `GroundingSourceProfile` canonical routing object по всьому runtime.
  Файли: `omen_scale.py`, `omen_grounding/source_routing.py`, `omen_grounding/types.py`.
  Done when: зникає паралельне життя `SourceRoutingDecision`.

- `G-020 [P0]` Підняти `parser_candidates` у runtime/task_context/world_graph metadata.
  Файли: `text_semantics.py`, `execution_trace.py`, `omen_scale.py`, `omen_symbolic/world_graph.py`.
  Done when: parser family candidates реально використовуються нижчими шарами.

- `G-021 [P0]` Додати segment-level ambiguity handling як first-class signal.
  Файли: `source_routing.py`, `text_semantics.py`, `emc_signals.py`.
  Done when: `mixed/unknown/ambiguous` сегменти не колапсують занадто рано.

- `G-022 [P1]` Посилити language identification beyond script heuristics.
  Файли: `source_routing.py`.
  Done when: мова не визначається лише marker-ами і crude Latin/Cyrillic counting.

- `G-023 [P1]` Додати per-segment language/script profile.
  Файли: `text_semantics.py`, `types.py`.
  Done when: mixed documents мають реальні segment-local language decisions.

- `G-024 [P1]` Додати subtype-specific parser family registry.
  Файли: `source_routing.py`, новий registry module.
  Done when: `scientific_text`, `dialogue_text`, `log_text`, `table_text`, `mixed_code_text` мають формалізовані parser stacks.

- `G-025 [P1]` Додати schema candidates для structured text.
  Файли: `source_routing.py`, `text_semantics.py`.
  Done when: `json/config/table/log` inputs несуть не лише subtype, а й кандидати схем/парсерів.

- `G-026 [P1]` Калібрувати routing confidence.
  Файли: `source_routing.py`, benchmark tooling.
  Done when: `confidence` і `ambiguity` проходять calibration benchmarks, а не просто hand-tuned formula.

- `G-027 [P1]` Додати mixed-content segmentation.
  Файли: `text_semantics.py`, `source_routing.py`.
  Done when: код-блоки, prose-блоки, таблиці, логи та діалогові turns в одному документі виділяються окремими segment families.

- `G-028 [P1]` Створити routing benchmark suite.
  Файли: `benchmarks/*`, `tests/*`.
  Done when: є accuracy/calibration/ambiguity benchmarks по L1.

### D. `L2` Structural Grounding

- `G-029 [P0]` Перетворити structural units з “metadata side-output” на real upstream evidence.
  Файли: `text_semantics.py`, `semantic_scene.py`, `interlingua.py`, `symbolic_compiler.py`.
  Done when: `clause/log/table/json/speaker/citation` objects реально впливають на downstream compilation.

- `G-030 [P0]` Дати structured inputs primary deterministic parsers.
  Файли: `text_semantics.py`, нові subtype parsers.
  Done when: `json`, `kv`, `table`, `log`, `config` не проходять через natural-language heuristic fallback як primary route.

- `G-031 [P1]` Посилити clause extraction.
  Файли: `text_semantics.py`.
  Done when: clause units будуються не лише split-ом по marker words і punctuation.

- `G-032 [P1]` Додати paragraph/document role units.
  Файли: `text_semantics.py`.
  Done when: з'являються `heading/body/evidence/example/exception/instruction` style structural roles.

- `G-033 [P1]` Додати parser disagreement objects.
  Файли: `text_semantics.py`, `source_routing.py`, `orchestrator.py`.
  Done when: L2 може виразити “у нас два структурні прочитання” замість раннього hard choice.

- `G-034 [P1]` Додати table schema inference і column typing.
  Файли: новий parser module + `text_semantics.py`.
  Done when: таблиці дають typed columns, row identity, schema confidence.

- `G-035 [P1]` Додати log normalization.
  Файли: новий log parser + `text_semantics.py`.
  Done when: логи виносять `timestamp`, `level`, `service`, `request_id`, `message`, `stack_fragment` як typed fields.

- `G-036 [P1]` Додати config/schema validators.
  Файли: нові structured validators + `source_routing.py`.
  Done when: `config_text` має реальні schema checks, а не лише key-value extraction.

- `G-037 [P1]` Додати structured-to-interlingua bridges.
  Файли: `interlingua.py`, `symbolic_compiler.py`.
  Done when: structural records мають власний compilation path у канонічні claims.

- `G-038 [P2]` Розширити AST/code structural alignment на mixed documents.
  Файли: `omen_scale.py`, `execution_trace.py`, `text_semantics.py`.
  Done when: code/comments/docstrings/notebook-like inputs мають спільний structural representation.

### E. `L3` Linguistic Grounding

- `G-039 [P0]` Ввести token objects як реальний contract, а не тільки normalized token strings.
  Файли: `types.py`, `text_semantics.py`.
  Done when: токени несуть surface, lemma, morphology, POS, offsets, language.

- `G-040 [P0]` Додати morphology для Ukrainian і mixed-language inputs.
  Файли: новий linguistic layer module.
  Done when: відмінки, число, рід, дієслівні форми і частка заперечення перестають губитися на поверхні.

- `G-041 [P0]` Додати lemma normalization.
  Файли: новий linguistic layer module, `interlingua.py`.
  Done when: різні словоформи стабільно сходяться до однієї semantic base.

- `G-042 [P0]` Додати POS layer.
  Файли: новий linguistic layer module.
  Done when: event/state/goal/entity candidates спираються не лише на marker words.

- `G-043 [P0]` Додати dependency structure.
  Файли: новий linguistic layer module, `semantic_scene.py`.
  Done when: subject/object/role hypotheses не будуються лише зі surface order або regex markers.

- `G-044 [P0]` Додати clause decomposition як окремий L3 object layer.
  Файли: новий linguistic layer module.
  Done when: clause boundaries, subordinate clauses, conditions, explanations, temporals мають окремі objects.

- `G-045 [P1]` Додати mention detector.
  Файли: новий linguistic layer module, `semantic_context.py`.
  Done when: mention candidates не залежать тільки від exact substring match по canonical entity name.

- `G-046 [P1]` Додати real coreference resolver.
  Файли: `semantic_context.py`, новий linguistic layer module.
  Done when: coreference більше не зводиться до `last_salient_entity_id`.

- `G-047 [P1]` Додати negation scope detection.
  Файли: новий linguistic layer module, `heuristic_backbone.py`, `interlingua.py`.
  Done when: заперечення визначається як scope, а не просто marker presence у segment.

- `G-048 [P1]` Додати quantifier detection.
  Файли: новий linguistic layer module, `scene_types.py`, `interlingua_types.py`.
  Done when: `all/some/none/generic` distinctions зберігаються до interlingua.

- `G-049 [P1]` Додати modality inventory.
  Файли: `heuristic_backbone.py`, новий linguistic layer module, `interlingua.py`.
  Done when: `must/should/can/may/obligation/permission/goal/desire` не змішуються.

- `G-050 [P1]` Додати discourse parsing beyond marker list.
  Файли: `semantic_context.py`.
  Done when: discourse relations виводяться не тільки по кількох hard-coded marker words.

- `G-051 [P1]` Додати sentence-role classification.
  Файли: новий linguistic layer module.
  Done when: система розрізняє claim, evidence, instruction, exception, report, question, goal.

- `G-052 [P1]` Додати multilingual equivalence regression suite.
  Файли: benchmarks/tests.
  Done when: одна і та сама семантика у різних формулюваннях реально порівнюється на L3/L4/L5.

### F. `L4` Semantic Scene Graph

- `G-053 [P0]` Перебудувати scene graph навколо event frames.
  Файли: `scene_types.py`, `heuristic_backbone.py`, `semantic_scene.py`.
  Done when: relation triples більше не є implicit substitute для повноцінних event frames.

- `G-054 [P0]` Додати first-class roles.
  Файли: `scene_types.py`, `heuristic_backbone.py`, `interlingua.py`.
  Done when: `agent/patient/instrument/location/beneficiary/source/target` живуть окремими objects/fields.

- `G-055 [P0]` Додати first-class times і locations.
  Файли: `scene_types.py`, `semantic_context.py`, `interlingua.py`.
  Done when: `time/location` не заховані тільки в relation modifiers або explanation strings.

- `G-056 [P0]` Додати quantifiers і obligations у scene graph.
  Файли: `scene_types.py`, `interlingua.py`.
  Done when: scene graph може виразити нормативні й квантифіковані структури напряму.

- `G-057 [P1]` Посилити entity persistence.
  Файли: `heuristic_backbone.py`, `semantic_context.py`, `interlingua.py`.
  Done when: одна сутність у кількох segments/documents стабільно живе як один entity identity.

- `G-058 [P1]` Додати mention-to-entity provenance graph.
  Файли: `semantic_context.py`, `scene_types.py`.
  Done when: entity grounding можна пояснити через chain `mention -> candidate -> resolved entity`.

- `G-059 [P1]` Додати same-event merging.
  Файли: `heuristic_backbone.py`, `semantic_scene.py`.
  Done when: кілька сегментів про ту саму подію можуть об'єднуватися, а не множити duplicate events.

- `G-060 [P1]` Додати alternative scene hypotheses.
  Файли: `semantic_scene.py`, `types.py`, `pipeline.py`.
  Done when: scene layer може передати вниз кілька прочитань при неоднозначності.

- `G-061 [P1]` Протягнути structural-unit references у scene objects.
  Файли: `scene_types.py`, `text_semantics.py`, `heuristic_backbone.py`.
  Done when: event/entity/state objects знають, з яких structural units вони виникли.

- `G-062 [P1]` Додати scene-level regression suite по paraphrases і mixed inputs.
  Файли: tests/benchmarks.
  Done when: scene graph перевіряється не лише на одному surface phrasing.

### G. `L5` Canonical Semantic Interlingua

- `G-063 [P0]` Створити canonical predicate inventory.
  Файли: новий predicate registry + `interlingua.py`.
  Done when: predicates нормалізуються не лише через `normalize_symbol_text`, а через керований inventory.

- `G-064 [P0]` Додати event template registry.
  Файли: новий template registry + `interlingua.py`.
  Done when: канонізація relation/event semantics спирається на templates, а не тільки на normalized strings.

- `G-065 [P0]` Додати generic vs instance semantics.
  Файли: `interlingua_types.py`, `interlingua.py`.
  Done when: “зірки генерують планети” і “ця зірка породила цю планету” не зливаються.

- `G-066 [P1]` Додати tense/aspect/epistemic stance.
  Файли: `interlingua_types.py`, `interlingua.py`.
  Done when: `reported/observed/hypothetical/required/past/ongoing/completed` distinctions не губляться.

- `G-067 [P1]` Канонізувати modality separately from goal semantics.
  Файли: `interlingua.py`, `scene_types.py`.
  Done when: “повинен”, “хоче”, “може”, “має на меті” мають різні canonical forms.

- `G-068 [P1]` Додати cross-language equivalence tests.
  Файли: tests/benchmarks.
  Done when: одна семантика в українській, англійській, scientific prose і compact declarative syntax сходиться в той самий interlingua form.

- `G-069 [P1]` Розвести canonical keys і human-readable labels.
  Файли: `interlingua_types.py`, `ontology_growth.py`.
  Done when: людиночитний label не є surrogate for canonical semantic identity.

- `G-070 [P1]` Додати ontology naming bridge.
  Файли: `ontology_growth.py`.
  Done when: `concept_name` більше не дорівнює просто `concept_key`, а проходить через кероване human-readable naming.

- `G-071 [P2]` Ввести interlingua revision policy.
  Файли: `interlingua.py`, `pipeline.py`.
  Done when: канонічні claims можуть бути переглянуті без зламу identities.

### H. `L6` Probabilistic Symbolic Compiler

- `G-072 [P0]` Навчити compiler тримати multiple candidate interpretations per segment.
  Файли: `symbolic_compiler.py`.
  Done when: одна неоднозначна фраза може породити кілька candidate hypotheses families.

- `G-073 [P0]` Додати candidate rules як окрему output family.
  Файли: `symbolic_compiler.py`, `interlingua.py`.
  Done when: система компілює не тільки facts/goals, а й candidate rules/templates.

- `G-074 [P0]` Додати support sets і evidence bundles.
  Файли: `symbolic_compiler.py`, `verification.py`.
  Done when: provenance це не лише strings на кшталт `segment:3`, а структурований набір evidences.

- `G-075 [P1]` Зробити hidden-cause hypotheses first-class compiled outputs.
  Файли: `symbolic_compiler.py`, `verification.py`, `verifier_stack.py`.
  Done when: abduction targets не виникають лише як repair action, а мають свої compiled objects.

- `G-076 [P1]` Додати family-specific confidence calibration.
  Файли: `symbolic_compiler.py`.
  Done when: state/relation/goal/rule/hypothesis families мають різні confidence models.

- `G-077 [P1]` Компілювати structural evidence і parser disagreement явно.
  Файли: `symbolic_compiler.py`, `text_semantics.py`.
  Done when: compiler знає, які parsers погоджуються/не погоджуються на джерелі.

- `G-078 [P1]` Підтримати hypothesis lineage across revisions.
  Файли: `symbolic_compiler.py`, `world_state_writeback.py`.
  Done when: нова версія hypothesis не втрачає lineage з попередньою.

- `G-079 [P1]` Додати compiler metrics gates.
  Файли: benchmark tooling, tests.
  Done when: `fact precision/recall`, `rule precision`, `hidden-cause quality` міряються окремо.

### I. `L7` Verification and Repair

- `G-080 [P0]` Переписати base verifier так, щоб він реально використовував `document`, `scene`, `interlingua`.
  Файли: `verification.py`.
  Done when: verify logic більше не робить `del document, interlingua, scene`.

- `G-081 [P0]` Додати parser-agreement validator family.
  Файли: `verification.py`, `verifier_stack.py`, `orchestrator.py`.
  Done when: parser agreement не живе лише як aggregate metadata, а стає окремим validation family.

- `G-082 [P0]` Додати memory-corroboration validator family.
  Файли: `orchestrator.py`, `verifier_stack.py`, memory modules.
  Done when: memory corroboration прямо впливає на validation records.

- `G-083 [P0]` Додати goal-alignment validator family.
  Файли: `verification.py`, `verifier_stack.py`.
  Done when: hypotheses перевіряються відносно `goal/target/task context`, а не лише внутрішнього score.

- `G-084 [P1]` Додати contradiction scope localization.
  Файли: `verification.py`, `verifier_stack.py`.
  Done when: система може сказати, що саме суперечить чому і на якому span/scope.

- `G-085 [P1]` Додати clarification-needed records.
  Файли: `verification.py`, `verifier_stack.py`.
  Done when: repair layer може породжувати “треба уточнення”, а не тільки `trigger_hidden_cause_abduction`.

- `G-086 [P1]` Зробити repair scheduler окремою підсистемою.
  Файли: `verifier_stack.py`, `planner_guidance.py`.
  Done when: repair actions дедупляться, ранжуються, групуються по dependent chains.

- `G-087 [P1]` Додати verification-driven scene/interlingua revision hooks.
  Файли: `pipeline.py`, `semantic_scene.py`, `interlingua.py`.
  Done when: verifier може просити перегляд scene/interlingua, а не тільки annotate compiled outputs.

- `G-088 [P1]` Додати calibration suite для `supported/deferred/conflicted`.
  Файли: benchmarks/tests.
  Done when: verification statuses проходять окрему calibration оцінку.

- `G-089 [P1]` Додати human-readable verifier rationales.
  Файли: `verification.py`, `verifier_stack.py`.
  Done when: кожен validation/repair record легко пояснюється оператору.

### J. `L8` Persistent World State and Memory Writeback

- `G-090 [P0]` Додати full provenance envelope у world-state records.
  Файли: `world_state_writeback.py`, `ontology_growth.py`, memory modules.
  Done when: запис у world state містить `source_id/document_id/episode_id/span/timestamp/confidence/status`.

- `G-091 [P0]` Додати relation-to-existing-objects semantics.
  Файли: `world_state_writeback.py`.
  Done when: writeback уміє позначати `new/merge/refine/supersede/contradict`.

- `G-092 [P1]` Додати revision policy для `active/hypothetical/contradicted`.
  Файли: `world_state_writeback.py`, memory modules.
  Done when: зміна статусу не просто заміняє record, а проходить через контрольовану policy.

- `G-093 [P1]` Додати exact symbolic memory filters by validator family.
  Файли: `memory_hints.py`, `memory_policy.py`, `omen_scale.py`.
  Done when: у memory не пишеться все підряд з однаковим пріоритетом.

- `G-094 [P1]` Підтягнути world-graph links до source provenance.
  Файли: `world_graph_records.py`, `omen_symbolic/world_graph.py`.
  Done when: world graph може відтрасувати grounding nodes до їх source spans і source ids.

- `G-095 [P1]` Розвинути ontology growth у керовану ontology program.
  Файли: `ontology_growth.py`, `ontology_atoms.py`.
  Done when: concepts мають naming, promotion/demotion, relation to member records, stability criteria.

- `G-096 [P1]` Додати writeback audit trail.
  Файли: `world_state_writeback.py`, runtime logging.
  Done when: кожне оновлення world state можна перевірити ретроспективно.

### K. `L9` Reasoning, Planning, EMC, Generation

- `G-097 [P0]` Змусити planner читати richer grounded objects, а не лише projection summary.
  Файли: `planner_state.py`, `planner_semantics.py`, planner consumers.
  Done when: planning використовує не тільки compressed resources/operators, а й verifier/world-state lineage.

- `G-098 [P0]` Підключити verifier-family outputs у EMC.
  Файли: `emc_signals.py`, `omen_scale.py`.
  Done when: EMC отримує не лише aggregate uncertainty, а й per-family validation signals.

- `G-099 [P0]` Заборонити raw-text shortcuts для planner/generation path.
  Файли: `omen_scale.py`, `omen_osf.py`.
  Done when: рішення й generation формуються з grounded state even when raw text is present.

- `G-100 [P1]` Додати counterfactual simulation hooks для hypothetical/contradicted worlds.
  Файли: planner/world model modules.
  Done when: альтернативні світи реально впливають на planning і reasoning.

- `G-101 [P1]` Додати grounding-aware reasoning budget allocation.
  Файли: `emc_signals.py`, `omen_emc.py`, `omen_scale.py`.
  Done when: verification pressure і ambiguity реально керують глибиною reasoning.

- `G-102 [P1]` Додати explanation path від answer back to grounded records.
  Файли: generation stack, `planner_state.py`, grounding records.
  Done when: відповідь можна пояснити через конкретні validated world-state objects.

- `G-103 [P1]` Підключити verified outcomes до online symbolic learning.
  Файли: symbolic learning runtime, `omen_scale.py`.
  Done when: grounding results реально живлять symbolic improvement loop.

- `G-104 [P1]` Додати controlled self-improvement loop для grounding.
  Файли: training/eval tooling, runtime logging.
  Done when: система може вчитися на verified outcomes без порушення eval discipline.

- `G-105 [P2]` Додати planner/generation eval на ambiguous/multilingual inputs.
  Файли: benchmarks/tests.
  Done when: видно, як якість grounding впливає на planning/generation success.

- `G-106 [P2]` Додати user/operator facing grounding explanation surface.
  Файли: external API/UI layer.
  Done when: оператор бачить grounding chain, а не тільки фінальну відповідь.

### L. Learned Backbone, Data, Benchmarks, Safe Training

- `G-107 [P0]` Зробити реальний implementation для `SemanticGroundingBackbone`.
  Файли: новий backbone module, `backbone.py`, `semantic_scene.py`, `omen_scale.py`.
  Done when: у репозиторії є не лише protocol slot, а production-grade learned або hybrid backbone.

- `G-108 [P0]` Побудувати gold corpus для L1 routing.
  Файли: benchmark/test data.
  Done when: можна міряти modality/subtype/script/ambiguity accuracy.

- `G-109 [P0]` Побудувати gold corpus для L3 linguistic grounding.
  Файли: benchmark/test data.
  Done when: є benchmark на morphology, clause decomposition, mention/coreference, role labeling.

- `G-110 [P0]` Побудувати gold corpus для L4 scene graphs.
  Файли: benchmark/test data.
  Done when: є structured targets на entities/events/roles/modality/temporality/polarity.

- `G-111 [P0]` Побудувати gold corpus для L5 interlingua equivalence.
  Файли: benchmark/test data.
  Done when: однаковий зміст у різних мовах/форматах оцінюється на convergence.

- `G-112 [P0]` Побудувати contradiction/repair corpus.
  Файли: benchmark/test data.
  Done when: L7 repair families оцінюються на реальних ambiguity/conflict scenarios.

- `G-113 [P1]` Побудувати hidden-cause abduction corpus.
  Файли: benchmark/test data.
  Done when: quality `trigger_hidden_cause_abduction` можна реально міряти.

- `G-114 [P1]` Побудувати structured/mixed-document corpus.
  Файли: benchmark/test data.
  Done when: `code + prose`, `logs + comments`, `tables + claims`, `configs + explanation` оцінюються системно.

- `G-115 [P1]` Зробити runtime replay buffer для grounding failures.
  Файли: runtime logging, benchmarks.
  Done when: усі `deferred/conflicted/ambiguous` приклади збираються в окремий loop.

- `G-116 [P1]` Додати shadow evaluation режим для learned backbone.
  Файли: `omen_scale.py`, benchmark tooling.
  Done when: learned path можна оцінювати паралельно з fallback path.

- `G-117 [P1]` Додати safe online update regime.
  Файли: training/runtime tooling.
  Done when: self-improvement не змішує production updates з uncontrolled train/eval leakage.

- `G-118 [P1]` Додати release gates для grounding quality.
  Файли: CI/benchmarks.
  Done when: зміни в routing/scene/interlingua/verifier не зливаються без quality gates.

### M. Тести І Якість

- `G-119 [P0]` Додати byte-provenance tests.
- `G-120 [P0]` Додати orchestrator-first integration tests.
- `G-121 [P0]` Додати router unification tests.
- `G-122 [P0]` Додати multilingual equivalence tests.
- `G-123 [P0]` Додати parser disagreement tests.
- `G-124 [P1]` Додати world-state revision tests.
- `G-125 [P1]` Додати planner alternative-world tests.
- `G-126 [P1]` Додати exact memory corroboration tests.
- `G-127 [P1]` Додати fuzz tests для malformed structured inputs.
- `G-128 [P1]` Додати benchmark/CI dashboards для grounding metrics.

## 10. Що Не Треба Робити Зараз

Ось які анти-патерни зараз найбільш небезпечні:

- не намагатися “добити якість” нескінченним розростанням regex/marker rules;
- не робити ще один окремий альтернативний runtime для grounding;
- не переносити semantic burden у planner або symbolic core як компенсацію слабкого front-end;
- не писати у world state і memory сліпо все, що пройшло fallback extraction;
- не робити learned backbone без shadow evaluation і quality gates;
- не зводити multilingual grounding до “є кирилиця чи нема кирилиці”;
- не маскувати `mixed` і `unknown` inputs під `generic_text`.

## 11. Рекомендована Послідовність Робіт

### Фаза 1. Архітектурне вирівнювання

- `G-001` - `G-008`
- `G-013`
- `G-019`

Результат:

- grounding стає canonical subsystem;
- provenance і routing contracts перестають роз'їжджатися;
- runtime має одну правильну точку входу в grounding.

### Фаза 2. Сильний L1/L2 фундамент

- `G-020` - `G-038`

Результат:

- mixed/structured inputs починають реально ground-итись через subtype-specific structural evidence;
- parser candidates і ambiguity стають реальними control signals.

### Фаза 3. Справжній multilingual L3

- `G-039` - `G-052`

Результат:

- natural language grounding перестає бути mostly marker-based.

### Фаза 4. Перебудова L4/L5

- `G-053` - `G-071`

Результат:

- scene graph і interlingua стають тими шарами, які реально несуть meaning.

### Фаза 5. Посилення L6/L7/L8

- `G-072` - `G-096`

Результат:

- символічний downstream отримує кращі, менш крихкі, краще верифіковані inputs.

### Фаза 6. Grounded reasoning і learning loop

- `G-097` - `G-128`

Результат:

- planner, EMC, generation і online symbolic learning починають жити на справді grounded state.

## 12. Мінімальний Definition Of Done По Шарах

### `L0`

- всі grounded objects traceable до byte spans;
- є `source_id/document_id/episode_id`.

### `L1`

- один canonical router;
- calibrated confidence;
- parser candidates і ambiguity реально споживаються нижче.

### `L2`

- structured inputs ground-яться через strong deterministic parsers;
- structural disagreements і units живуть далі document layer.

### `L3`

- є morphology, lemma, POS, dependency, clause objects, mention/coreference candidates;
- українська і mixed-language inputs працюють не як special case hacks.

### `L4`

- meaning represented event-frame graph objects with roles, polarity, modality, time, provenance;
- є entity persistence.

### `L5`

- однаковий зміст у різних мовах/стилях сходиться до одного canonical semantic form.

### `L6`

- compiler тримає multiple interpretations, support sets, conflict tags, candidate rules, hidden-cause hypotheses.

### `L7`

- verifier stack має parser/memory/world-model/temporal/goal families;
- repair scheduler координує follow-up actions.

### `L8`

- writeback не сліпий;
- provenance, revision, merge/supersede semantics контрольовані;
- ontology growth має meaningful naming і policy.

### `L9`

- planners/EMC/generator читають grounded state, а не raw text shortcuts;
- grounding results живлять online symbolic learning без втрати verification discipline.

## 13. Найточніший Підсумок

OMEN вже має правильний skeleton.

Що не треба робити:

- ламати world-state/symbolic/memory backbone;
- переписувати все з нуля;
- компенсувати слабке grounding ядро новими downstream костилями.

Що треба робити:

- зробити grounding first-class architecture program;
- посилити L1/L2 як typed perception + structural evidence;
- сильно посилити L3-L5 як справжній multilingual semantic front-end;
- розширити L6/L7 так, щоб uncertainty, repair і alternative interpretations були природною частиною runtime;
- довести L8/L9 до стану, де reasoning/planning/generation дійсно спираються на grounded world state;
- побудувати data/eval/training flywheel для learned grounding, safe online symbolic learning і controlled self-improvement.

Саме це і є шлях від “strong research-grade runtime with heuristic semantic front-end” до того robust world-grounded AI system, який описаний у `concept.md` і `GROUNDING_MASTERPLAN.md`.
