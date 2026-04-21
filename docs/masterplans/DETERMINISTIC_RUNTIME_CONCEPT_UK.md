# OMEN — Детермінізована канонічна концепція runtime, grounding і керування станом

## 1. Статус документа

Цей документ є нормативним masterplan-доповненням до `docs/masterplans/concept.md`
для всього зовнішнього циклу:

- ingest;
- routing;
- grounding;
- verification;
- world-state writeback;
- memory writeback/recall;
- planner ingress;
- generation from canonical state.

Нормативний документ для жорсткого розподілу ownership між нейронною та
символьною частинами:
`docs/masterplans/NEURO_SYMBOLIC_BOUNDARY_UK.md`.

Його мета не змінити ідентичність OMEN, а зробити її максимально
детермінізованою без втрати наявних можливостей.

Якщо між цим документом і старішими формулюваннями про grounding,
semantic ingestion, verification або planner ingress є суперечність,
для цих підсистем пріоритет має цей документ.

Якщо суперечність стосується саме межі між `neuro`, `symbolic` і
deterministic boundary, пріоритет має
`NEURO_SYMBOLIC_BOUNDARY_UK.md`.

## 2. Що саме вже є в коді і є правильною опорою

У поточному репозиторії вже існує сильний кістяк, який треба не викидати,
а зробити жорсткішим і більш детермінованим.

Канонічний runtime:

- `omen.py`
- `omen_canonical.py`
- `omen_scale.py`

Канонічний зовнішній grounding-конвеєр уже має явні фази:

- `omen_grounding/source_routing.py`
- `omen_grounding/text_semantics.py`
- `omen_grounding/structural_scene.py`
- `omen_grounding/semantic_scene.py`
- `omen_grounding/interlingua.py`
- `omen_grounding/symbolic_compiler.py`
- `omen_grounding/verification.py`
- `omen_grounding/verifier_stack.py`
- `omen_grounding/world_state_writeback.py`
- `omen_grounding/memory_policy.py`
- `omen_grounding/planner_state.py`
- `omen_grounding/planner_bridge.py`

Канонічні внутрішні об'єкти, які вже існують і мають лишитися основою:

- `GroundingSourceProfile`
- `GroundingRuntimeContract`
- `GroundedTextDocument`
- `GroundingSpan`
- `SemanticSceneGraph`
- `CanonicalInterlingua`
- `SymbolicCompilationResult`
- `GroundingVerificationReport`
- `GroundingWorldStateWriteback`
- `SymbolicTaskContext`
- `PlannerWorldState`
- `CanonicalWorldState`

Канонічний symbolic/world ingress уже існує:

- `omen_symbolic/execution_trace.py`
- `omen_symbolic/world_graph.py`
- `omen_symbolic/memory_index.py`
- `omen_prolog.py`

Проблема не в тому, що архітектури немає.

Проблема в тому, що зараз у ній змішані:

- справді детерміновані структурні шари;
- сильні, але ще надто вільні semantic fallback-и;
- евристичні кандидати, які місцями занадто близько підходять до смислового ядра.

## 3. Головний принцип нового канону

### 3.1 Що означає "максимально детермінізована" система

За однакових:

- байтів;
- source metadata;
- конфіга;
- версії коду;
- frozen-параметрів моделей;
- увімкнених модулів;

система повинна будувати той самий:

- routing;
- segmentation;
- structural-unit graph;
- interlingua;
- symbolic hypotheses;
- verification result;
- world-state status;
- planner ingress;
- generation context.

Невизначеність у системі дозволена лише як явно представлена епістемічна
невизначеність, а не як нечітка поведінка пайплайна.

### 3.2 Детермінізм не означає "без невизначеності"

Система може і повинна зберігати:

- `active`
- `hypothetical`
- `contradicted`
- `cited`
- `questioned`
- `hedged`
- branching alternatives

Але сам механізм переходу між цими станами має бути детермінованим.

### 3.3 Евристика не може володіти смислом

Евристика може:

- запропонувати кандидат;
- знайти дешевий prior;
- локалізувати підозрілу ділянку;
- підказати verification scheduler;
- згенерувати hypothesis branch.

Евристика не може:

- напряму створювати `active` world state;
- напряму створювати verified rule;
- тихо переозначати structural-primary факт;
- тихо підміняти canonical interlingua;
- робити planner truth source.

## 4. Канонічна ієрархія авторитету

Кожен шар має чіткий authority level.

### 4.1 Authority A — carrier truth

Найвищий авторитет мають:

- сирі UTF-8 байти;
- їхні span-и;
- `source_id`;
- `document_id`;
- `episode_id`.

Це незаперечний source of record.

### 4.2 Authority B — typed structural truth

Другий рівень авторитету:

- `GroundingSourceProfile`
- `GroundedStructuralUnit`
- deterministic segment boundaries
- deterministic structural fields

Якщо документ є структурно зрозумілим, саме цей рівень володіє первинним смислом.

### 4.3 Authority C — canonical normalized meaning

Третій рівень:

- `SemanticSceneGraph`
- `CanonicalInterlingua`
- deterministic claim semantics

Цей рівень нормалізує смисл, але не має права втратити provenance до A/B.

### 4.4 Authority D — epistemic truth maintenance

Четвертий рівень:

- verification;
- verifier stack;
- world-state status assignment;
- memory writeback policy;
- planner projection.

Саме цей рівень вирішує, що є:

- активним фактом;
- гіпотезою;
- суперечністю;
- правилом у symbolic lifecycle.

### 4.5 Authority E — operational consumption

П'ятий рівень:

- `SymbolicTaskContext`
- `PlannerWorldState`
- `CanonicalWorldState`
- decoding/generation

Generation і planning мають споживати тільки вже нормалізований і
епістемічно класифікований стан.

### 4.6 Authority H — bounded heuristic support

Найнижчий авторитет:

- regex/fallback extraction;
- heuristic semantic backbone;
- cheap hint builders;
- hidden-cause proposal heuristics;
- weak discourse/coreference guesses.

Вони ніколи не можуть перескочити напряму до `active`.

## 5. Канонічний фазовий граф

Єдиний дозволений зовнішній цикл має бути таким:

`bytes -> source profile -> segments/spans -> structural units -> semantic scene -> canonical interlingua -> deterministic symbolic lowering -> verification -> world-state writeback -> memory/planner/task context -> canonical world state -> generation`

Нижче визначено, як саме має працювати кожна фаза.

## 6. Фаза 0. Ingest і identity

Вхідний контракт:

- carrier = UTF-8 bytes;
- усі downstream span-и мають бути відтворюваними по bytes;
- кожен документ повинен, де можливо, мати:
  - `source_id`
  - `document_id`
  - `episode_id`

Обов'язкові інваріанти:

- `GroundingSpan.text` має точно відповідати `byte_start:byte_end`;
- жоден semantic object не може існувати без посилання принаймні на segment або source span;
- жодна downstream фаза не має права міняти carrier truth.

## 7. Фаза 1. Source routing

### 7.1 Що лишається

Контракт `GroundingSourceProfile` лишається канонічним.

Поля, які є обов'язковими:

- `language`
- `script`
- `domain`
- `modality`
- `subtype`
- `verification_path`
- `confidence`
- `ambiguity`
- `parser_candidates`
- `evidence`

### 7.2 Що треба замінити

Поточний score-style router у `omen_grounding/source_routing.py` треба
концептуально замінити з "розмитого набору marker weights" на:

- декларативний rule registry;
- явну subtype-precedence matrix;
- фіксовані deterministic thresholds;
- стабільний tie-breaker;
- повний evidence ledger.

Нормативне правило:

- однаковий текст і однакові hints завжди дають той самий `GroundingSourceProfile`;
- tie-break ніколи не залежить від випадкового порядку умов;
- `parser_candidates` будуються лише з route result, а не окремою евристичною логікою.

### 7.3 Маршрутизація не визначає істину, а лише verification lane

Routing вирішує:

- який deterministic parser запускати;
- який verification lane обов'язковий;
- хто structural primary.

Routing не вирішує:

- чи факт істинний;
- чи relation має стати `active`;
- чи rule додається в KB.

## 8. Фаза 2. Segmentation і structural units

Сегментація має бути детермінованою і span-preserving.

Канонічні джерела сегментації:

- line boundaries;
- explicit dialogue turns;
- instruction steps;
- punctuation-clause boundaries;
- deterministic block markers.

Канонічні structural units:

- `section_header`
- `key_value_record`
- `json_record`
- `log_entry`
- `table_row`
- `speaker_turn`
- `clause`
- `citation_region`

Потрібна жорстка policy:

- якщо unit можна витягнути структурно, він має витягуватися структурно;
- semantic fallback не має права дублювати цей самий unit як primary source;
- один segment може бути або fully structural-primary, або hybrid, але це має бути явно позначено в metadata.

## 9. Фаза 3. Structural-primary semantics

### 9.1 Це головний детермінізований semantic path

`omen_grounding/structural_scene.py` має бути authoritative path для сегментів,
де є:

- config/schema structure;
- JSON-like records;
- tables/logs;
- speaker turns;
- instructional clauses.

### 9.2 Що саме structural-primary має будувати

Він має напряму і детерміновано будувати:

- entities;
- states;
- goals;
- attributed claims;
- частину relation/event frames там, де є достатньо структурних маркерів.

### 9.3 Правило пріоритету

Для segment-ів, позначених як structural-primary:

- structural scene owner = єдиний primary semantic owner;
- fallback backbone може лише збагачувати low-authority поля;
- fallback не має права переписувати state/goal/claim source.

## 10. Фаза 4. Semantic fallback і learned/backbone path

### 10.1 Що зберігається

`SemanticGroundingBackbone` і `HeuristicFallbackSemanticBackbone` залишаються,
бо вони дають:

- relation proposals;
- event proposals;
- condition/explanation/temporal proposals;
- coreference-like proposals;
- hidden semantic hints для неструктурованого тексту.

### 10.2 Новий статус цього шару

Відтепер це не semantic owner, а proposal layer.

Його виходи повинні мати такі властивості:

- low authority by default;
- явне provenance marker;
- явний `claim_source`;
- неможливість напряму породити `active` state без verification.

### 10.3 Сегментна модель володіння

Потрібно жорстко ввести `segment ownership`:

- structural-primary segments: owner = structural scene;
- non-structural segments: owner = semantic fallback/backbone;
- hybrid segments: structural layer owns states/goals/attribution, fallback owns only event/relation proposals на вільних слотах.

## 11. Фаза 5. Canonical interlingua

`omen_grounding/interlingua.py` вже має правильний напрямок і стає
обов'язковим canonical normalization layer.

Вимоги:

- один semantic object -> одна canonical form;
- stable ids/keys;
- preservation of:
  - source segments
  - source spans
  - evidence refs
  - epistemic status
  - claim source
  - semantic mode
  - quantifier mode

Interlingua не повинна бути "best effort paraphrase layer".

Interlingua повинна бути deterministic normalization layer.

## 12. Фаза 6. Символічне зниження

### 12.1 Що треба концептуально змінити

Фразу "probabilistic symbolic compilation" треба вважати застарілою для
канонічного inference-контуру.

Канонічно має бути:

- deterministic symbolic lowering;
- explicit epistemic scoring;
- explicit deferred/conflict flags;
- deterministic candidate-rule eligibility.

### 12.2 Що саме має робити compiler

`omen_grounding/symbolic_compiler.py` повинен:

- детерміновано переводити interlingua в hypotheses;
- не губити provenance;
- детерміновано позначати:
  - `deferred`
  - `conflict_tag`
  - `semantic_mode`
  - `quantifier_mode`
  - `claim_source`

### 12.3 Candidate rule policy

Rule candidates можна породжувати лише якщо виконано все:

- claim non-heuristic;
- semantic mode = `generic` або `rule` або `obligation`;
- є достатній support/evidence;
- provenance не містить heuristic authority markers.

Будь-який rule candidate із heuristic source не може потрапляти в canonical rule lifecycle.

## 13. Фаза 7. Verification

### 13.1 Verification має бути deterministic scoring engine

`omen_grounding/verification.py` і `verifier_stack.py` мають бути єдиним місцем,
де proposal переходить у:

- `supported`
- `deferred`
- `conflicted`

Ці рішення мають ґрунтуватися лише на:

- rule-based score composition;
- world-state overlap;
- memory overlap;
- temporal/discourse evidence;
- ontology overlap;
- source/claim status policy.

### 13.2 Hidden-cause abduction лишається, але в жорстких межах

При конфлікті система має право будувати hidden-cause proposal.

Але hidden-cause mechanism:

- ніколи не створює `active` факт одразу;
- завжди створює `hypothetical` branch;
- має stable ranking;
- має bounded beam;
- має provenance до trigger hypothesis.

## 14. Фаза 8. World-state writeback

`omen_grounding/world_state_writeback.py` стає центральним truth-maintenance gate.

Канонічні стани world record:

- `active`
- `hypothetical`
- `contradicted`

Канонічні правила переходу:

- supported asserted non-heuristic instance claim -> `active`;
- supported but nonasserted claim -> `hypothetical`;
- supported rule/generic/obligation claim -> `hypothetical` + `route_to_symbolic_rule_lifecycle`;
- heuristic claim -> тільки `hypothetical` + `require_grounding_confirmation`;
- conflicted verified claim -> `contradicted`, якщо це не heuristic-only case;
- hidden-cause proposal -> `hypothetical`.

World state — це не dump усіх hypotheses.

World state — це вже детермінізовано класифікований epistemic state.

## 15. Фаза 9. Memory

### 15.1 Grounding memory

`omen_grounding/memory_policy.py` вже задає правильний напрям:

- status-aware selection;
- family-aware diversity;
- no collapse of contradictions.

Це треба зафіксувати як канон:

- memory пише не "все підряд", а status-diverse evidence;
- `active`, `hypothetical`, `contradicted` мають зберігатися окремо;
- recall ніколи не має знімати contradiction тільки тому, що активний запис "схожий".

### 15.2 Symbolic memory

`omen_symbolic/memory_index.py` лишається exact symbolic recall path.

Вимога:

- symbolic recall і grounding recall не можна змішувати в один неявний канал;
- у `SymbolicTaskContext` вони мають лишатися окремими buckets.

## 16. Фаза 10. SymbolicTaskContext

`omen_prolog.SymbolicTaskContext` є канонічним symbolic ingress object.

Це треба не змінювати, а посилити.

Канонічні bucket-и:

- `observed_now_facts`
- `memory_derived_facts`
- `memory_grounding_records`
- `grounding_ontology_facts`
- `grounding_world_state_active_facts`
- `grounding_world_state_hypothetical_facts`
- `grounding_world_state_contradicted_facts`
- `saliency_derived_facts`
- `net_derived_facts`
- `grounding_derived_facts`
- `world_context_facts`
- `abduced_support_facts`

Жорсткі правила:

- reasoning consumes `active` facts;
- planner may consume `active + hypothetical`;
- contradiction scope is separate and never silently merged into reasoning facts;
- raw text itself is never a symbolic truth source.

## 17. Фаза 11. Planner ingress

`omen_grounding/planner_state.py` і `planner_bridge.py` вже роблять майже
потрібне: проектують world-state buckets, verification lineage, graph records,
candidate rules і repair directives в planner space.

Канонічно має бути так:

- planner читає тільки `PlannerWorldState`;
- planner не читає raw text як truth source;
- planner не читає heuristic hint напряму;
- planner бачить:
  - active world
  - hypothetical branches
  - contradictions
  - constraints
  - repair directives
  - lineage symbols

Будь-яка planner operator/action повинна бути простежуваною назад до:

- world-state record;
- verification record;
- hypothesis;
- graph lineage;
- source span.

## 18. Фаза 12. CanonicalWorldState і generation

`omen_symbolic/world_graph.py` і `omen_scale.py` вже задають правильний канон:

- graph-primary world state;
- dense state як derived view;
- `CanonicalWorldState` як єдиний носій для decode-facing state.

Нормативне правило:

- generation не працює безпосередньо від prompt-as-truth;
- generation працює від `CanonicalWorldState`;
- planner state і symbolic state входять у decode context як вже
  нормалізовані компоненти;
- `grounded_state` не є альтернативною природою істини, а лише readout.

## 19. Що на що треба замінити

| Поточний слабкий патерн | Канонічна заміна |
| --- | --- |
| score soup у source routing | declarative route registry + stable precedence + deterministic tie-break |
| heuristic ownership of natural-language meaning | segment ownership model + proposal-only fallback |
| "probabilistic symbolic compilation" | deterministic lowering + explicit epistemic scoring |
| silent mixing of structural and fallback semantics | structural-primary authority + hybrid policy |
| heuristic candidate rules | only non-heuristic rule lifecycle inputs |
| planner from loose text semantics | planner from `PlannerWorldState` only |
| memory as generic retrieval bag | status-aware grounding memory + separate symbolic memory |
| generation from token context with optional symbolic extras | generation from `CanonicalWorldState` with graph/symbolic/planner state as primary |

## 20. Де евристики ще дозволені

Евристики дозволені тільки там, де формальний або структурний path
справді не покриває задачу.

Дозволені зони:

- fallback segmentation for messy text;
- cheap discourse-marker hints;
- cheap relation/event proposals for free natural language;
- coarse coreference proposals;
- hidden-cause proposal generation;
- rare-format bootstrap before schema is known.

Обов'язкові умови:

- heuristic output має бути явно позначений;
- heuristic output не може стати `active` без verification;
- heuristic output не може породити verified rule;
- heuristic disagreement з structural/formal path автоматично знижує authority.

## 21. Модульна відповідальність

### 21.1 `omen_scale.py`

Відповідає за:

- canonical stack forcing;
- виклик ingress/grounding;
- збір `SymbolicTaskContext`;
- підключення memory, world graph, prover, planner, decoder;
- побудову `CanonicalWorldState`.

Не відповідає за:

- локальне перевизначення semantic truth поза grounding contracts.

### 21.2 `omen_grounding/source_routing.py`

Відповідає тільки за:

- typed routing;
- parser lane selection;
- verification lane selection.

### 21.3 `omen_grounding/text_semantics.py`

Відповідає за:

- deterministic normalization;
- segmentation;
- structural extraction;
- byte/char span traceability.

Не відповідає за фінальну істину relation/event claims.

### 21.4 `omen_grounding/structural_scene.py`

Є primary semantic owner для structured і structural-natural segments.

### 21.5 `omen_grounding/semantic_scene.py`

Є merge/orchestration layer:

- structural primary first;
- fallback only where allowed;
- no overwrite of structural truth.

### 21.6 `interlingua.py`, `symbolic_compiler.py`, `verification.py`, `world_state_writeback.py`

Це canonical deterministic lowering core.

### 21.7 `memory_policy.py`, `planner_state.py`, `planner_bridge.py`

Це operational determinization core:

- what survives;
- what gets recalled;
- what planner sees.

## 22. Непорушні інваріанти

Система не вважається відповідною новому канону, якщо не виконується хоча б
один із цих пунктів:

1. Один і той самий input не дає той самий `GroundingSourceProfile`.
2. Span-и не round-trip-яться до carrier bytes.
3. Structural-primary segment може бути semantic-overwritten fallback-ом.
4. Heuristic claim може стати `active` без verification gate.
5. Heuristic claim може стати verified rule input.
6. Planner працює з raw text або неепістемічно класифікованими фактами.
7. Generation читає prompt як істину, минаючи canonical world state.
8. Memory writeback колапсує contradiction в один позитивний запис.
9. Contradicted facts silently enter `reasoning_facts()`.

## 23. Що треба перевіряти тестами

Мінімальний регресійний пакет має закріплювати:

- routing stability;
- span round-trip;
- structural-primary precedence;
- heuristic demotion;
- deterministic interlingua ids/keys;
- deterministic compilation outcome;
- world-state transition policy;
- memory status diversity;
- planner not-from-text;
- canonical world state composition.

Опора вже є в наявних тестах:

- `tests/test_grounding_text_semantics.py`
- `tests/test_grounding_scene_pipeline.py`
- `tests/test_grounding_verifier_stack.py`
- `tests/test_grounding_world_state_writeback.py`
- `tests/test_grounding_memory_policy.py`
- `tests/test_grounding_planner_state.py`
- `tests/test_omen_world_graph.py`
- `tests/test_symbolic_task_context_world_state.py`
- `tests/test_canonical_stack_protocol.py`

## 24. Поетапна міграція в коді

### 24.1 Спочатку

- централізувати deterministic route rules;
- винести stable tie-break policy;
- зафіксувати segment ownership.

### 24.2 Потім

- відокремити structural-primary semantics від fallback proposals жорсткіше;
- перевести compiler vocabulary з "probabilistic" у "deterministic + epistemic";
- централізувати status transition table.

### 24.3 Далі

- прибрати будь-який шлях heuristic -> active без verification;
- прибрати будь-який шлях heuristic -> rule lifecycle;
- зробити planner ingest тільки через `PlannerWorldState`.

### 24.4 В кінці

- вирівняти generation path так, щоб prompt був лише carrier/context, а не truth source;
- вирівняти memory writeback/recall з новою authority lattice;
- додати snapshot tests для однакових input -> однакових artifact trees.

## 25. Остаточний канон

OMEN не повинен ставати "простішим".

OMEN повинен ставати:

- жорсткіше типізованим;
- жорсткіше нормалізованим;
- жорсткіше provenance-aware;
- жорсткіше epistemic;
- жорсткіше route/verify/write controlled.

Фінальна формула цього документа:

`same bytes + same metadata + same config -> same typed grounding artifacts -> same interlingua -> same epistemic world state -> same planner ingress -> same generation state`

При цьому:

- багатозначність не зникає;
- вона просто стає явною, контрольованою і детерміновано представленою.

Саме так OMEN має залишити всі свої можливості, але перестати бути залежним
від евристики там, де архітектура вже готова мати формальний канон.
