# OMEN — Жорсткий розподіл відповідальностей між нейронною та символьною частинами

## 1. Статус документа

Цей документ є нормативним masterplan-доповненням до:

- `docs/masterplans/concept.md`
- `docs/masterplans/DETERMINISTIC_RUNTIME_CONCEPT_UK.md`
- `docs/masterplans/GROUNDING_MASTERPLAN.md`

Його задача: жорстко й однозначно визначити, що саме в OMEN належить:

- нейронній частині;
- символьній частині;
- детермінізованому boundary-шару між ними.

Якщо в старіших описах OMEN нейронна і символічна ролі змішані,
для питань ownership, authority, admission, verification, planner truth
і generation truth source пріоритет має цей документ.

## 2. Коротка відповідь на головне питання

### 2.1 Чи система зараз занадто евристична

Так, але не всюди однаково.

Система не є критично евристичною у:

- verification;
- world-state writeback;
- memory policy;
- planner projection.

Там уже існують сильні обмеження, які не дають low-authority path
напряму ставати `active`.

Система все ще занадто евристична в місцях, де proposal path
підходить надто близько до смислового ядра:

- fallback semantic extraction;
- hybrid merge у `omen_grounding/semantic_scene.py`;
- saliency / NET / abduction material, який занадто легко виглядає як
  reasoning-ready substance;
- neural / heuristic candidate flow усередині `omen_prolog.py`.

### 2.2 Чи ти даєш занадто багато symbolic, а не neuro

Не в головному сенсі.

Проблема не в тому, що symbolic має забагато влади.

Проблема в тому, що symbolic місцями змушений носити на собі ще й proposal-роль,
яка логічно належить нейронному або low-authority proposal-шару.

Правильна формула така:

- symbolic повинен мати максимальну владу над істиною, статусами, правилами,
  пам'яттю, planner ingress і canonical world state;
- neuro повинен мати максимальну силу в перцепції, узагальненні,
  similarity, proposal generation, saliency, latent world prior і decode;
- heuristic path повинен бути підкласом proposal path, а не підкласом truth.

Отже, виправлення не в тому, щоб "послабити symbolic".

Виправлення в тому, щоб різко відрізати:

- proposal;
- verification;
- truth maintenance;
- operational consumption.

## 3. Канонічне правило в одному абзаці

Нейронна частина:

- вчиться;
- компресує;
- оцінює подібність;
- будує latent priors;
- виділяє saliency;
- пропонує кандидати;
- реалізує відповідь у поверхневу форму.

Символьна частина:

- типізує;
- канонізує;
- відслідковує provenance;
- верифікує;
- призначає epistemic/world status;
- керує rule lifecycle;
- вирішує, що попадає в пам'ять;
- вирішує, що бачить planner;
- формує canonical action/state surface для generation.

Boundary-шар:

- переводить neuro proposals у deterministic typed contracts;
- не дозволяє proposal path перескочити в truth path;
- зберігає всю невизначеність явно, а не неявно.

## 4. Канонічний конвеєр ownership

Єдиний правильний ownership-граф:

`bytes -> neuro perception/proposal -> deterministic normalization/boundary -> symbolic verification/status/rule lifecycle -> canonical world state/planner state -> neuro realization`

Розшифровка по фазах:

1. `bytes / tokens / encoder state`
   Owner: `neuro`
2. `routing / segments / spans / structural units`
   Owner: `deterministic boundary`
3. `fallback semantic proposals / saliency proposals / retrieval proposals`
   Owner: `neuro`
4. `canonical scene / interlingua / symbolic lowering`
   Owner: `deterministic boundary`
5. `verification / contradiction / world-state status / rule lifecycle`
   Owner: `symbolic`
6. `memory admission / planner ingress / canonical world state`
   Owner: `symbolic`
7. `surface decoding / linguistic realization`
   Owner: `neuro`

## 5. Що саме має робити нейронна частина

### 5.1 Перцепція і репрезентація

Нейронна частина повинна відповідати за:

- byte/token encoding;
- latent compression;
- distributed representation learning;
- multilingual abstraction;
- soft pattern completion;
- similarity and retrieval embeddings;
- saliency detection;
- latent world priors;
- learned forecasting of plausible continuations.

Це означає:

- neuro працює з безперервними репрезентаціями;
- neuro може бути дуже сильним у тому, що не можна жорстко виписати правилами;
- neuro не повинен бути truth authority.

### 5.2 Proposal generation

Нейронна частина повинна генерувати:

- semantic proposals;
- entity/relation/event/state hints;
- retrieval candidates;
- abduction candidates;
- hidden-cause candidates;
- candidate rules;
- uncertainty features;
- ranking scores;
- saliency-led hypotheses.

Але всі ці outputs мають бути proposal-only.

Нейронний output повинен мати зміст:

- "це може бути так";
- "це хороший кандидат";
- "ось latent evidence";
- "ось ranking";
- "ось prior".

Нейронний output не повинен означати:

- "це вже істина";
- "це вже active world fact";
- "це вже accepted rule";
- "це вже planner truth".

### 5.3 Retrieval, saliency, similarity

Саме neuro повинен володіти:

- approximate recall ranking;
- semantic nearest-neighbour behavior;
- associative linkage;
- soft attention over memory;
- token-to-latent and latent-to-candidate mapping.

Але memory admission policy не належить neuro.

### 5.4 Surface realization

Саме neuro повинен володіти:

- fluency;
- lexicalization;
- stylistic realization;
- response synthesis;
- language-specific surface choices.

Generation може використовувати:

- `CanonicalWorldState`;
- `PlannerWorldState`;
- latent state;
- retrieval context.

Але generation не має права породжувати новий truth state лише тому,
що decoder "так сформулював".

## 6. Що саме має робити символьна частина

### 6.1 Канонічні типи і контракти

Символьна частина повинна бути єдиним власником:

- дискретних фактів;
- discrete relation/state/goal forms;
- canonical identifiers;
- typed propositions;
- provenance buckets;
- semantic mode;
- quantifier mode;
- epistemic status;
- world status.

Тільки symbolic має право сказати:

- що є state;
- що є relation;
- що є goal;
- що є generic rule;
- що є obligation;
- що є cited/questioned/hedged/asserted claim.

### 6.2 Verification і contradiction handling

Символьна частина повинна володіти:

- support/conflict evaluation;
- contradiction scopes;
- status transitions;
- branching policy;
- hidden-cause trigger semantics;
- repair-action semantics.

Neuro може подати сигнал.

Але тільки symbolic вирішує:

- `supported`;
- `deferred`;
- `conflicted`;
- `active`;
- `hypothetical`;
- `contradicted`.

### 6.3 Rule lifecycle

Символьна частина повинна бути єдиним власником:

- candidate rule admission;
- promotion to accepted symbolic rule;
- conflict preservation;
- rule retirement;
- rule versioning;
- obligation/generic/rule distinction.

Neuro може:

- згенерувати candidate rule;
- ранжувати candidate rules;
- оцінити plausibility rule surface.

Neuro не може:

- саме приймати правило в KB;
- саме робити rule planner-visible truth;
- саме робити rule memory-persistent.

### 6.4 Memory, planner, world state

Символьна частина повинна бути єдиним власником:

- memory writeback semantics;
- memory recall status semantics;
- what is recallable as fact;
- what is recallable as hypothesis;
- planner-visible action/state surface;
- canonical world-state assembly.

Planner не повинен бачити "сирі neural candidates".

Planner повинен бачити тільки:

- `active` facts;
- дозволені `hypothetical` alternatives;
- explicit repair directives;
- explicit alternative worlds.

## 7. Що саме має робити boundary-шар

Boundary-шар існує для того, щоб не було змішування логічних ролей.

Він повинен бути:

- детермінізованим;
- типізованим;
- provenance-preserving;
- idempotent за однакового input.

Boundary-шар відповідає за:

- source routing;
- segmentation;
- span anchoring;
- structural parsing;
- canonical normalization;
- interlingua building;
- deterministic symbolic lowering;
- conversion proposal -> typed candidate;
- filtering by authority class.

Boundary-шар не вирішує істину.

Boundary-шар вирішує тільки:

- форму;
- тип;
- provenance;
- authority class;
- allowed next stage.

## 8. Жорсткі заборони для нейронної частини

Нейронна частина не має права:

- напряму створювати `active` records;
- напряму створювати `accepted` symbolic rules;
- напряму створювати planner truth;
- напряму створювати memory-persistent factual state;
- тихо редагувати `CanonicalInterlingua`;
- тихо редагувати `GroundingVerificationReport`;
- тихо обходити `world_state_writeback`;
- тихо змішувати proposal score з world-status decision.

Окремо заборонено:

- трактувати decoder output як truth source;
- трактувати saliency як факт;
- трактувати embedding proximity як факт;
- трактувати abduction score як accepted explanation;
- трактувати heuristic extraction як observation-level truth.

## 9. Жорсткі заборони для символьної частини

Символьна частина не має права:

- підміняти learned perception rules вручну там, де потрібна learned representation;
- займатися byte-level fluency modeling;
- підміняти retrieval embeddings ручними lookup-хитрощами як основний механізм;
- ставати surface language generator замість neuro decoder;
- оголошувати новий факт лише тому, що він добре виводиться з rule prior без grounding path;
- вважати свою внутрішню зручність підставою для truth admission.

Символьний шар повинен бути сильним, але не повинен удавати з себе
perception engine.

## 10. Канонічний контракт між neuro і symbolic

### 10.1 Neuro output contract

Будь-який learned або heuristic output повинен входити в boundary-шар
лише як proposal-об'єкт з полями такого класу:

- `proposal_id`
- `proposal_type`
- `source_ref`
- `source_span`
- `segment_index`
- `candidate_payload`
- `confidence`
- `uncertainty`
- `evidence_features`
- `origin`
- `authority_class=proposal`

Якщо output не може бути виражений у такому вигляді,
він не повинен переходити в symbolic truth path.

### 10.2 Symbolic admission contract

Щоб proposal перейшов у symbolic path, він повинен пройти:

1. deterministic typing
2. provenance preservation
3. authority labeling
4. verification
5. status assignment

Лише після цього proposal може стати:

- `hypothetical`;
- `contradicted`;
- `active`;
- `rule_lifecycle_input`.

### 10.3 Canonical actionability contract

Actionable state для planner або memory існує лише тоді, коли об'єкт:

- має stable type;
- має source provenance;
- має explicit status;
- не є heuristic-only shortcut;
- не обійшов verification/writeback.

## 11. Розподіл ownership по підсистемах репозиторію

| Підсистема | Основний owner | Роль |
|---|---|---|
| `omen_scale.py` encoder/latent/saliency/decode | `neuro` | learned perception, latent state, generation |
| `omen_grounding/source_routing.py` | `boundary` | deterministic source classification contract |
| `omen_grounding/text_semantics.py` | `boundary` | deterministic segmentation, normalization, span-safe text semantics |
| `omen_grounding/structural_scene.py` | `boundary` | structural-primary semantic extraction |
| `omen_grounding/heuristic_backbone.py` | `neuro/proposal` | low-authority fallback proposals |
| `omen_grounding/semantic_scene.py` | `boundary` | merge policy, authority filtering, owner map |
| `omen_grounding/interlingua.py` | `boundary` | canonical normalized meaning |
| `omen_grounding/symbolic_compiler.py` | `boundary` | deterministic lowering into symbolic hypotheses and candidate rules |
| `omen_grounding/verification.py` | `symbolic` | explicit support/conflict/status decision |
| `omen_grounding/world_state_writeback.py` | `symbolic` | world-status assignment and repair routing |
| `omen_grounding/memory_policy.py` | `symbolic` | status-aware memory admission |
| `omen_grounding/planner_state.py` | `symbolic` | planner-visible world projection |
| `omen_prolog.py` reasoning core | `symbolic` | KB, unification, proof policy, rule lifecycle |
| `omen_prolog.py` neural helper heads | `neuro/proposal` | learned candidate generation only |
| `omen_symbolic/creative_cycle.py` | `mixed, but proposal-bounded` | produces candidates, not truth |
| `omen_symbolic/world_graph.py` | `symbolic/operational` | canonical world-state carrier |

## 12. Що в поточному коді треба вважати правильним

Уже правильно, що:

- `heuristic_policy.py` явно мітить heuristic sources;
- `symbolic_compiler.py` відфільтровує heuristic candidate rules;
- `verification.py` занижує authority heuristic claims і не дає їм тихо стати `supported`;
- `world_state_writeback.py` не дає heuristic claims стати `active`;
- `planner_state.py` не пропускає heuristic candidate rules в planner contract;
- `SymbolicTaskContext` уже відрізає heuristic candidate rules як direct symbolic rule input.

Це треба посилювати, а не ламати.

## 13. Що в поточному коді треба змінити концептуально

### 13.1 `semantic_scene.py`

Поточний `fallback_primary / hybrid / structural_primary` механізм є корисним,
але має бути ще жорсткішим.

Потрібна норма:

- якщо segment є `structural_primary`, fallback semantic objects не володіють смислом;
- `hybrid` дозволений тільки для явно визначених типів сегментів;
- retained fallback material у `hybrid` має бути narrow overlay, а не другий власник істини.

### 13.2 `SymbolicTaskContext`

Поточний контекст занадто легко виглядає як одна велика миска фактів.

Потрібна норма:

- `saliency_derived_facts` не повинні поводитися як reasoning truth автоматично;
- `net_derived_facts` не повинні поводитися як reasoning truth автоматично;
- `abduced_support_facts` не повинні поводитися як accepted truth автоматично;
- `observed_facts` не повинен маскувати різницю між `observed`, `proposed`,
  `verified`, `active`, `hypothetical`.

Інакше symbolic ядро вимушено живе в напівпрозорому proposal-тумані.

### 13.3 `omen_prolog.py`

`DifferentiableProver` і пов'язані абдуктивні механізми не треба послаблювати.

Але треба чітко зафіксувати:

- neural abduction head є proposal producer;
- trace/contextual abduction є candidate-generation mechanism;
- accepted rule lifecycle лежить не в neural head, а в symbolic admission path.

Якщо learned candidate rule фізично генерується в `omen_prolog.py`,
це не робить її symbolic truth.

### 13.4 Generation path

Generation має залежати від:

- `CanonicalWorldState`;
- planner/world-state contract;
- explicit latent residuals.

Generation не повинна брати raw candidate rules або raw heuristic proposals
як прихований truth source.

## 14. Що на що треба замінити

Нижче наведено жорстку заміну неправильних ментальних моделей.

Замінити:

- `heuristic fact`
на:
- `heuristic proposal`

Замінити:

- `neural fact`
на:
- `neural signal` або `neural proposal`

Замінити:

- `abduced rule already available to reasoning truth`
на:
- `abduced rule candidate pending symbolic lifecycle gate`

Замінити:

- `saliency support fact`
на:
- `saliency-triggered candidate evidence`

Замінити:

- `NET-derived fact`
на:
- `NET-derived proposal record`

Замінити:

- `decoder knows the answer`
на:
- `decoder realizes the answer from canonical state plus latent fluency support`

Замінити:

- `symbolic owns meaning discovery`
на:
- `symbolic owns typed meaning stabilization and truth governance`

## 15. Непорушні інваріанти

У канонічному OMEN завжди мають бути істинними такі правила:

1. `active` факт може виникнути тільки після deterministic symbolic verification/writeback.
2. Heuristic або neural proposal не може напряму породити `active`.
3. Candidate rule не є accepted rule.
4. Decoder output не є truth source.
5. Planner не отримує raw proposal lane.
6. Memory не отримує raw proposal lane як canonical fact.
7. `CanonicalWorldState` не збирається напряму з heuristic-only material.
8. Structural-primary parsing має вищий authority, ніж fallback semantics.
9. Fallback path повинен бути traceable, tagged і degradable.
10. Уся невизначеність повинна бути явною через statuses, а не схованою в merge logic.
11. Neuro ніколи не приймає остаточне epistemic рішення.
12. Symbolic ніколи не підміняє decoder/perception як learned continuous machinery.

## 16. Практичний канон: хто що робить

Формула для майбутніх змін має бути такою.

`Neuro`:

- бачить;
- стискає;
- узагальнює;
- підсвічує;
- шукає;
- пропонує;
- ранжує;
- формулює поверхню.

`Boundary`:

- типізує;
- канонізує;
- маркує authority;
- переносить provenance;
- відсікає нелегальні переходи.

`Symbolic`:

- верифікує;
- класифікує;
- вирішує статус;
- тримає суперечність;
- керує правилами;
- керує planner truth;
- керує memory truth;
- збирає canonical world state.

## 17. Порядок міграції

Щоб ця межа стала не тільки документом, а реальною архітектурою,
наступні зміни мають пріоритет:

1. Відокремити proposal buckets від reasoning truth buckets у `SymbolicTaskContext`.
2. Дотиснути `semantic_scene.py`, щоб `structural_primary` майже завжди перемагав fallback path.
3. Зробити explicit `proposal authority class` для saliency, NET, heuristic, abduction outputs.
4. Не дозволяти candidate rules минати symbolic lifecycle gate.
5. Тримати planner strictly on `PlannerWorldState`.
6. Тримати generation strictly on `CanonicalWorldState` плюс latent realization support.
7. Виносити learned candidate producers логічно в neuro plane навіть тоді,
   коли вони технічно живуть усередині `omen_prolog.py`.

## 18. Кінцева формула

OMEN не повинен бути:

- neural system with symbolic decorations;
- symbolic system that pretends to perceive;
- heuristic system that accidentally promotes guesses to truth.

OMEN повинен бути:

- neural in perception, ranking, proposal and realization;
- symbolic in truth, verification, rule governance, planner ingress and memory semantics;
- deterministic at the boundary between them.
