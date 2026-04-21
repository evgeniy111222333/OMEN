# Повний аудит реалізації OMEN відносно `docs/masterplans`

Дата аудиту: 2026-04-21

## 1. Мета і рамка аудиту

Цей аудит відповідає на чотири питання:

1. Наскільки поточний код реально відповідає канону з `docs/masterplans`.
2. Де репозиторій уже рухається правильно, а де тільки імітує потрібну архітектуру.
3. Чи не почався зсув від AI/self-learning системи до жорстко виписаного алгоритмічного контролю.
4. Наскільки система вже є самонавчальною, автономною і реально готовою до дії.

Нормативна база цього аудиту:

- `docs/masterplans/concept.md`
- `docs/masterplans/DETERMINISTIC_RUNTIME_CONCEPT_UK.md`
- `docs/masterplans/NEURO_SYMBOLIC_BOUNDARY_UK.md`
- `docs/masterplans/GROUNDING_MASTERPLAN.md`
- `docs/masterplans/TRAINING_MASTERPLAN.md`
- `docs/masterplans/SYSTEM_REQUIREMENTS_MASTERPLAN.md`
- `docs/masterplans/DOMAIN_ACTION_REQUIREMENTS_MASTERPLAN.md`

Основні перевірені модулі:

- `omen.py`
- `omen_scale.py`
- `omen_scale_config.py`
- `omen_world_model.py`
- `omen_net_tokenizer.py`
- `omen_prolog.py`
- `omen_grounding/*`
- `omen_symbolic/*`
- `omen_osf.py`
- `omen_osf_planner.py`
- `omen_osf_decoder.py`

Додаткова перевірка:

- `python -m pytest tests/test_symbolic_task_context_world_state.py tests/test_grounding_world_state_writeback.py tests/test_online_symbolic_learning_eval.py -q`
- `python -m pytest tests/test_grounding_scene_pipeline.py tests/test_decoder_path_protocol.py tests/test_osf_protocol.py -q`

Результат локальної верифікації:

- 41 тест пройдено.

Технічна примітка:

- `codegraph` backend був успішно відновлений через `tools/reset-codegraph.ps1`, але сам MCP tool не був змонтований у цій нитці після старту сесії.
- Тому цей аудит зроблено через пряме читання репозиторію, пошук по коду, уже наявні durable memory-нотатки і локальні тести.

## 2. Короткий вердикт

Коротко і без пом’якшень:

- Репозиторій уже дуже добре зібраний як **канонічний runtime-каркас** OMEN.
- Репозиторій уже добре зібраний як **truth-governed symbolic/world-state system**.
- Репозиторій **ще не зібраний як mature learned grounding system**.
- Репозиторій **ще не зібраний як fully autonomous action system**.
- Репозиторій **частково самонавчається**, але це поки вузьке й локальне самонавчання, а не повний lifelong self-improving loop із masterplan.

Найважливіший висновок цього аудиту:

- головний дрейф не в тому, що систему вже почали жорстко вчити "як говорити";
- головний дрейф у тому, що систему поки занадто сильно вчать **як інтерпретувати** через rules/markers/regex/heuristic scoring замість learned grounding;
- тобто surface generation у репо лишається переважно нейронною;
- але meaning formation у grounding-шарі поки значною мірою тримається на алгоритмічному scaffolding.

Простіше:

- **говорити** система поки не перетворена на canned-template машину;
- **розуміти** вона поки ще занадто залежить від hand-built deterministic/heuristic machinery.

## 3. Найсильніші зони відповідності masterplan

### 3.1 Один канонічний runtime справді зафіксований

Це сильна сторона.

Підтвердження:

- `omen.py` явно виставляє `OMEN = OMENScale`.
- `omen_scale.py` має `_enforce_canonical_stack(...)`.
- `tests/test_canonical_stack_protocol.py` перевіряє, що канонічним рантаймом є `omen_scale.OMENScale`, а `omen_v2.py` позначений як legacy.

Висновок:

- вимога з `concept.md` про єдиний головний runtime виконана добре.

### 3.2 Byte-first і NET-first реалізовані серйозно

Це одна з найкраще реалізованих частин концепту.

Підтвердження:

- `omen_scale_config.py`: `vocab_size=256`.
- `omen_net_tokenizer.py`: `ByteContextEncoder` працює на сирих UTF-8 байтах.
- `omen_net_tokenizer.py` прямо формулює NET як заміну fixed subword tokenization.
- `omen_scale.py` за замовчуванням примусово вмикає `net_enabled=True`, якщо не дозволено noncanonical ablation.

Висновок:

- вимога `Byte-first, а не BPE-first` виконана концептуально правильно.

### 3.3 Graph-primary world state реально існує, а не лише описаний в документах

Підтвердження:

- `omen_symbolic/world_graph.py` містить `WorldGraphState`, `WorldGraphBatch`, `CanonicalWorldState`.
- `omen_scale.py` збирає `CanonicalWorldState`, де graph state є primary, а dense state є derived view.
- `omen_world_model.py` має реальний `WorldRNN`, а не лише декларацію.

Висновок:

- вимога `world-state-first` і `graph-primary` реалізована сильно.

### 3.4 Символьне ядро не декоративне

Підтвердження:

- `omen_prolog.py` містить `KnowledgeBase` і `DifferentiableProver`.
- `omen_symbolic/executor.py` виконує чистий symbolic execution only over verified rules.
- `omen_scale.py` реально подає `task_context.reasoning_facts()` у prover.

Висновок:

- masterplan-вимога про справжній symbolic substrate із truth maintenance виконана добре.

### 3.5 Truth-governance у grounding path зараз одна з найздоровіших частин системи

Підтвердження:

- `omen_grounding/world_state_writeback.py` не дає heuristic claims стати `active`.
- `omen_grounding/planner_state.py` відфільтровує heuristic candidate rules.
- `omen_grounding/memory_hints.py` допускає в grounding-memory тільки `grounding_world_state` і `grounding_ontology`.
- `omen_prolog.py` у `SymbolicTaskContext.reasoning_facts()` використовує лише admitted symbolic material.
- `omen_scale.py` у `_program_anchor_facts(...)` віддає пріоритет `grounding_world_state_active_facts` і `grounding_ontology_facts`.

Висновок:

- вимоги `DETERMINISTIC_RUNTIME_CONCEPT_UK.md` і `NEURO_SYMBOLIC_BOUNDARY_UK.md` про truth-maintenance, planner ingress і memory admission тут виконані добре.

### 3.6 Generation path не виглядає як жорстко заскриптоване "як говорити"

Підтвердження:

- `omen_scale.py` має нейронний `TokenDecoder`.
- `omen_osf.py` і `omen_osf_decoder.py` будують генерацію через intent/planning/template/decoder stack.
- не знайдено окремого rule engine, який би на рівні surface realization директивно задавав стиль відповіді через canned phrases.

Уточнення:

- в OSF є template layer, але це structure/planning template, а не набір готових людських формулювань.
- тобто репозиторій не перетворив surface generation на жорсткий "банально як йому говорити" движок.

Висновок:

- у generation немає головного концептуального провалу;
- головний провал зараз не в speaking, а в grounding/understanding.

## 4. Найбільші проблеми і розриви з masterplan

У попередній версії аудиту цей блок був коротким. Нижче йде вже повна карта розривів у форматі:

- як зараз;
- як має бути в ідеалі згідно `docs/masterplans`;
- точний розрив;
- чому це критично;
- що саме треба міняти.

### 4.1 Learned grounding backbone фактично відсутній

Як зараз:

- `omen_grounding/backbone.py` містить лише `SemanticGroundingBackbone` protocol.
- у `omen_grounding/*` не знайдено реального learned `nn.Module`-подібного backbone для L1-L8 grounding.
- у `omen_grounding/semantic_scene.py` робочий шлях за замовчуванням іде через `backbone or HeuristicFallbackSemanticBackbone()`.
- отже реальна семантична робота зараз виконується або deterministic structural path, або heuristic fallback path.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` розділ `8. Grounding Learning Contract` grounding має бути trainable багаторівневим стеком:
  - L1 typed perception and segmentation;
  - L2 structural grounding;
  - L3 linguistic grounding;
  - L4 semantic scene graph;
  - L5 canonical interlingua;
  - L6 probabilistic symbolic compiler;
  - L7 verification and repair;
  - L8 world-state and memory writeback.
- за `GROUNDING_MASTERPLAN.md` фінальна траєкторія має бути:
  - `carrier bytes -> typed perception -> structural grounding -> linguistic grounding -> semantic scene graph -> canonical interlingua -> deterministic symbolic lowering -> verification/repair -> world state`.
- за `NEURO_SYMBOLIC_BOUNDARY_UK.md` нейронна частина має володіти perception, multilingual abstraction, proposal generation, saliency і ranking, але не бути truth authority.

Точний розрив:

- trainable learned semantics як окремий grounding front-end у репо відсутній;
- отже роль, яку в концепті має нести learned perception stack, зараз несуть deterministic helpers і heuristic fallback;
- через це система може бути сильною в канонічному runtime і symbolic truth-governance, але слабкою саме в перетворенні сирого тексту на справді learned semantic evidence.

Чому це критично:

- це головний блокер для всього, що в masterplan названо self-learning grounding;
- без learned grounding backbone не з’являється нормальний route для `A_ground`;
- без нього heuristics не можуть бути нормально витіснені, бо нема чим;
- без нього multilingual convergence, hidden-slot exposure, event persistence і cross-domain semantics лишаються переважно деклараціями.

Що саме треба міняти:

- ввести реальний trainable grounding backbone як окремий перший-class module, а не тільки protocol;
- розбити його хоча б на L1-L5 trainable carriers із явними виходами proposal-типу;
- прив’язати його до explicit losses із `A_ground`;
- зробити так, щоб `HeuristicFallbackSemanticBackbone` був не default carrier, а аварійний bounded fallback із метрикою activation rate.

### 4.2 `source_routing.py` бере на себе занадто багато семантичної дискримінації

Як зараз:

- у `omen_grounding/source_routing.py` є великі marker-набори:
  - `_ROUTER_LANGUAGE_MARKERS`
  - `_INSTRUCTIONAL_MARKERS`
  - `_LEGAL_MARKERS`
  - `_MEDICAL_MARKERS`
  - `_NARRATIVE_MARKERS`
- routing і parser selection суттєво спираються на lexical/regex-style markers.
- це означає, що boundary-шар зараз не лише маршрутизує evidence, а й фактично виконує частину meaning classification вручну.

Як має бути в ідеалі за masterplan:

- за `GROUNDING_MASTERPLAN.md` `Source Router Contract` повинен видавати:
  - modality distribution;
  - subtype distribution;
  - language distribution;
  - parser candidates;
  - verification-path candidates;
  - ambiguity flags.
- router не повинен бути final authority on truth і не повинен бути місцем, де вручну закодована глибока natural-language semantics.
- за `TRAINING_MASTERPLAN.md` L1 має бути trainable:
  - language identification;
  - modality classification;
  - subtype classification;
  - segment boundary detection;
  - parser candidate ranking;
  - ambiguity detection.
- за heuristic retirement policy евристики тут дозволені лише як coarse routing priors і deterministic format recognition.

Точний розрив:

- саме existence deterministic router не є проблемою;
- проблема в тому, що semantic burden routing-шару зараз завеликий;
- він уже не тільки типізує шлях, а частково визначає смислову природу сегмента через hand-built markers.

Чому це критично:

- marker-heavy routing погано масштабується на real messy documents;
- mixed-language input, scientific prose, dialogue, code-plus-comments і atypical domain text будуть деградувати нерівномірно;
- така архітектура створює тиху залежність від curated lexical triggers, а не від learned evidence.

Що саме треба міняти:

- залишити router deterministic за контрактом, але зменшити його semantic ownership;
- перенести parser ranking, ambiguity retention і subtype confidence у learned L1 module;
- лишити в `source_routing.py` тільки:
  - source metadata ingestion;
  - deterministic format recognition;
  - coarse priors;
  - provenance-preserving dispatch.

### 4.3 `text_semantics.py` досі є переважно deterministic/heuristic проміжним шаром, а не learned linguistic grounding

Як зараз:

- у файлі видно суттєвий обсяг regex/pattern логіки:
  - `_STATE_PAIR_RE`
  - `_NEGATION_PATTERNS`
  - `_extract_speaker_turn_unit`
- наприкінці файлу `extract_relation_hints`, `extract_goal_hints`, `extract_entity_hints`, `extract_event_hints` просто делегують у `heuristic_backbone.py`.
- тобто `text_semantics.py` зараз частково виконує роль обгортки над hand-built text hints, а не повноцінного learned linguistic grounding layer.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` L3 linguistic grounding має вчитись:
  - multilingual token and lemma formation;
  - morphology;
  - POS and dependency structure;
  - clause decomposition;
  - mention detection;
  - coreference candidate generation;
  - discourse relation detection.
- за `SYSTEM_REQUIREMENTS_MASTERPLAN.md` система має:
  - підтримувати multilingual grounding;
  - не бути English-only;
  - тримати parser disagreement;
  - не зводити все рано до одного жорсткого тлумачення.
- за `GROUNDING_MASTERPLAN.md` dialogue, scientific text, mixed docs і instructions повинні мати differentiated grounding behavior.

Точний розрив:

- поточний шар добре виконує span-safe normalization і деяку bounded deterministic preprocessing-функцію;
- але він не є learned multilingual linguistic grounding у masterplan-сенсі;
- звідси випливає слабкість у morphology-heavy мовах, discourse structure, stable coreference і nested attribution.

Чому це критично:

- без сильного L3 система не може стабільно виділяти хто що сказав, до чого відноситься negation, де general claim, де instance, де citation, де instruction, де quoted content;
- саме тут і виникає головна різниця між “парсить текст” і “справді grounding-ить meaning”.

Що саме треба міняти:

- залишити в `text_semantics.py` лише boundary-safe normalization, span tracking і deterministic sanitation;
- окремо додати learned linguistic grounding layer із typed proposal outputs;
- зробити всі entity/relation/event/goal hints не прямою функцією regex/heuristics, а виходом learned proposal heads з явним authority class `proposal`.

### 4.4 `HeuristicFallbackSemanticBackbone` зараз не просто fallback, а фактичний default semantic carrier

Як зараз:

- `omen_grounding/semantic_scene.py` автоматично бере `HeuristicFallbackSemanticBackbone`, якщо learned backbone не передано;
- `omen_grounding/heuristic_backbone.py` робить велику частину реальної semantic work:
  - entity accumulation;
  - relation proposals;
  - event proposals;
  - goal proposals;
  - claim support;
  - coreference-like behavior.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` heuristics допустимі лише як temporary bounded support;
- acceptable heuristic roles:
  - coarse routing priors;
  - parser proposal;
  - deterministic format recognition;
  - diagnostics;
  - fallback extraction.
- unacceptable heuristic roles:
  - owning deep natural-language meaning;
  - final multilingual equivalence judgement;
  - final persistent entity resolution;
  - final verified world writes;
  - planner-state creation from ambiguous prose.

Точний розрив:

- формально heuristics у репо вже позначені як fallback/low-authority;
- але по факту semantic fallback path без них не існує;
- тому вони зараз не bounded support, а production workhorse для частини natural language grounding.

Чому це критично:

- поки fallback є semantic workhorse, learned grounding не відбувається навіть концептуально;
- у системи нема стимулу мігрувати від heuristics до trainable modules, бо heuristics уже закривають головну operational діру;
- це прямий конфлікт із heuristic retirement policy.

Що саме треба міняти:

- зробити activation rate heuristics окремою метрикою;
- відокремити `heuristic_backbone.py` як degrade path із жорстким telemetry;
- не дозволяти йому бути implicit default для broad natural language semantics;
- поступово замінювати його по задачах:
  - route;
  - entity persistence;
  - coreference;
  - event/goal extraction;
  - multilingual equivalence.

### 4.5 L4 semantic scene graph і L5 canonical interlingua реалізовані неповно

Як зараз:

- у repo є structural scene building, semantic scene merging і далі lowering у symbolic path;
- однак не видно повноцінного learned scene graph construction, який би системно навчався на:
  - entity-role assignment;
  - modality/polarity;
  - quantifier handling;
  - generic-vs-instance separation;
  - claim attribution;
  - hidden-slot exposure.
- не видно повного learned L5 interlingua contract для cross-language paraphrase convergence і stable predicate inventory.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` L4 має вчитися:
  - event extraction;
  - entity-role assignment;
  - modality and polarity detection;
  - quantifier separation;
  - claim attribution;
  - explanation and hidden-slot exposure.
- L5 має вчитися:
  - language-invariant meaning collapse when meaning is equivalent;
  - preservation of polarity, modality, quantification, epistemic stance;
  - canonical predicate and role naming stability.
- за `GROUNDING_MASTERPLAN.md` equivalent meanings across languages should converge into the same canonical interlingua when appropriate.

Точний розрив:

- у репо є good symbolic discipline after grounding;
- але сам шлях від raw text до rich scene graph/interlingua поки не виглядає як повноцінний learned stack;
- відповідно generic/instance, cited/asserted, quoted/narrated, equivalent/different meaning поки більше залежать від manual decomposition, ніж від matured semantic collapse policy.

Чому це критично:

- без L4/L5 система не стає по-справжньому multilingual and cross-domain;
- без них memory dedup, ontology growth, symbolic compiler precision і planner input quality залишаються крихкими;
- саме тут вирішується, чи однаковий зміст у двох різних формулюваннях справді сходиться до одного canonical meaning.

Що саме треба міняти:

- додати explicit scene graph objects і interlingua supervision path;
- ввести окремі метрики:
  - entity persistence;
  - event extraction;
  - causal link accuracy;
  - temporal link accuracy;
  - paraphrase convergence;
  - translation invariance;
  - epistemic preservation.

### 4.6 `symbolic_compiler.py` ще не виглядає як повний probabilistic symbolic compiler з довгим життям альтернатив

Як зараз:

- symbolic lowering існує і добре вписаний у truth-governed path;
- є candidate facts/rules, є status-aware downstream gating;
- але не видно повного mature contract, де багатозначність системно живе достатньо довго як competing interpretations, а не швидко спрощується до однієї форми.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` L6 compiler повинен:
  - retain multiple plausible interpretations;
  - compile candidate facts and candidate rules;
  - attach support sets and provenance;
  - calibrate confidence and defer weak claims.
- за `GROUNDING_MASTERPLAN.md` output compiler-а повинен містити:
  - candidate Horn facts;
  - candidate clauses;
  - deferred hypotheses;
  - support links;
  - contradiction links;
  - confidence scores.
- за `GROUNDING_MASTERPLAN.md` hard early collapse є explicit failure mode.

Точний розрив:

- deterministic lowering є;
- сильний truth gate є;
- але probabilistic survival of alternatives виглядає ще недостатньо центральним принципом;
- тобто система добре вміє не пустити слабке в `active`, але ще не настільки добре вміє зберегти кілька інтерпретацій як first-class competing hypotheses.

Чому це критично:

- без цього страждає dialogue ambiguity, contradictory reports, incident analysis, scientific disagreement, planning under uncertainty;
- система ризикує бути обережною, але недостатньо багатогіпотезною.

Що саме треба міняти:

- підсилити compiler так, щоб deferred hypotheses і support links були центральними артефактами, а не побічними;
- окремо міряти alternative-hypothesis survival quality;
- прив’язати compiler decisions до planner branch quality і later verification outcomes.

### 4.7 Verification зроблений правильно як truth gate, але ще не став mature learned+formal hybrid verifier

Як зараз:

- `omen_grounding/verification.py` обчислює `support` і `conflict` через вручну підібрані коефіцієнти;
- у scoring входять:
  - confidence;
  - provenance;
  - duplicate support;
  - document alignment;
  - structural alignment;
  - dialogue support;
  - citation support;
  - speaker attribution;
  - epistemic support/conflict.

Як має бути в ідеалі за masterplan:

- за `GROUNDING_MASTERPLAN.md` verification має повертати:
  - accepted;
  - proposed;
  - contradicted;
  - uncertain;
  - clarification-needed.
- за `TRAINING_MASTERPLAN.md` L7 verifier має вчитися:
  - support/conflict estimation;
  - contradiction localization;
  - repair action selection;
  - hidden-cause trigger policy.
- primary supervision має приходити з:
  - execution outcomes;
  - world-model consistency;
  - parser agreement;
  - memory corroboration;
  - contradiction sets;
  - synthetic conflict benchmarks.

Точний розрив:

- status machine і object-level truth gate вже хороші;
- але calibration майже повністю ручна;
- не видно окремого learned calibration layer, який би вчився на future correctness, repair success і downstream utility.

Чому це критично:

- якщо verifier не калібрується від реальних outcome-ів, то його confidence ризикує лишатися “правдоподібним”, але не добре зв’язаним із реальною майбутньою коректністю;
- це б’є по online adaptation, memory writes, planner constraints і contradiction handling.

Що саме треба міняти:

- зберегти deterministic final status assignment;
- але перед ним додати learned support/conflict estimators і learned calibrators;
- міряти:
  - verified vs contradicted calibration;
  - repair success rate;
  - contradiction localization quality;
  - clarification utility.

### 4.8 World-state writeback і memory writeback дисципліновані, але ще не дотягують до повного L8 contract

Як зараз:

- `omen_grounding/world_state_writeback.py` уже добре захищає `active` world state від heuristic leakage;
- `omen_grounding/memory_hints.py` і дотичні шари вже обмежують, що взагалі може потрапляти у grounding-memory;
- planner ingress теж уже status-aware.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` L8 має навчатися вирішувати:
  - create vs merge vs defer vs reject;
  - preserve provenance and confidence;
  - protect memory from pollution;
  - update entity identity over time.
- за `SYSTEM_REQUIREMENTS_MASTERPLAN.md` memory retrieval і writes повинні матеріально покращувати future grounding, reasoning, planning, generation;
- write policy має оптимізувати не питання “чи можна зберегти”, а питання “чи принесе цей write майбутню користь”.

Точний розрив:

- truth-governance уже сильне;
- але learned write policy як utility-governed long-horizon module майже не видно;
- writeback зараз більше нагадує дисциплінований guardrail, ніж зрілий learning-driven future-utility controller.

Чому це критично:

- без цього система або буде надто консервативною і нічого не накопичуватиме, або накопичуватиме knowledge without measured future utility;
- це прямо впливає на тезу про “повністю самонавчальну систему”.

Що саме треба міняти:

- ввести write decision learning з використанням contradiction outcomes і future utility feedback;
- окремо міряти merge accuracy, pollution penalty, future-utility gain;
- зв’язати writeback із long-horizon entity persistence, а не тільки локальною admissibility.

### 4.9 Grounding training contract із `TRAINING_MASTERPLAN.md` у коді реалізований дуже нерівномірно

Як зараз:

- у `omen_scale.py` є загальний training loop і багато loss terms для NET, world, symbolic, generation, meta;
- `set_semantic_grounding_backbone(...)` існує як hook;
- але реального `A_ground` як first-class objective family у working grounding stack немає;
- немає trainable grounding adapters;
- немає повного curriculum від L1 до L8.

Як має бути в ідеалі за masterplan:

- `A_ground = w_route L_route + w_struct L_struct + w_ling L_ling + w_scene L_scene + w_inter L_inter + w_compile L_compile + w_verify L_verify + w_write L_write + w_calib L_calib`
- curriculum має бути етапний:
  - Stage 1B typed perception foundation;
  - Stage 1C structural and linguistic foundation;
  - Stage 1D scene and interlingua foundation;
  - Stage 2B grounding-symbolic bridge training;
  - Stage 3B guarded online grounding adaptation.
- adaptive weight scheduling має залежати від subsystem maturity, grounding conflict rate, calibration health.

Точний розрив:

- training story для ядра OMEN уже є;
- training story для grounding front-end у репо ще здебільшого декларативна;
- тобто весь репозиторій не однаково дійшов до masterplan maturity.

Чому це критично:

- без explicit objective grounding завжди буде “покращуватися випадково” або не покращуватися взагалі;
- без curriculum grounding не пройде шлях від scaffold до reliable learned stack;
- без adaptive weighting не видно, як система має перекидати тиск навчання саме туди, де semantic failures народжуються.

Що саме треба міняти:

- вивести grounding losses у явний first-class training API;
- вести окрему статистику по L1-L8 modules;
- додати graduation gates, після яких weaker heuristics реально відключаються, а не залишаються назавжди.

### 4.10 Online grounding adaptation, shadow mode і heuristic retirement policy практично не реалізовані

Як зараз:

- online adaptation у repo існує переважно для world model і symbolic parts;
- grounding update path як окремий guarded online subsystem не видно;
- немає явного shadow update mode, reversible write mode, bounded-trust online grounding mode.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` online можуть оновлюватися:
  - memory content;
  - memory read/write policies;
  - rule statuses;
  - selected symbolic proposal heads;
  - EMC policies;
  - confidence calibrators;
  - grounding adapters.
- обов’язкові safe modes:
  - shadow update mode;
  - propose-only mode;
  - reversible write mode;
  - bounded-trust online learning mode;
  - human-review escalation mode.
- retirement policy вимагає для кожної heuristics:
  - isolated implementation;
  - explicit fallback status;
  - measurable activation rate;
  - measurable disagreement with learned/formal modules;
  - replaceability by trained modules.

Точний розрив:

- heuristics ізольовані й мічені непогано;
- але механізму їх планомірного витіснення нема;
- немає також online grounding loop, який поступово забирав би роботу в heuristics і доводив learned modules до production maturity.

Чому це критично:

- без retirement policy heuristics стають вічним фундаментом;
- без shadow mode і reversible updates будь-яка майбутня online grounding adaptation буде або надто ризикованою, або відключеною;
- це прямо підрізає ідею safe self-improvement.

Що саме треба міняти:

- додати per-heuristic telemetry;
- ввести learned-vs-heuristic disagreement dashboards;
- зробити explicit switch criteria, коли heuristic path demotes itself to narrow fallback only;
- ввести shadow evaluation для grounding proposals before memory/world write impact.

### 4.11 Тестова культура досі сильно зміщена в noncanonical ablation mode

Як зараз:

- по `tests/*` зафіксовано:
  - `allow_noncanonical_ablation = True`: 59 входжень;
  - `net_enabled = False`: 59 входжень;
  - `osf_enabled = False`: 61 входження;
  - `net_enabled = True` або `osf_enabled = True`: лише 3 входження.
- частина ключових тестів свідомо перевіряє спрощений режим без канонічного stack.

Як має бути в ідеалі за masterplan:

- за `concept.md` і `TRAINING_MASTERPLAN.md` канонічний runtime є один, а не факультативний;
- за graduation gates мають існувати окремі subsystem metrics і final system gates;
- canonical path повинен бути не лише “офіційно правильним”, а й найсильніше покритим verification surface.

Точний розрив:

- код уже canonical-first;
- тестовий тиск поки що значною мірою noncanonical-first;
- це значить, що repo може бути архітектурно правильним на папері, але недоперевіреним у найбільш важливому operational path.

Чому це критично:

- так народжується прихована двошаровість:
  - один stack для документів;
  - інший stack для реально стабільних тестів.
- це сповільнює будь-яку серйозну міграцію до canonical-only maturity.

Що саме треба міняти:

- перенести більшість subsystem тестів на canonical runtime;
- залишити ablation only для explicit diagnostics;
- ввести окремі suites:
  - canonical grounding stack;
  - canonical NET+world+symbolic;
  - canonical generation/OSF;
  - online adaptation safety;
  - heuristic retirement progress.

### 4.12 Дрейф із AI на алгоритми є, але не там, де часто інтуїтивно здається

Як зараз:

- generation path лишається переважно нейронним:
  - `omen_scale.py` має `TokenDecoder`;
  - `omen_osf.py` та `omen_osf_decoder.py` працюють як synthesis/planning/generation stack;
  - окремого canned-speech engine не знайдено.
- зате grounding ingress дуже насичений:
  - markers;
  - regex;
  - heuristic extraction;
  - rule-tuned support/conflict scoring.

Як має бути в ідеалі за masterplan:

- за `NEURO_SYMBOLIC_BOUNDARY_UK.md`:
  - neuro owns perception, proposal, ranking, saliency, retrieval, surface realization;
  - boundary owns deterministic typing, canonical normalization, provenance preservation;
  - symbolic owns truth, verification, rule governance, planner ingress and memory semantics.
- unacceptable heuristic role: owning deep natural-language meaning.

Точний розрив:

- system speaking ще не “закодоване вручну”;
- system understanding усе ще занадто scaffolded manually;
- отже дрейф реально не у “як йому говорити”, а у “як йому сказати, що саме значить вхід”.

Чому це критично:

- якщо speaking neural, а understanding heuristic-heavy, система може виглядати розумною на виході, але залишатися крихкою саме там, де формується meaning;
- це найнебезпечніший вид псевдозрілості.

Що саме треба міняти:

- зберегти нинішній generation ownership;
- перенести semantic burden з heuristic ingress на learned grounding;
- не посилювати ручні meaning rules там, де masterplan вимагає learned semantics.

### 4.13 Самонавчання вже існує, але воно фрагментарне, а не повне lifelong

Як зараз:

- є `omen_train_code.py` зі Stage 1 / Stage 2;
- є `train_epoch_scale(...)`;
- є `_maybe_eval_world_self_update(...)` у `omen_scale.py`;
- є `_online_train_step(...)` у `omen_symbolic/ontology_engine.py`;
- у symbolic layer є continuous cycle, abduction, induction-related updates.

Як має бути в ідеалі за masterplan:

- за `TRAINING_MASTERPLAN.md` не повинно бути hard wall between train and think;
- Stage 3 включає:
  - guarded online symbolic adaptation;
  - guarded online grounding adaptation;
  - lifelong episodic learning.
- система має покращуватися через:
  - interaction;
  - execution;
  - contradiction;
  - memory;
  - world-model mismatch;
  - calibration feedback.

Точний розрив:

- world self-update є;
- online symbolic growth є;
- grounding adaptation немає;
- subsystem maturity gates, replay/shadow training і bounded drift orchestration майже не видно.

Чому це критично:

- без grounding adaptation самонавчання не торкається головного semantic bottleneck;
- без safe orchestration будь-який майбутній self-improvement або буде слабким, або буде небезпечним для epistemic stability.

Що саме треба міняти:

- додати grounding adapters і calibrators як online-updatable components;
- з’єднати contradiction outcomes, memory utility і verifier calibration з online updates;
- ввести subsystem maturity gates, після яких дозволяються сильніші online зміни.

### 4.14 Planner runtime є, але completed autonomy і реального agent loop поки нема

Як зараз:

- є planner substrate;
- є `planner_state`, uncertainty markers, repair directives, alternative worlds;
- є internal symbolic executor;
- є OSF planner;
- але `omen_osf_planner.py` працює з внутрішнім operator library, де `OP_TYPES = ["define", "call", "assign", "return", "branch", "loop", "import", "yield"]`;
- не знайдено реального external `act()`/`execute()` substrate.

Як має бути в ідеалі за masterplan:

- за `SYSTEM_REQUIREMENTS_MASTERPLAN.md` planner має:
  - consume grounded state;
  - support goals, constraints, branch alternatives;
  - plan under uncertainty;
  - repair/re-plan on failure.
- за `DOMAIN_ACTION_REQUIREMENTS_MASTERPLAN.md` система має:
  - convert grounded understanding into correct, useful, revisable action;
  - preserve provenance for consequential action;
  - know when to defer, ask clarification, or branch;
  - improve action policies through verified outcomes.
- за `GROUNDING_MASTERPLAN.md` external integrations мають бути first-class:
  - document loaders;
  - code repositories;
  - schema registries;
  - telemetry/log streams;
  - evaluation harnesses;
  - simulation environments;
  - planner executors.

Точний розрив:

- architecture for planning already exists;
- architecture for acting in external environment almost does not;
- отже repo вже побудував cognitive planning surface, але ще не побудував operational action surface.

Чому це критично:

- без зовнішнього `plan -> act -> observe -> revise` циклу важко назвати систему “вільною до дій” у сильному сенсі;
- autonomy без action substrate лишається внутрішньою симуляцією, а не повним агентним циклом.

Що саме треба міняти:

- додати explicit external action interface;
- зв’язати його з grounded state, verifier constraints і planner assumptions;
- писати verified execution outcomes назад у memory/world state;
- окремо тренувати action policy not from eloquent plans, but from real verified outcome loops.

### 4.15 Domain-action masterplan поки підтриманий лише частково

Як зараз:

- для reasoning, coding, analysis і structured tasks repo already has strong internal abstractions;
- але зовнішньо-доменна операційна спроможність поки не виглядає завершеною:
  - incident response;
  - operations;
  - multi-stage procedures;
  - environment-reactive planning;
  - verified external execution.

Як має бути в ідеалі за masterplan:

- за `DOMAIN_ACTION_REQUIREMENTS_MASTERPLAN.md` система повинна в реальних доменах:
  - відрізняти observation, instruction, claim, plan, proposal, warning, verified result;
  - декомпозувати задачі у grounded subgoals;
  - підтримувати revision when premises change;
  - maintain domain-specific memory that improves future work;
  - learn domain-specific action abstractions from verified outcomes.
- у coding/SE вона має рухатись від symptom -> cause -> fix -> verification;
- в incident/ops не давати ранню overconfident narrative;
- у planning/procedures не перетворювати ambiguous procedure на fake determinism.

Точний розрив:

- на рівні внутрішньої архітектури багато передумов уже є;
- на рівні зовнішньої domain-capable action loop реалізація поки неповна;
- тобто repo сильний як внутрішній neuro-symbolic runtime, але ще не як finalized domain actor.

Чому це критично:

- це саме те місце, де користувач бачить різницю між “дуже розумний внутрішній runtime” і “система, яка реально може працювати в полі”.

Що саме треба міняти:

- з’єднати planner, world state, verification і execution outcomes в один operational loop;
- робити domain memory не просто storage, а source of reused verified procedures;
- міряти success не тільки по local reasoning correctness, а по scenario completion rate, planner success і long-horizon stability.

### 4.16 Підсумковий жорсткий висновок по розривах

Як зараз:

- repo уже сильний як:
  - canonical runtime;
  - graph-primary world-state system;
  - symbolic truth-governed system;
  - planner-safe neuro-symbolic architecture.
- repo ще слабкий як:
  - learned grounding system;
  - lifelong grounding-adaptive system;
  - real-world autonomous action system.

Як має бути в ідеалі:

- grounded meaning має народжуватися в learned perception stack;
- symbolic core має керувати truth, status, planner ingress, memory semantics;
- heuristics мають бути вимірюваним тимчасовим scaffold;
- self-learning має охоплювати grounding, verification calibration, memory utility і action policy;
- autonomy має включати реальний зовнішній act-observe-revise loop.

Точний системний висновок:

- найбільший розрив репозиторію не в world model, не в symbolic substrate і не в decoder;
- найбільший розрив у front-end meaning formation, у його trainability і в переході від внутрішнього reasoning до зовнішнього action loop.

## 5. Аудит: чи не почали переносити з AI на алгоритми

## 5.1 Коротка відповідь

Так, але головний зсув іде не в surface generation, а в semantic ingress.

Точне формулювання:

- репозиторій не перетворився на алгоритмічну canned-response систему;
- але grounding-периметр зараз дійсно занадто залежить від rules, markers, regex і hand-tuned scoring;
- тобто speaking лишається переважно learned/neural, а understanding ще надто scaffolded.

## 5.2 Де AI/learned підхід реально живий

- `omen_net_tokenizer.py`
- `omen_perceiver.py`
- `omen_world_model.py`
- `omen_emc.py`
- `omen_osf.py`
- `omen_osf_planner.py`
- `omen_osf_decoder.py`
- `omen_prolog.py` у частині differentiable/symbolic learning support
- `omen_symbolic/ontology_engine.py`
- `omen_symbolic/hypergraph_gnn.py`

Тут уже є реальні trainable modules, gradients, optimizers, online updates і learned policies.

## 5.3 Де алгоритми домінують сильніше, ніж дозволяє ідеал

- `omen_grounding/source_routing.py`
- `omen_grounding/text_semantics.py`
- `omen_grounding/heuristic_backbone.py`
- `omen_grounding/verification.py`
- `omen_grounding/structural_scene.py`
- частково `omen_grounding/symbolic_compiler.py`
- частково `omen_symbolic/ontology_engine.py` у fallback candidate generation

Найсильніші симптоми:

- marker tables;
- regex parsing;
- hand-built semantic hint extraction;
- hand-built support/conflict coefficients;
- heuristic fallback як default semantic carrier;
- недостатня survival-політика для ambiguous interpretations.

## 5.4 Що masterplan вважає прийнятним, а що вже неприйнятним

Прийнятно:

- coarse routing priors;
- deterministic format recognition;
- parser proposal;
- diagnostics;
- bounded fallback extraction.

Неприйнятно як кінцевий стан:

- owning deep natural-language meaning;
- final multilingual equivalence judgement by heuristics;
- final persistent entity resolution by heuristics;
- planner state from ambiguous prose;
- verified world writes з heuristic-only semantics.

Поточний стан репо:

- guardrails уже правильні;
- truth governance уже правильне;
- але heuristic roles усе ще ширші, ніж дозволяє mature target.

## 5.5 Найважливіший висновок: дрейф іде не в "як говорити", а в "як зрозуміти"

Як зараз:

- decoder і OSF не виглядають як canned-language engine;
- натомість routing/semantics/verification занадто багато вирішують rules-first способом.

Як має бути в ідеалі:

- neuro повинен бачити, абстрагувати, пропонувати, ранжувати і формулювати;
- boundary має лише типізувати і переносити provenance;
- symbolic має вирішувати status, contradiction, truth admission, planner visibility.

Отже:

- **не головна проблема**: що систему вже вчать “банально як їй говорити”;
- **головна проблема**: що систему ще занадто вчать “банально як їй вирішувати meaning class, route і semantic support”.

## 5.6 Це вже критичний збій чи ще прийнятний scaffold

Поки що це ще прийнятний scaffold, але вже на межі допустимого.

Чому ще не критичний збій:

- heuristics ізольовані;
- вони мічені як fallback/low-authority;
- world/planner/memory truth path уже добре захищений.

Чому це вже майже критично:

- learned grounding path і досі не забрав semantic burden;
- отже scaffold затримався в ролі permanent workhorse;
- а це прямо суперечить `GROUNDING_MASTERPLAN.md` і `TRAINING_MASTERPLAN.md`.

## 6. Аудит самонавчання

## 6.1 Що реально є зараз

У коді вже існує не нульове self-learning/adaptation behavior.

Підтвердження:

- `omen_train_code.py` має Stage 1 / Stage 2 training loop;
- `omen_scale.py` має `train_epoch_scale(...)`;
- `omen_scale.py` має `_maybe_eval_world_self_update(...)`, який оновлює `world_rnn` у eval/generate path;
- `omen_symbolic/ontology_engine.py` має `_online_train_step(...)`;
- `omen_prolog.py` має continuous symbolic cycle, abduction і induction-related learning surfaces.

Це означає:

- система вже не purely static;
- система вже не purely offline-trained;
- система вже має окремі механізми competence accretion.

## 6.2 Як має виглядати повне self-learning у masterplan-сенсі

За masterplan повна самонавчальність означає не просто “є кілька online updates”, а весь замкнений цикл:

- grounding learns from contradiction, repair, cross-language equivalence і memory corroboration;
- verifier calibrates itself on future correctness;
- memory write policy learns future utility;
- symbolic layer learns rule proposal and rule lifecycle;
- EMC learns compute allocation;
- action policy learns from verified execution outcomes;
- online updates ідуть у shadow/reversible/bounded-trust modes.

Тобто в ідеалі система повинна поліпшувати:

- розуміння;
- верифікацію;
- пам’ять;
- reasoning policy;
- planning policy;
- action policy.

## 6.3 Точний розрив між поточним станом і повною самонавчальністю

Що вже є:

- world self-update;
- online symbolic training;
- continuous symbolic cycle;
- training curriculum для core runtime.

Що поки відсутнє або слабке:

- online grounding adaptation;
- trainable grounding backbone;
- explicit grounding calibration layer;
- shadow mode для grounding/self-update;
- reversible write orchestration;
- subsystem maturity gates;
- action-policy learning від verified external outcomes.

Найжорсткіше формулювання:

- система вчиться там, де вже є trainable core;
- система майже не вчиться там, де зараз знаходиться головний semantic bottleneck.

## 6.4 Чому це важливо саме для вашої вимоги "повністю самонавчальна"

Якщо дивитися жорстко, “повністю самонавчальна” система повинна вміти:

- ставати кращою в інтерпретації сирого входу;
- ставати кращою в ухваленні epistemic status;
- ставати кращою в запам’ятовуванні того, що реально корисне;
- ставати кращою в діях по verified outcomes.

Поточний repo цього ще не робить повністю, бо:

- найголовніший фронт помилок, grounding front-end, майже не навчається online;
- автономні дії назовні майже не утворюють training signal loop назад у систему.

## 6.5 Найчесніший висновок про self-learning

Поточна система:

- уже не статична;
- уже не purely hand-coded;
- уже має eval-time adaptation;
- уже має online symbolic growth;
- але ще не є повністю самонавчальною системою, бо не має mature learned grounding loop і не має повного action-driven lifelong learning loop.

## 7. Аудит автономності та "вільної до дій" системи

## 7.1 Що вже добре

- planner не працює по raw ambiguous prose як по truth;
- planner отримує grounded/planner state;
- є constraints, repair directives, alternative worlds;
- є symbolic execution discipline;
- generation уже зав’язана на canonical state;
- overall architecture вже добре готує систему до autonomy under epistemic control.

## 7.2 Як має виглядати повна автономність за masterplan

Повна автономність у masterplan-сенсі не дорівнює просто “уміє планувати”.

Вона означає цикл:

1. grounded perception;
2. state assembly;
3. uncertainty-aware planning;
4. external or simulated action;
5. observation of outcomes;
6. contradiction-aware revision;
7. memory/world writeback;
8. policy improvement from verified outcomes.

За `SYSTEM_REQUIREMENTS_MASTERPLAN.md` і `DOMAIN_ACTION_REQUIREMENTS_MASTERPLAN.md` система має:

- підтримувати explicit goals, constraints, branch alternatives;
- plan under uncertainty;
- revise plan when premises change;
- know when to defer or ask clarification;
- preserve provenance for consequential action;
- improve action policies through verified outcomes.

## 7.3 Точний розрив у поточному repo

Що вже є:

- architecture for grounded planning;
- architecture for uncertainty-aware branch handling;
- architecture for symbolic execution and repair directives.

Що критично не вистачає:

- реального external action substrate;
- стабільного tool/action interface;
- циклу `plan -> act -> observe -> revise` у зовнішньому середовищі;
- execution memory по verified outcomes;
- action-policy learning від реальних дій;
- long-horizon domain orchestration across tools/environments.

Через це:

- repo вже вміє бути сильним внутрішнім мислячим ядром;
- repo ще не вміє бути завершеним автономним агентом у реальному середовищі.

## 7.4 Чому це важливо саме для вимоги "вільна до дій"

Фраза “вільна до дій” у сильному технічному сенсі означає:

- не просто скласти план;
- не просто дати відповідь;
- а мати право і механізм діяти на основі grounded assumptions, потім бачити наслідок і коригувати себе.

Поки що repo до цього не доходить, бо:

- action mostly ends at internal planning or synthesis;
- execution substrate назовні майже не побудований;
- verified real-world outcomes ще не стали основним джерелом skill accumulation.

## 7.5 Найжорсткіший висновок про автономність

Якщо формулювати без пом’якшень:

- система вже має **architecture for autonomous reasoning**;
- система ще не має **completed autonomous action loop**;
- отже вона ще не є повністю “вільною до дій” у masterplan-сенсі.

## 8. Аудит по masterplan-документах

## 8.1 `concept.md`

Статус:

- сильно частково реалізовано

Що добре:

- single canonical runtime
- byte-first
- graph-primary state
- real symbolic substrate
- memory operational
- generation over state
- online adaptation path існує

Що слабко:

- повний training curriculum не доведений до реалізації
- text-only недостатність правильно визнана концептуально, але grounding learning stack ще не побудований
- Stage 3 online adaptation реалізовано лише частково

## 8.2 `DETERMINISTIC_RUNTIME_CONCEPT_UK.md`

Статус:

- найкраще узгоджений документ відносно коду

Що добре:

- heuristic authority demotion
- world-state writeback gate
- planner ingress gating
- memory gating
- SymbolicTaskContext buckets

Що ще болить:

- deterministic boundary усе ще несе занадто багато semantic burden через відсутність learned grounding

## 8.3 `NEURO_SYMBOLIC_BOUNDARY_UK.md`

Статус:

- частково добре реалізовано

Що добре:

- proposal/truth separation набагато краща, ніж раніше
- heuristic proposals не проходять напряму в active truth
- planner/memory/world-state добре захищені

Що не дотягнуто:

- neuro side ще надто слабка в grounding semantics
- symbolic/boundary side досі компенсує відсутній learned semantic engine

## 8.4 `GROUNDING_MASTERPLAN.md`

Статус:

- pipeline реалізований наскрізно, але не в mature target form

Що добре:

- L0/L1/L6/L7/L8/L9 каркас уже дуже відчутний
- provenance discipline сильна
- verification/writeback/planner/memory integration хороша

Що погано:

- L3/L4/L5 досі переважно heuristic-heavy
- learned grounding backbone відсутній
- text grounding усе ще ближче до hint extraction, ніж до mature learned semantics

## 8.5 `TRAINING_MASTERPLAN.md`

Статус:

- реалізований нерівномірно

Що добре:

- training loop є
- joint runtime training path є
- online adaptation не нульова

Що погано:

- grounding objective family практично не імплементована
- heuristic retirement policy описана, але фактично grounding усе ще сильно на heuristics
- online adaptation policy поки вужча за норму документа

## 8.6 `SYSTEM_REQUIREMENTS_MASTERPLAN.md`

Статус:

- як внутрішній cognitive/runtime skeleton частково узгоджений
- як high-end practical system ще далекий

Головний висновок:

- система добре просувається в epistemic discipline;
- система ще не вийшла на рівень practical autonomous competence across domains.

## 8.7 `DOMAIN_ACTION_REQUIREMENTS_MASTERPLAN.md`

Статус:

- здебільшого ще проектна ціль

Чому:

- у репо є reasoning/planning substrate;
- але майже немає зовнішнього action substrate;
- немає сильного domain execution layer;
- немає verified long-horizon real-world action loop.

## 9. Пріоритетні проблеми за важливістю

## P0

- Реалізувати learned grounding backbone замість залежності від `HeuristicFallbackSemanticBackbone`.
- Ввести explicit grounding loss family і trainable grounding modules за `TRAINING_MASTERPLAN.md`.
- Перевести test culture з ablation-heavy на canonical-first.

## P1

- Відрізати semantic ingress від marker/regex dominance там, де потрібен learned multilingual semantics.
- Додати guarded online grounding adaptation.
- Додати реальний autonomy/action substrate поверх planner state.

## P2

- Замінити частину hand-tuned verifier calibration на learned calibration поверх deterministic truth gate.
- Зменшити heuristic symbolic fallback в `ontology_engine.py`.
- Розширити end-to-end canonical integration tests з увімкненими `NET`, `OSF`, `EMC`.

## 10. Що саме робити далі

### 10.1 Якщо пріоритет — не з’їхати з AI на алгоритми

Треба робити не "ще більше правил", а:

- trainable grounding backbone
- trainable grounding adapters
- disagreement metrics між heuristic path і learned path
- retirement metrics для heuristic modules

### 10.2 Якщо пріоритет — повна самонавчальність

Треба добудувати:

- online grounding adaptation
- shadow/reversible online update modes
- broader curriculum/maturity gating
- future-outcome-based learning для action/planning

### 10.3 Якщо пріоритет — автономність і свобода до дій

Треба добудувати:

- public action API
- execution bridge до зовнішніх інструментів/середовищ
- observe/act/revise cycle
- verified action memory
- action-policy learning from verified outcomes

## 11. Остаточний вердикт

Найточніше формулювання поточного стану таке:

- OMEN у цьому репозиторії вже дуже сильний як **канонічний neuro-symbolic runtime skeleton**.
- OMEN у цьому репозиторії вже досить сильний як **truth-disciplined symbolic/world-state architecture**.
- OMEN у цьому репозиторії ще слабкий як **learned grounding system**.
- OMEN у цьому репозиторії ще частковий як **self-learning lifelong system**.
- OMEN у цьому репозиторії ще не доведений до рівня **fully autonomous free-to-act system**.

Найгостріша проблема не в тому, що систему вже "заскриптували як говорити".

Найгостріша проблема в тому, що систему поки ще занадто "заскриптували як розуміти" на вході.

Поки не з’явиться реальний learned grounding backbone і реальний grounding training contract у коді, репозиторій буде залишатися:

- дуже хорошим neuro-symbolic research architecture;
- але ще не тим OMEN, який masterplan описує як повністю самонавчальну, автономну, вільну до дій систему.
