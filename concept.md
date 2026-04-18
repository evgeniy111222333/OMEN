# OMEN — Канонічна концепція і технічна специфікація продукту

## 1. Статус документа

Цей документ є канонічною концепцією OMEN у поточному репозиторії.

Він замінює історичну багатошарову версію концепту, в якій змішувалися:

- ранні гіпотези;
- проміжні ревізії;
- критика старих реалізацій;
- майбутні ідеї;
- частково вже реалізовані виправлення.

Відтепер цей файл має нормативний, а не історичний статус.

Його призначення:

- зафіксувати, що таке OMEN як продукт;
- визначити канонічну архітектуру;
- визначити канонічні стани, цикли та цілі;
- визначити, що є ядром, а що є розширенням;
- прибрати концептуальні суперечності попередніх редакцій.

Якщо якийсь старий фрагмент історичного концепту суперечить цьому документу, правильним вважається цей документ.

Канонічний рантайм продукту в репозиторії: `omen_scale.py`, доступ через `omen.py`.

Канонічний публічний surface:

- `omen.OMEN`
- `omen.OMENConfig`
- `omen.build_omen(...)`

OMEN не має кількох рівноправних основних рантаймів. Усі інші історичні модулі можуть існувати лише як legacy або research layers, але не як альтернативний "справжній OMEN".

## 2. Що таке OMEN як продукт

OMEN — це world-grounded neuro-symbolic cognitive runtime, який поєднує:

- байтове сприйняття і компресію;
- побудову структурованого стану світу;
- довготривалу нейронну і точну символічну пам'ять;
- формальне символьне виведення;
- абдукцію, дедукцію та індукцію;
- контроль вартості міркування;
- генерацію тексту, коду і планів через структурований стан, а не лише через поверхневу мовну статистику.

OMEN не є просто мовною моделлю.

OMEN не є просто символьним солвером.

OMEN не є просто world model.

OMEN є інтегрованою системою, в якій:

- мова є вхідним і вихідним носієм;
- світовий стан є центральною внутрішньою реальністю;
- символічний шар є операбельним механізмом перевірки, виведення та узагальнення;
- пам'ять є обчислювальним ресурсом, а не пасивним архівом;
- ціна обчислення і складність правил входять у загальну економіку системи.

## 3. Кінцева мета OMEN

Кінцева мета продукту:

1. Перетворювати сирі послідовності спостережень на структурований стан світу.
2. Компресувати досвід не лише статистично, а й концептуально.
3. Виводити, перевіряти, відкидати і ремонтувати правила про світ.
4. Відрізняти перевірене знання від гіпотези, помилки або шуму.
5. Використовувати пам'ять, reasoning і планування лише тоді, коли це виправдано інформаційно та обчислювально.
6. Генерувати відповіді, програми або плани як наслідок стану світу, symbolic reasoning і memory recall, а не лише через поверхневе next-token continuation.
7. Підтримувати online symbolic learning і контрольоване самовдосконалення без розриву між train і eval режимами.

У короткій формі:

OMEN повинен будувати пояснювально корисний, перевірюваний, керований за вартістю внутрішній світ і діяти через нього.

## 4. Що вважається ядром, а що розширенням

### 4.1 Обов'язкове ядро

Без цього OMEN не вважається концептуально валідним:

- byte-level input;
- NET як канонічний tokenizer/compressor;
- graph-primary world state;
- world-graph-grounded perception;
- WorldRNN як модель переходів;
- M-Core як довготривала пам'ять;
- exact symbolic substrate з FOL-термами, змінними та уніфікацією;
- KnowledgeBase з епістемічним статусом правил;
- execution-trace-first symbolic supervision;
- unified MDL/VFE-style objective;
- graph-centered decoder state.

### 4.2 Інтегровані підсистеми другого рівня

Вони є частиною канонічного стеку, але концептуально залежать від ядра:

- Saliency Trace;
- VeM;
- continuous symbolic cycle;
- eval-capable online symbolic learning;
- program anchoring;
- EMC як meta-controller reasoning depth;
- symbolic memory index.

### 4.3 Просунуті генеративні розширення

Вони є канонічними support-модулями, але не визначають базову валідність OMEN без ядра:

- OSF;
- creative cycle engines;
- ICE і внутрішні цілі;
- analogy / counterfactual / ontology / aesthetic / intrinsic engines.

Ці розширення не повинні існувати як декоративні надбудови. Вони мають бути похідними від уже стабільного world/symbolic/memory ядра.

## 5. Базові принципи архітектури

### 5.1 World-state-first

Внутрішнім центром OMEN є не текст і не щільний latent-вектор сам по собі, а структурований стан світу.

### 5.2 Graph-primary

Канонічний `z` є graph-primary. Щільні вектори є похідними або спеціалізованими видами цього стану.

### 5.3 Byte-first, а не BPE-first

Канонічний вхідний substrate — сирі UTF-8 байти. BPE/WordPiece не є канонічною токенізацією продукту.

### 5.4 Symbolic layer must be real

Символьний шар вважається валідним лише тоді, коли він має:

- терми;
- змінні;
- уніфікацію;
- правила;
- верифікацію;
- протиріччя;
- trace-grounded supervision;
- epistemic status.

Графова або GNN-подібна "символьність" без цього не є достатньою.

### 5.5 Memory is operational

Пам'ять має не лише зберігати, а й:

- впливати на стан;
- керувати recall;
- брати участь у curiosity loop;
- бути оцінюваною через вартість і корисність.

### 5.6 Reasoning cost is part of the objective

Глибина доказу і кількість reasoning steps не є безкоштовними. Cost of computation входить у цільову економіку системи.

### 5.7 Execution grounds meaning

Символьне навчання повинно бути прив'язане до execution traces, переходів стану, цільових фактів і контрприкладів.

### 5.8 Generation is synthesis over state

Генерація має бути читанням із graph-centered world state, memory state, symbolic state і program state, а не лише декодуванням поверх токенного контексту.

## 6. Канонічний surface і модулі репозиторію

### 6.1 Канонічний runtime

- `omen_scale.py`
- `omen_scale_config.py`
- `omen_canonical.py`
- `omen.py`

### 6.2 Канонічні support modules

- `omen_world_model.py`
- `omen_perceiver.py`
- `omen_prolog.py`
- `omen_saliency.py`
- `omen_net_tokenizer.py`
- `omen_emc.py`
- `omen_osf*.py`
- `omen_symbolic/world_graph.py`
- `omen_symbolic/execution_trace.py`
- `omen_symbolic/integration.py`
- `omen_symbolic/memory_index.py`
- `omen_symbolic/creative_cycle.py` та пов'язані творчі модулі

### 6.3 Legacy статус

Будь-які історичні модулі на кшталт старих runtime-варіантів не є концептуальною нормою продукту.

## 7. Канонічний контракт стану системи

### 7.1 Вхід

Канонічний вхід продукту зараз — послідовність UTF-8 байтів.

Це означає:

- `vocab_size = 256` є канонічним режимом;
- вхідні символи не передбачають наперед заданого словника на десятки тисяч токенів;
- семантичні одиниці не фіксуються наперед на рівні BPE, а формуються далі через NET, world graph і symbolic layer.

### 7.2 Канонічний `z`

Канонічний контракт:

- `out["z"]` — це структурований `CanonicalWorldState`;
- `out["z_dense"]` — це щільний decoder-facing readout, похідний від graph-primary state;
- `out["world_state"]` — це той самий canonical world state;
- `out["z_graph_struct"]` — graph view цього стану;
- `out["z_world"]` — dense grounded state, отриманий із graph-centered fusion.

Отже:

- primary state = graph/world state;
- dense state = derived view;
- symbolic state = integrated reasoning view;
- program state = optional task-specific grounding view.

### 7.3 `CanonicalWorldState`

Канонічний стан світу включає:

- `graphs`: один або кілька `WorldGraphState`;
- `neural_state`: базовий нейронний стан до повного graph/symbolic fusion;
- `graph_grounded_state`: стан після world-graph grounding;
- `graph_projection`: projection view графа;
- `graph_readout_state`: attention/readout view графа;
- `grounded_state`: підсумковий dense decoder-facing state;
- `symbolic_state`: state після symbolic reasoning;
- `memory_state`: memory-conditioned state;
- `program_state`: state, заякорений у program target facts;
- `symbolic_facts`: факти, які беруть участь у reasoning контексті;
- `target_facts`: цільові факти для доказу, індукції або програмного якірення;
- `metadata`: сервісні кількісні ознаки канонічного циклу.

Суттєве правило:

`grounded_state` не є окремою альтернативною природою `z`. Це похідний readout для декодування і downstream heads. Першинною сутністю залишається graph-primary world state.

### 7.4 `WorldGraphState`

Світовий граф є grounding substrate для світу. Його вузли та ребра утворюються з:

- фактів;
- термів;
- execution trace transitions;
- saliency links;
- counterfactual relations;
- co-occurrence relations;
- task context facts.

Канонічні типи зв'язків:

- `shared_term`
- `same_pred`
- `trace_step`
- `counterfactual`
- `saliency`
- `cooccurs`

`WorldGraphState` повинен репрезентувати:

- локальну структуру фактів;
- динаміку переходів;
- trace-based supervision;
- міст між world model і symbolic layer.

### 7.5 `SymbolicTaskContext`

Кожен symbolic reasoning episode працює не в абстракції, а в task context, який містить:

- `observed_facts`;
- `target_facts`;
- `goal`;
- `execution_trace`;
- saliency-derived facts;
- memory-derived facts;
- NET-derived concept facts;
- metadata з gap/statistics/world graph summaries.

Саме task context, а не глобальна KB у відриві від поточного прикладу, є локальною операційною сценою reasoning.

## 8. Рівень 1 — NET: канонічний tokenizer і компресор

## 8.1 Призначення

NET замінює BPE/WordPiece як основну tokenizer-парадигму.

Його роль:

- працювати на сирих байтах;
- утворювати контекстуально осмислені концепти;
- забезпечувати MDL-контроль словника;
- створювати міст між byte-level input і symbolic/world structure.

### 8.2 Чому BPE не є канонічним шляхом

BPE і WordPiece не є епістемічними одиницями. Вони є компромісом між частотою, індексацією і зручністю для мовної моделі, але не репрезентують стабільні концепти світу.

Тому в OMEN:

- байт є базовим носієм;
- концепт виникає як результат контекстного кодування і квантованого стискання;
- словник є динамічним codebook, а не фіксованим lexical inventory.

### 8.3 Компоненти NET

1. `ByteContextEncoder`

- приймає байти `[0..255]`;
- кодує їх у контекстні вектори;
- використовує двонапрямну увагу;
- виконує segment-aware pooling, щоб збирати локальні семантичні сегменти.

2. `EpistemicQuantizer`

- зіставляє контекстні вектори з codebook-концептами;
- використовує VQ-VAE style quantization;
- підтримує EMA-оновлення codebook;
- підтримує динамічний ріст словника;
- запобігає колапсу dead codes.

3. `ByteDecoder`

- відновлює байтову послідовність з концептів і фінального стану системи;
- робить компресію оборотною та навчально корисною.

4. `NeuralEpistemicTokenizer`

- оркеструє `ByteContextEncoder -> EpistemicQuantizer -> ByteDecoder`.

### 8.4 Формалізація NET

Канонічний принцип токенізації:

`Tokenization* = argmin_V [ Length(Z) + Distortion(X, X_hat) + Complexity(V) ]`

де:

- `X` — байтова послідовність;
- `Z` — послідовність квантованих концептів;
- `V` — codebook концептів;
- `Distortion` — втрата реконструкції;
- `Complexity(V)` — вартість самого словника.

Канонічний objective NET:

`L_NET = L_vq + L_rec + λ_voc * Σ_v Complexity(v) - λ_sem * I(Z; Γ) + L_anti_collapse`

де:

- `L_vq` — commitment / quantization loss;
- `L_rec` — reconstruction loss;
- `Complexity(v)` — MDL-вартість code entry;
- `I(Z; Γ)` — взаємна інформативність між NET-концептами і symbolic structure;
- `L_anti_collapse` — антиколапсні терміни codebook usage.

У поточному продукті `I(Z; Γ)` апроксимується semantic feedback pairs із symbolic layer.

### 8.5 Канонічний висновок по NET

NET-first є канонічною tokenizer-лінією OMEN.

Classic token path може існувати тільки як compatibility/ablation режим, але не як концептуальний baseline продукту.

## 9. Рівень 2 — Perception, world graph і variational concept state

### 9.1 Від byte stream до concept state

Після NET байтова послідовність переходить у контекстні токенні/концептні представлення.

Далі OMEN будує concept-level state через:

- `Perceiver` або graph-native posterior;
- world graph construction;
- variational sampling для `z`, `μ`, `logvar`.

### 9.2 Перевага graph-native posterior

Якщо world graph уже доступний, канонічний шлях — будувати posterior не з "голого" щільного латента, а з graph-grounded signal.

Тобто canonical posterior є graph-aware whenever available.

### 9.3 WorldGraphEncoder

`WorldGraphEncoder` виконує:

- signature encoding термів і атомів;
- message passing по graph edges;
- побудову pooled graph state;
- формування node states;
- підготовку transition states / transition targets;
- збагачення графа saliency та task context фактами.

Це робить світовий граф не просто службовою структурою, а центральним носієм world semantics.

## 10. Рівень 3 — World model: `WorldRNN`

### 10.1 Призначення

`WorldRNN` моделює переходи стану світу, а не просто приховану динаміку тексту.

Він:

- читає поточний state;
- враховує дії або action probabilities;
- conditionиться на graph context;
- видає наступний state;
- оцінює causal error і graph alignment.

### 10.2 Graph-conditioned transitions

Канонічно `WorldRNN` не ізольований від графа. Він використовує:

- pooled graph state;
- trace transition states;
- graph-conditioned machine state;
- state refinement toward graph context.

Тобто world model у продукті не є просто GRU над latent-вектором. Вона є graph-conditioned transition model.

### 10.3 Діагностика переходу

Для кожного переходу важливі:

- `causal_error`;
- `graph_alignment`;
- `state_residual`.

Ці величини використовуються не лише як діагностика, а й як сигнали для training та reasoning economy.

### 10.4 World objective

Канонічний world-term включає:

`L_world = log(1 + Huber(z_sim, z_target))`

та відповідний world NLL у бітах для MDL-частини.

У центрі стоїть така ідея:

world model повинна вчитися симулювати concept/world state, а не тягнути сам concept state до власних помилок.

## 11. Епістемічна прогалина і curiosity loop

### 11.1 `EpistemicGapDetector`

OMEN обчислює епістемічну прогалину між:

- поточним grounded concept/world state `z`;
- симульованим станом `z_sim`.

Є два режими:

1. Exact-grad режим:

`E(z) = || ∂L_world / ∂z ||^2`

2. Approximate режим:

`E(z) ≈ (z - z_sim)^2 * (1 + alignment_gap)`

де `alignment_gap` відображає косинусну неузгодженість між реальним і симульованим станами.

Далі:

- `gap_norm = ||E(z)||`
- `hot_dims` — найбільш проблемні виміри стану

### 11.2 `CuriosityModule`

Curiosity активується, коли `gap_norm > τ_epi`.

Він:

- читає memory по запиту з гарячих вимірів;
- робить episodic recall;
- поєднує memory signals;
- запускає counterfactual rollouts;
- enrich-ить `z`;
- піднімає `unknown` flags, якщо memory не дає корисного сигналу.

Curiosity у продукті не є окремим "натхненням". Це строго керований механізм обробки epistemic insufficiency.

## 12. M-Core — довготривала пам'ять

### 12.1 Призначення

M-Core забезпечує:

- довготривале зберігання станів;
- асоціативний recall;
- episodic recall;
- точний symbolic recall;
- буферизований асинхронний запис.

### 12.2 Два memory substrate

OMEN канонічно має два комплементарні memory levels:

1. Neural long-term memory

- для щільних state recalls;
- для similarity-based retrieval;
- для інтеграції з curiosity і world state.

2. SymbolicMemoryIndex

- exact symbolic interface;
- зберігає факти разом з embedding-представленнями;
- підтримує vector recall;
- підтримує pattern recall за предикатами і anchor values.

### 12.3 Канонічний memory contract

Пам'ять повинна вміти:

- `read(z_query)` для neural retrieval;
- `episodic_recall(z_query)` для епізодичної вибірки;
- `write(facts, embeddings)` для symbolic index;
- `recall(query, predicate_hints, anchor_values)` для змішаного structured/vector retrieval.

### 12.4 Чому symbolic memory є обов'язковою

Точний recall фактів не може бути повністю замінений лише щільним векторним memory. Символьний шар має мати exact storage path, інакше довготривала knowledge continuity буде нестабільною.

## 13. Символічне ядро — FOL-capable ∂-Prolog substrate

## 13.1 Канонічний symbolic baseline

Символічний рівень OMEN є FOL-capable Horn-clause substrate.

Його базові сутності:

- `Const`
- `Var`
- `Compound`
- `HornAtom`
- `HornClause`
- `Substitution`

Отже, канонічний symbolic core не є pseudo-symbolic GNN. Він є реально операбельним логічним шаром.

### 13.2 Уніфікація

Канонічно OMEN повинен мати справжню уніфікацію, а не підстановку-константну евристику.

Уніфікація потрібна для:

- forward chaining;
- proof search;
- matching body/head правил;
- variable-grounded induction;
- program/task queries.

### 13.3 KnowledgeBase

`KnowledgeBase` містить:

- факти;
- правила;
- епістемічні записи по правилах;
- механізм forward chaining;
- механізм consolidation.

### 13.4 Епістемічний статус правил

Кожне правило має статус:

- `proposed`
- `verified`
- `contradicted`

Це не опціонально. Без цього OMEN не відрізняє гіпотезу від знання.

### 13.5 Корисність правила

Канонічна utility-модель:

`Utility(R) = success_count / (1 + age_steps)`

Канонічний rule cost:

`L_rule(R) = Complexity(R) - η_util * Utility(R)`

Канонічний rule regularizer:

`L_rules = Σ_{R in KB} L_rule(R)`

Це означає:

- корисні правила не повинні штрафуватися так само, як шум;
- застарілі, непотрібні або суперечливі правила повинні бути витіснені.

### 13.6 Consolidation

База знань повинна регулярно:

- видаляти contradicted rules;
- видаляти слабкі proposed rules;
- зберігати verified і utility-rich rules.

### 13.7 TensorKnowledgeBase

Для практичного масштабування OMEN канонічно допускає tensorized symbolic path. Це не змінює логічної природи symbolic core, а робить її більш операційною для GPU execution.

## 14. Verification Module (VeM)

VeM оцінює очікувану корисність правила ще до його закріплення.

Його ідея:

`U(R) = E_future[ Success(R) - α_cost * Cost(R) ]`

Практична форма в продукті:

- rule embeddings;
- neural utility scoring;
- фільтрація candidate rules;
- retrospective self-supervision на базі подальшого успіху/невдачі правила.

Канонічний VeM penalty:

`L_VeM = δ_vem * E_{R~Abduction}[ max(0, τ_vem - U(R)) ]`

VeM потрібен, щоб абдукція не перетворила symbolic layer на генератор сміття.

## 15. Execution-trace-first supervision

### 15.1 Принцип

Символьне навчання в OMEN не повинно бути відірване від виконання.

Тому canonical symbolic supervision будується execution-trace-first.

### 15.2 `SymbolicExecutionTraceBundle`

Канонічний trace bundle містить:

- `language`
- `source_text`
- `observed_facts`
- `target_facts`
- `transitions`
- `counterexamples`

### 15.3 Що саме дає execution trace

Execution traces дають:

- факти про значення стану;
- факти про типи;
- assign / return / error events;
- transition structure;
- counterexamples;
- target facts для induction/verification;
- міст між програмною поведінкою і логічними правилами.

Саме це переводить symbolic learning із режиму "логіка поверх тексту" у режим "логіка поверх виконуваного світу".

## 16. `DifferentiableProver` і continuous symbolic cycle

### 16.1 Роль прувера

`DifferentiableProver` є центральним інтегратором символічного шару.

Він повинен:

- materialize task context;
- load observed facts у working memory;
- використовувати facts/rules/world context;
- проводити proof search;
- відповідати на symbolic query;
- абдукувати нові правила;
- підтримувати online induction;
- збагачувати decoder-facing state.

### 16.2 Принцип циклу

Канонічний symbolic cycle має три логічні фази:

1. Абдукція:

- висунути правило або пояснювальну гіпотезу.

2. Дедукція:

- застосувати правила до фактів і вивести наслідки.

3. Індукція:

- узагальнити стабільні успішні структури в нові правила.

### 16.3 Continuous symbolic cycle

У канонічному продукті symbolic learning не обмежений train-only фазою.

Є continuous cycle, який:

- приймає contextual candidates;
- оцінює їх на symbolic/world/token criteria;
- перевіряє через thresholds;
- позначає як verified або contradicted;
- виконує repairs;
- допускає eval-time learning;
- пов'язує symbolic corrections з world self-update.

### 16.4 Program anchoring

Для задач програмного або trace-driven типу OMEN будує `program_state` з `program_target_facts`.

Це дає:

- жорсткіший зв'язок між symbolic goal і decoder;
- можливість graph-centered decode не лише від загального state, а й від program-grounded target structure.

## 17. Інтеграція concept, memory, symbolic і graph state

Канонічний інтегратор — `SymbolicStateIntegrator`.

Він реалізує дві принципові фази.

### 17.1 Pre-symbolic fusion

Перед reasoning memory повинна збагатити concept state:

`z_pre = v_mem + g_pre([z_concept, v_mem]) * (z_concept - v_mem)`

Це означає:

- memory не просто додається;
- вона задає informative prior для symbolic reasoning.

### 17.2 Post-symbolic fusion

Після reasoning symbolic state має право перевизначати inconsistent concept dimensions:

`z_base = v_mem + g_mem([z_concept, z_symbolic, v_mem]) * (z_concept - v_mem)`

`z_fused = z_base + g_sym([z_base, z_symbolic, z_symbolic - z_base]) * (z_symbolic - z_base)`

Це критично:

- symbolic layer не є слабким additive hint;
- вона може коригувати стан, якщо доведення сильніше за сирий concept state.

### 17.3 Graph-centered readout

Декодер читає не абстрактний fused vector у вакуумі, а graph-centered state:

- graph readout будується через attention до node states;
- pooled graph state слугує структурним якіренням;
- program state, якщо є, вплітається в graph-centered projection.

Отже, канонічний decoder state є graph-centered, а не purely latent-centered.

## 18. Saliency Trace

### 18.1 Призначення

Saliency потрібна, щоб перетворити neural attention у структуровані символічні сигнали.

Канонічний напрям:

`attention -> role/link facts -> expected facts -> symbolic reasoning -> correction`

### 18.2 Фіксована рольова онтологія

Канонічна рольова онтологія має фіксований список ролей, зокрема:

- `agent`
- `patient`
- `action`
- `modifier`
- `coref`
- `context`
- а також розширені ролі типу `subject`, `recipient`, `instrument`, `location`, `temporal`, `attribute`

Важливий принцип:

saliency ontology не повинна розмиватися безмежними анонімними `role_n`. Вона має бути фіксованою і контрольованою.

### 18.3 Виходи Saliency

Saliency module будує:

- raw facts;
- semantic facts;
- expected facts;
- role targets;
- graph latent;
- consistency score;
- кількість абдукованих trace rules.

### 18.4 Saliency losses

Канонічно saliency включає:

- role loss;
- structural loss;
- consistency loss;
- trace-rule penalty.

Узагальнено:

`L_sal = γ_role L_role + β_struct L_struct + δ_cons L_cons + η_rule L_trace_rule`

### 18.5 Saliency як міст, а не косметика

Saliency не є пояснювальною візуалізацією уваги. Вона є механізмом побудови symbolic facts із neural dynamics.

## 19. EMC — Efficient Meta-Controller

### 19.1 Проблема, яку він розв'язує

Reasoning depth не повинен бути фіксованим гіперпараметром незалежно від задачі.

Система повинна вміти вирішувати:

- зупинитись;
- згадати пам'ять;
- зробити ще один крок доведення;
- абдукувати;
- переключитися на intrinsic goal.

### 19.2 Простір дій EMC

Канонічні дії:

- `Stop`
- `RecallMCore`
- `ForwardChainStep`
- `Abduce`
- `FocusIntrinsicGoal`

### 19.3 Bellman-formulation

Канонічне рівняння:

`V*(s) = max( U_stop(s), max_a [ -C(a) + γ E[V*(s')] ] )`

де:

`U_stop(s) = R_task(s) + η_int R_intermediate(s) - λ_gap GapNorm(s) - λ_mdl MDL(proof) - λ_time T`

Це означає:

- зупинка є раціональною дією;
- додатковий крок reasoning повинен виправдати свою вартість;
- reasoning cost є частиною глобальної економіки системи.

### 19.4 Actor-Critic шар

EMC концептуально є control layer над уже наявним reasoning substrate, а не заміною symbolic core.

Він навчається через:

- value estimation;
- trajectory rewards;
- action costs;
- entropy/strategy trade-off.

### 19.5 Статус EMC

EMC є канонічною інтегрованою підсистемою, але не визначає логіку світу сам по собі. Він керує тим, скільки reasoning потрібно, а не тим, що є істиною.

## 20. OSF — OMEN Synthesis Framework

### 20.1 Призначення

OSF є генеративно-планувальною надбудовою над world/symbolic core.

Він потрібен для складнішого synthesis режиму, де недостатньо просто decode-ити послідовність.

### 20.2 Ієрархія OSF

Канонічна чотирирівнева схема:

1. `IntentEncoder`

- перетворює фінальний state у symbolic goal / intent.

2. `SymbolicPlanner`

- будує operator sequence.

3. Template / plan realization layer

- перетворює план у структурні шаблони виразу чи програми.

4. `HierarchicalDecoder`

- генерує кінцеві logits.

### 20.3 Simulation and reflection

OSF включає:

- `WorldSimulator`;
- `SymbolicPlanVerifier`;
- `ReflectionModule`;
- meta-controller стратегії синтезу.

Це означає, що генерація може:

- спланувати;
- симулювати результат;
- перевірити план;
- локально відремонтувати невідповідність.

### 20.4 OSF objective

Канонічно:

`J_OSF = L_ce + λ_plan L_plan + λ_sim L_sim + λ_refl L_refl + λ_meta L_meta`

OSF не замінює OMEN core. Він читає state, який уже побудований ядром.

## 21. Creative cycle і intrinsic goals

### 21.1 Статус

Creative engines є частиною канонічного support stack, але їхній сенс існує лише поверх стабільного symbolic/world ядра.

### 21.2 Призначення

Вони потрібні для:

- rule recombination;
- analogy;
- counterfactual hypothesis generation;
- ontology expansion;
- intrinsic goal scheduling;
- пошуку нових корисних explanatory structures.

### 21.3 Канонічна умова валідності

Creative cycle є валідним лише тоді, коли його результати:

- піддаються VeM/verification;
- можуть бути виражені в rule/task/world термінах;
- мають utility або compression gain;
- не руйнують epistemic discipline KB.

Без цього creativity перетворюється на нерегульоване породження шуму, що концептуально неприпустимо.

## 22. Канонічний `forward` цикл продукту

Нормативний цикл `forward` у продукті має такий зміст.

### Крок 1. Byte input

OMEN отримує `src` і `tgt` як UTF-8 байтові послідовності.

### Крок 2. Byte/context encoding

Через NET утворюються:

- контекстні токенні вектори;
- квантовані концепти;
- symbolic facts із NET-концептів;
- attention maps для saliency, якщо вона увімкнена.

### Крок 3. Perception world graph

Система будує первинний perception graph із вхідних спостережень.

### Крок 4. Variational concept state

OMEN будує variational latent state. Якщо world graph доступний, posterior має бути graph-native.

### Крок 5. Saliency pass

Attention maps і token hidden states переводяться у saliency facts, expected facts та graph latent.

### Крок 6. Enriched world graph

Початковий граф збагачується:

- saliency facts;
- task context facts;
- execution trace targets;
- contextual anchors.

### Крок 7. Graph grounding

Базовий нейронний стан `z_neural` заземлюється у world graph і перетворюється на graph-grounded state.

### Крок 8. Memory retrieval

M-Core повертає memory state `v_mem`.

### Крок 9. World rollout

`WorldRNN` симулює траєкторію станів і формує `z_sim` та world targets.

### Крок 10. Epistemic gap

Обчислюються:

- epistemic map;
- gap norm;
- hot dimensions;
- gap statistics.

### Крок 11. Curiosity

За потреби система:

- читає memory глибше;
- робить episodic recall;
- будує counterfactual rollouts;
- enrich-ить стан.

### Крок 12. Symbolic memory seeding

Із decoder surprise, saliency і NET будуються memory hints і відбувається recall symbolic facts.

### Крок 13. Symbolic task context

Система формує повний symbolic task context:

- observed facts;
- target facts;
- execution trace;
- current symbolic goal;
- memory facts;
- NET facts;
- saliency support;
- AST / program facts, якщо вони доступні.

### Крок 14. Prover world priming

Прувер отримує world context, observed facts і task context.

### Крок 15. Symbolic reasoning

Через EMC або напряму:

- proof search;
- forward chaining;
- abduction;
- induction;
- online symbolic updates.

### Крок 16. VeM + cycle diagnostics

Оцінюються:

- utility candidate rules;
- induction outcomes;
- verified / contradicted / retained / repaired counts;
- continuous cycle stats.

### Крок 17. State fusion

Збираються разом:

- concept/world state;
- memory state;
- symbolic state;
- optional program state.

### Крок 18. Graph-centered decoder state

Фінальний decoder state будується через graph-centered readout.

### Крок 19. Compose canonical world state

Створюється `CanonicalWorldState` як канонічний `z`.

### Крок 20. Decode / synthesize

Вихідний decoding layer читає:

- graph-centered state;
- symbolic state;
- program state;
- OSF, якщо увімкнений synthesis mode.

### Крок 21. Buffer writes

Після цього M-Core і symbolic memory отримують дані для буферизованого запису.

## 23. Канонічний `generate` цикл

У режимі генерації OMEN не повинен перетворюватися на "голий decoder".

На кожному кроці генерації система має право:

- повторно оцінити world state;
- оновити symbolic task context;
- виконати additional reasoning;
- recall-ити memory;
- скоригувати decoder state через graph-centered fusion;
- застосувати EMC для рішення, чи потрібен додатковий reasoning step;
- за потреби активувати OSF для synthesis-heavy decoding.

Отже, generation у продукті є online reasoning-aware process.

## 24. Канонічна економіка системи: MDL + VFE + reasoning cost

## 24.1 Головний принцип

Усе ядро продукту повинно підпорядковуватися одній економіці:

- краще пояснювати;
- краще стискати;
- не накопичувати непотрібну модельну складність;
- не накопичувати непотрібні правила;
- не витрачати reasoning resources без вигоди.

### 24.2 Канонічний outer objective

Канонічна зовнішня форма:

`J_total = FE + A_aux`

де:

- `FE` — основна free-energy / bits-per-token частина;
- `A_aux` — допоміжні стабілізаційні та керувальні терміни.

### 24.3 Основна бітова частина

Спостережні біти:

`B_obs = B_token + B_world`

Локальна складність:

`B_local = β_kl * (KL_q + KL_mem + KL_sym) + α_mem * NLL_read + B_latent_scale`

Глобальна модельна складність:

`B_model = B_scale + B_rules`

де:

`B_rules = Σ_{R in KB} [ Complexity(R) - η_util * Utility(R) ]`

Тоді:

`FE = BitsPerToken = (B_obs + B_local) / N_valid + B_model / N_seen`

де:

- `N_valid` — кількість валідних поточних токенів;
- `N_seen` — кількість уже побачених токенів для амортизації глобальної модельної складності.

Це канонічна форма економіки продукту. Вона прибирає стару нечіткість, де MDL, perplexity, symbolic penalties і memory cost існували як набір паралельних майже-фінальних формул.

### 24.4 Допоміжна енергія

Допоміжна частина:

`A_aux = λ_world L_world + λ_sym L_sym + λ_recall L_recall + δ L_complex + λ_cur L_curiosity + λ_net L_NET + λ_vem L_VeM + λ_meta (L_meta + L_traj + C_reason) + λ_prog L_prog_anchor + λ_prog_dec L_prog_decode + λ_sal L_sal + λ_osf J_OSF`

Тут:

- `L_world` — transition/world consistency term;
- `L_sym` — symbolic reasoning loss;
- `L_recall` — якість memory recall;
- `L_complex` — додаткові складнісні регуляризатори;
- `L_curiosity` — counterfactual / curiosity term;
- `L_NET` — tokenizer/compressor loss;
- `L_VeM` — penalty за слабкі правила;
- `L_meta`, `L_traj`, `C_reason` — meta-control and reasoning cost;
- `L_prog_anchor`, `L_prog_decode` — program-target support;
- `L_sal` — saliency terms;
- `J_OSF` — synthesis objective.

### 24.5 Висновок по objective

Канонічно:

- MDL/FE частина є центральною;
- auxiliary terms не підміняють її, а стабілізують і направляють;
- reasoning cost є явною частиною економіки;
- rule utility входить у ціну правил;
- tokenizer, symbolic layer і world model не живуть на різних несумісних objective.

## 25. Канонічний training curriculum

### 25.1 Stage 1 — NET pretraining

Перший етап:

- byte-level compression;
- codebook stabilization;
- reconstruction quality;
- anti-collapse control;
- без повної залежності від symbolic core.

Ціль:

NET повинен мати стабільний і неколапсований codebook до повного joint training.

### 25.2 Stage 2 — joint OMEN training

Другий етап:

- NET;
- graph-native perception;
- world model;
- memory;
- symbolic prover;
- saliency;
- VeM;
- EMC;
- program anchoring;
- continuous symbolic cycle;
- OSF за потреби.

Тут система навчається як єдиний runtime.

### 25.3 Stage 3 — eval-capable online symbolic adaptation

Третій рівень зрілості:

- online symbolic updates;
- eval-time continuous cycle;
- self-update world terms;
- repair / contradiction handling;
- intrinsic goal control.

OMEN не повинен мати жорсткий розрив між "вчиться" і "думає". Він повинен мати контрольований online adaptation path.

## 26. Дані і типи задач

Канонічні джерела сигналу в продукті:

- реальні UTF-8 текстові корпуси;
- програмний код;
- execution traces;
- synthetic rule-transfer tasks;
- counting / algorithmic / program-like tasks;
- benchmarks, що змушують symbolic/world integration бути корисною, а не декоративною.

Канонічний принцип:

тексту недостатньо. Потрібні також задачі, де:

- є стани;
- є переходи;
- є правила;
- є цільові факти;
- є контрприклади;
- є цінність від доведення, а не лише від локальної мовної статистики.

## 27. Evaluation story

OMEN вважається концептуально успішним лише тоді, коли доведено не лише next-token quality, а й такі осі:

### 27.1 Compression and prediction

- token bits / perplexity;
- world NLL;
- free energy dynamics.

### 27.2 World modeling

- graph alignment;
- causal error;
- transition quality;
- state anchoring quality.

### 27.3 Symbolic competence

- proof success;
- target fact hit rate;
- verified vs contradicted rule statistics;
- induction precision;
- usefulness of abduced rules.

### 27.4 Memory usefulness

- recall quality;
- effect of memory on gap reduction;
- exact symbolic recall utility;
- memory pressure and consolidation quality.

### 27.5 Adaptive reasoning economy

- EMC stop quality;
- cost-aware proof depth;
- trajectory reward;
- value of computation.

### 27.6 Non-decorative integration

OMEN має показувати, що:

- saliency реально допомагає symbolic reasoning;
- world graph реально покращує posterior і decoding;
- NET concepts реально корелюють із symbolic semantics;
- symbolic reasoning реально змінює final state;
- OSF реально покращує synthesis, а не лише дублює decoder.

### 27.7 Online learning

Система повинна демонструвати:

- утримання verified rules;
- відсів contradicted rules;
- repair capability;
- контрольоване eval-time adaptation без руйнування KB.

## 28. Явно відкинуті або deprecated ідеї

Нижче зафіксовано, що більше не є канонічною концепцією OMEN.

### 28.1 BPE/WordPiece як основа продукту

Не є каноном.

### 28.2 Великий статичний словник як основний tokenizer-базис

Не є каноном.

### 28.3 `z` як просто один щільний latent-vector без структурованого світу

Не є каноном.

### 28.4 Symbolic core як pure graph/GNN "натяк на логіку"

Не є каноном.

### 28.5 "Без solver" у значенні "без справжнього reasoning substrate"

Не є каноном.

Правильна інтерпретація:

OMEN не хоче архітектурно зовнішнього, погано інтегрованого, мертвого symbolic appendage. Але йому потрібен інтегрований reasoning-capable symbolic substrate.

### 28.6 Правила без epistemic status

Не є каноном.

### 28.7 Навчання лише на тексті як достатня умова для symbolic/world emergence

Не є каноном.

### 28.8 Кілька рівноправних основних runtime-ліній

Не є каноном.

## 29. Мінімальний валідний OMEN

Щоб система вважалася концептуально валідним OMEN, а не "частково схожою моделлю", вона повинна мати одночасно:

- byte-level input;
- NET-first tokenizer;
- graph-primary world state;
- graph-conditioned world transitions;
- long-term neural memory;
- exact symbolic memory path;
- FOL-capable symbolic rules;
- epistemic status and consolidation;
- execution-trace-first symbolic supervision;
- graph-centered decoder state;
- unified MDL/FE objective.

Без цього система може бути цікавою дослідницькою моделлю, але не OMEN у канонічному сенсі.

## 30. Формула кінцевого продукту

У найкоротшому канонічному формулюванні:

OMEN = byte-level epistemic compressor
+ graph-primary world model
+ exact neuro-symbolic reasoning substrate
+ long-term memory
+ adaptive reasoning controller
+ synthesis layer over structured state

під єдиною економікою:

мінімізувати surprise і description length,
максимізувати verified explanatory utility,
мінімізувати непотрібну rule і reasoning complexity.

## 31. Остаточний вердикт цього концепту

Цей документ фіксує такий канон:

- OMEN є world-state-centered, а не text-centered;
- OMEN є graph-primary, а не dense-latent-primary;
- OMEN є NET-first, а не BPE-first;
- OMEN є FOL-capable, а не pseudo-symbolic;
- OMEN є execution-grounded, а не purely textual;
- OMEN має exact symbolic memory, а не лише vector recall;
- OMEN рахує cost of reasoning, а не ігнорує його;
- OMEN генерує через structured state, а не лише через continuation;
- OMEN має один канонічний runtime.

Це і є той фінальний технічний опис продукту, який повинен вважатися актуальним концептом OMEN у цьому репозиторії.
