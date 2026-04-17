# Оновлений концептуальний аудит кодової бази OMEN відносно `concept.md`

Дата аудиту: 2026-04-16

## Що змінилося відносно попередньої версії цього аудиту

Попередній аудит уже не відповідав фактичному стану репозиторію. Після повторної звірки з кодом і тестами треба зафіксувати такі зміни:

- У canonical stack з'явився прямий structural anchor на кшталт `Program(y)`:
  - `omen_scale.py` формує `program_target_facts`;
  - будує `z_program`;
  - додає `program_anchor_loss`;
  - додає `program_decoder_ce`.
- World-model більше не навчається лише на проєкції token hidden states:
  - у `omen_scale.py` є `world_graph`;
  - `z` заземлюється через `z_graph`;
  - rollout може брати trace states і trace targets із `WorldGraphEncoder`.
- Saliency-рівень більше не є тільки анонімними слотами:
  - у `omen_saliency.py` введено канонічні ролі `agent`, `patient`, `action`, `modifier`, `coref`, `context`;
  - окремий тест перевіряє named predicates для цих ролей.
- Формалізований benchmark/eval layer уже існує:
  - `benchmarks/benchmark_omen_scale_eval.py`;
  - є synthetic benchmark;
  - є transfer-suite;
  - є локальний real-text protocol;
  - є multilingual метрики для Python / JavaScript / Rust.
- Abduce -> Deduce -> Induce вже не повністю прив'язаний лише до `self.training`:
  - у `DifferentiableProver` є `continuous_cycle_eval_enabled`;
  - є окремий тест на роботу гіпотезного циклу в `eval()`.
- Частина претензій до формул у попередньому аудиті стала застарілою:
  - `vfe_beta_kl`, `alpha`, `lambda_rule`, `lambda_tok`, `lambda_conc` реально проходять у `OMENScaleLoss`;
  - `program_anchor_weight` і `program_decoder_weight` теж реально входять у `total`.

Отже, старий висновок "багато важливих частин існують лише як намір" тепер занадто песимістичний. Головний борг змістився: зараз проблема не стільки у відсутності блоків, скільки у відсутності одного явно оголошеного канону і в неповній емпіричній доведеності на великому реальному масштабі.

## Обсяг перегляду

- 46 Python-файлів.
- Близько 30.5 тис. рядків Python-коду.
- Повторно звірено `concept.md` з ключовими файлами:
  - `omen_scale.py`
  - `omen_scale_config.py`
  - `omen_prolog.py`
  - `omen_saliency.py`
  - `omen_train_code.py`
  - `benchmarks/benchmark_omen_scale_eval.py`
  - `omen_symbolic/world_graph.py`
  - `tests/test_saliency_semantics.py`
  - `tests/test_omen_world_graph.py`
  - `tests/test_transfer_suite_protocol.py`
  - `tests/test_benchmark_protocol.py`
  - `tests/test_symbolic_cycle_eval.py`
- Запущено легкі релевантні тести:
  - `test_saliency_semantics.py` -> 1 test OK
  - `test_omen_world_graph.py` -> 3 tests OK
  - `test_transfer_suite_protocol.py` -> 2 tests OK
  - `test_benchmark_protocol.py` -> 2 tests OK
  - `test_symbolic_cycle_eval.py` -> 2 tests OK
- Разом: 10 lightweight tests passed.

## Принцип оцінки

- Нижче немає "як я би це проєктував".
- Нижче тільки звірка з тим, що реально заявлено в `concept.md`.
- Де код реалізує ідею буквально, це позначено окремо.
- Де код реалізує ідею через інженерну апроксимацію, це теж позначено окремо.

## 1. Поточний вердикт

Поточна кодова база вже істотно ближча до `concept.md`, ніж це випливало з попередньої редакції аудиту.

Стан на зараз такий:

- `omen_scale.py` є фактичним canonical runtime контуром системи.
- Ранні пласти OMEN не зникли, але вже не визначають основний стан репозиторію.
- Більшість великих концептуальних блоків із `concept.md` уже не декоративні:
  - M-Core;
  - WorldRNN;
  - S-Core / `∂-Prolog`;
  - NET;
  - EMC;
  - OSF;
  - saliency bridge;
  - creative engines;
  - structural anchoring;
  - world-graph grounding;
  - benchmark / transfer protocol.

Тому головний висновок треба оновити:

- Кодова база вже не просто "концептуально насичена, але не зібрана".
- Вона вже має досить сильний canonical stack.
- Головний концептуальний борг тепер у трьох місцях:
  - репозиторій досі не проголошує один офіційний архітектурний канон достатньо явно;
  - частина раннього "чистого" OMEN і пізнього OMEN-Scale співіснують як два наративи;
  - масштабна перевірка на великих реальних корпусах усе ще слабша за амбіцію документа.

## 2. Що зараз добре узгоджено з `concept.md`

### 2.1. Відмова від зовнішніх солверів збережена

У `concept.md` одна з базових тез: не покладатися на зовнішній solver на кшталт Z3.

У коді це дотримано:

- окремого SMT/Z3 стеку немає;
- логічний шар реалізований власним шляхом через `omen_prolog.py`;
- абдукція, верифікація, пошук, induction loop і symbolic control живуть усередині репозиторію.

Цей пункт узгоджений добре.

### 2.2. M-Core реалізований сильно

Ідея тензорної пам'яті з outer product і surprise/confidence-guided записом у коді не втрачена.

У репозиторії це представлено щонайменше двома поколіннями:

- `TensorProductMemory` у `omen_v2.py`;
- `AsyncTensorProductMemory` у `omen_scale.py`.

Що узгоджується з концептом:

- фіксований тензорний розмір;
- окремі read/write шляхи;
- surprise/confidence впливає на запис;
- є episodic recall;
- є інтеграція з symbolic layer.

За M-Core код і далі виглядає одним із найсильніших місць усієї системи.

### 2.3. S-Core / `∂-Prolog` перейшов від метафори до реально діючого ядра

У `concept.md` є еволюція від ранньої "символьної надбудови" до повноцінного theorem prover.

У коді це вже не просто намір:

- `Const`, `Var`, `Compound`;
- `Substitution`;
- `unify_mm`;
- Horn atoms / Horn clauses;
- `DifferentiableProver`;
- `TensorKnowledgeBase`;
- trace-aware reasoning;
- induction / verification / VeM.

Тобто критичний розрив між раннім концептом і реальною FOL-уніфікацією вже закритий.

### 2.4. Saliency bridge тепер ближчий до концепту, ніж це було в старому аудиті

Старий аудит справедливо критикував saliency за анонімні `role_*` слоти. Це більше не точний опис системи.

У поточному коді:

- `omen_saliency.py` має `CANONICAL_ROLE_NAMES`;
- перші ролі мають явні імена:
  - `agent`
  - `patient`
  - `action`
  - `modifier`
  - `coref`
  - `context`
- ці ролі мають named predicates у symbolic space;
- `tests/test_saliency_semantics.py` прямо перевіряє, що saliency видає named facts для канонічних ролей.

Уточнення:

- якщо `saliency_role_slots` більший за 6, додаткові канали все ще стають `role_n`;
- отже повна семантична онтологія ролей ще не закрита до кінця;
- але стара формула "тільки анонімні слоти" вже невірна.

### 2.5. Structural anchor типу `Program(y)` уже існує в canonical stack

Це одна з найважливіших змін проти старого аудиту.

Раніше було правдою, що structural supervision існувала тільки боковими каналами. Тепер це вже не так.

У `omen_scale.py`:

- будується `task_context`;
- із нього вибираються `program_target_facts`;
- prover grounding формує `z_program`;
- додається явний `program_anchor_loss`;
- додається `program_decoder_ce`;
- обидва терми йдуть у `total`.

Важливе уточнення:

- це не буквальний ранній `Program(y)` як окремий незалежний encoder, натренований зовсім окремо від тексту;
- але це вже прямий structural anchor між латентним станом і execution/trace/symbolic structure.

Тобто цей концептуальний борг уже не можна формулювати як "якоря немає". Правильно формулювати так: якір є, але його реалізація пішла через `task_context + prover grounding`, а не через ранню чисту формулу.

### 2.6. WorldRNN тепер частково заземлений у trace/world graph, а не лише в hidden states

Це друга велика зміна проти старого аудиту.

У поточному `omen_scale.py`:

- є `_build_world_graph_batch()`;
- є `WorldGraphEncoder`;
- у `z` домішується `z_graph` через `_ground_world_state()`;
- `_world_rollout_from_hidden()` міксує:
  - teacher states з `world_target_proj(h_tok)`;
  - pooled graph states;
  - trace states;
  - trace targets.

Крім того:

- `tests/test_omen_world_graph.py` перевіряє, що:
  - trace supervision реально витягується;
  - `z_graph` і `z_program` реально виходять із `forward()`;
  - rollout може використовувати trace targets як primary targets.

Отже стара критика "WorldRNN дивиться тільки на проєкцію token hidden states" теж уже застаріла.

Нова, точніша формула така:

- WorldRNN усе ще частково спирається на hidden-state projection;
- але canonical stack уже явно домішує graph/trace supervision;
- отже "світовість" WorldRNN помітно посилена, хоча ще не доведена до повністю чистого execution-state world model.

### 2.7. Формульні коефіцієнти в `OMENScaleLoss` тепер реально працюють

Старий аудит окремо критикував те, що частина коефіцієнтів живе тільки в конфігу.

Після повторної перевірки це твердження потрібно суттєво пом'якшити.

У поточному `OMENScaleLoss` реально використовуються:

- `vfe_beta_kl`;
- `alpha` як вага memory-read likelihood;
- `lambda_rule`;
- `lambda_tok`;
- `lambda_conc`;
- `program_anchor_weight`;
- `program_decoder_weight`.

Тобто тепер некоректно писати, що формульні коефіцієнти лишилися декоративними.

Коректніше так:

- значна частина coefficients уже активна;
- семантика деяких членів усе ще є інженерною адаптацією, а не буквальним переписом ранньої формули;
- але це вже не "мертві поля конфігу".

### 2.8. Benchmark / transfer protocol уже існує як окремий шар системи

Старий аудит стверджував, що formal benchmark-suite ще не сформований. Це більше не точний опис.

У поточному коді є:

- `benchmarks/benchmark_omen_scale_eval.py`;
- `run_benchmark()`;
- `build_transfer_tasks()`;
- `run_transfer_suite()`;
- synthetic tasks;
- multilingual tasks;
- local real-text corpora path;
- transfer deltas для Python / JavaScript / Rust / multilingual.

Також є тести:

- `tests/test_transfer_suite_protocol.py`;
- `tests/test_benchmark_protocol.py`.

Отже тепер правильно казати так:

- benchmark protocol уже є;
- але він ще не доріс до масштабного pipeline під The Stack / CodeParrot / CommonCrawl і великих зовнішніх репліковних експериментів.

### 2.9. Abduce -> Deduce -> Induce уже частково живе в eval/inference

Старий аудит був надто жорсткий у твердженні, що цикл майже весь прив'язаний до `self.training`.

Поточний стан точніший:

- у `DifferentiableProver` є `continuous_cycle_enabled`;
- є `continuous_cycle_eval_enabled`;
- hypothesis cycle може запускатися в `eval()` якщо конфіг це дозволяє;
- це перевіряється тестом `tests/test_symbolic_cycle_eval.py`.

Отже:

- "always-on reasoning" і далі не є повністю тотожним онлайн-оновленню ваг;
- але цикл гіпотез і перевірки вже не обмежений лише training mode.

## 3. Що досі лишається частковим або концептуально незавершеним

### 3.1. Репозиторій усе ще багатоканонічний

Це досі головний концептуальний борг.

У репозиторії одночасно існують:

- ранній OMEN у `omen_v2.py`;
- інтегрований OMEN-Scale у `omen_scale.py`;
- окремі/паралельні symbolic експерименти;
- кілька поколінь наративу про те, що є "справжнім" ядром системи.

Фактичний канон уже де-факто змістився в `omen_scale.py`, але це ще не оформлено достатньо жорстко на рівні репозиторної осі.

### 3.2. `z` усе ще не є буквально "графом світу"

Навіть після появи `world_graph` центральний латентний стан у коді залишається щільним вектором.

Поточна схема така:

- нейронний `z`;
- окремий `z_graph`;
- окремий `z_program`;
- symbolic / trace / saliency шари;
- змішування цих рівнів уже в `omen_scale.py`.

Тобто концепт реалізовано сильніше, ніж раніше, але все ще не буквально в дусі ранньої фрази "`z_t` є граф причин, об'єктів і властивостей".

### 3.3. World model уже заземлений, але ще не є чисто execution-state driven

Зараз модель робить важливий крок у правильний бік:

- trace states і graph states реально беруть участь у training target;
- world rollout telemetry віддається назовні;
- є окремі параметри міксування `world_graph_teacher_mix`, `world_graph_pooled_mix`, `world_graph_trace_mix`.

Але навіть у цій версії:

- частина teacher target усе ще походить із `world_target_proj(h_tok)`;
- тобто нейронна проєкція прихованих станів не прибрана з центру повністю.

Отже, борг тут не зник, а зменшився.

### 3.4. Online reasoning уже є, але online learning у сильному сенсі ще обмежений

Навіть після появи eval-cycle лишається межа між:

- online symbolic restructuring;
- online weight update.

На практиці:

- symbolic cycle у `eval()` уже можливий;
- reasoning, query routing, abduction candidates, world grounding працюють під час inference;
- але частина loss-driven adaptation, VeM penalty і policy-style навчання все ще природно лишаються training-time механізмами.

Тобто теза з `concept.md` про "постійний self-updating loop" уже частково реалізована на рівні правил і reasoning path, але не доведена до режиму повністю автономної безперервної перебудови всього параметричного стану.

### 3.5. Benchmark layer є, але ще не доводить масштаб заяв із `concept.md`

Наявність benchmark-suite вже не проблема.

Проблема тепер інша:

- немає великого стандартного pipeline під The Stack / CodeParrot / CommonCrawl;
- немає репозиторно закріпленого великого протоколу довгого навчання;
- немає важкого зовнішнього звіту з міжмовного узагальнення на масштабному реальному корпусі.

Отже, evaluation story уже існує, але ще не доводить всю силу концепту в повному масштабі.

### 3.6. Ранній dual-stream / graph-attention / causal-graph decoder більше не канон, але це не проговорено явно

Це не баг рантайму, а концептуальна неясність репозиторію.

Ранні секції `concept.md` дуже сильно асоціюють OMEN із:

- `DualStreamAttention`;
- `GraphAttentionEncoder`;
- `CausalGraphDecoder`.

У коді ці речі не зникли з історії, але canonical stack вже переїхав в іншу архітектурну точку:

- token/NET;
- perceiver concept layer;
- world graph grounding;
- symbolic/task/trace context;
- prover / EMC / OSF.

Якщо це і є новий канон, це варто вважати не технічним багом, а ще не закритим концептуальним рішенням.

## 4. Які твердження старого аудиту тепер треба вважати застарілими

Нижче перелік тверджень зі старого аудиту, які тепер уже не можна лишати в такому формулюванні.

### 4.1. "Saliency має лише анонімні role slots"

Це вже не так.

Тепер правильно так:

- перші ролі канонічно названі;
- додаткові канали лишаються fallback-слотами.

### 4.2. "У canonical sense бракує прямого `Program(y)`"

Це вже не так у старому формулюванні.

Тепер правильно так:

- буквального раннього `Program(y)` encoder немає;
- але structural anchor через `program_target_facts -> z_program -> program_anchor_loss` уже є.

### 4.3. "WorldRNN навчається тільки на hidden-state projection"

Це вже не так.

Тепер правильно так:

- hidden-state projection лишився;
- але поверх нього вже є pooled graph states і trace supervision.

### 4.4. "Формули та коефіцієнти не живуть у loss"

Це вже занадто сильне твердження.

Тепер правильно так:

- значна частина коефіцієнтів реально активна;
- ще лишаються інженерні адаптації семантики;
- але декоративною оболонкою це вже назвати не можна.

### 4.5. "Немає формалізованого benchmark protocol"

Це вже просто невірно.

Правильно так:

- benchmark protocol є;
- він поки що не великий і не зовнішньо-масштабний.

### 4.6. "Abduce -> Deduce -> Induce прив'язаний лише до training"

Це вже теж неточно.

Правильно так:

- повний weight-learning cycle лишається training-centric;
- але symbolic hypothesis cycle уже може працювати в `eval()`.

## 5. Реальні пріоритети, якщо продовжувати рух строго за `concept.md`

Якщо не фантазувати, а буквально дивитися на документ і поточний код, найраціональніші пріоритети зараз такі.

### 5.1. Явно оголосити канон

Потрібно жорстко зафіксувати:

- що є canonical stack;
- які файли є legacy / historical / experimental;
- як саме `omen_v2.py` співвідноситься з `omen_scale.py`.

Без цього навіть сильна реалізація виглядає концептуально роздвоєною.

### 5.2. Дотиснути world-state supervision ще далі від hidden-state proxy до trace/world-state центру

Зараз цей рух уже почався і він правильний.

Наступний логічний крок за документом:

- менше залежати від `world_target_proj(h_tok)`;
- більше залежати від execution trace / world graph / symbolic transition states як primary supervision.

### 5.3. Дотиснути online cycle від "eval-capable symbolic loop" до справді постійного adaptive loop

Тут уже є хороший фундамент:

- continuous cycle;
- eval path;
- symbolic induction metrics;
- abduction candidates;
- query/meta routing.

Наступний концептуальний крок:

- зробити online induction/repair ще більш типовим режимом inference;
- не обмежувати "самоперебудову" лише training-time сценаріями.

### 5.4. Доростити benchmark story до заявленого масштабу

Поточний benchmark layer вже хороший як інфраструктурний каркас.

Але щоб відповідати найсміливішим формулюванням `concept.md`, потрібні:

- великі реальні корпуси;
- зафіксовані transfer protocols;
- відтворювані великі звіти;
- окремий evaluation narrative на рівні репозиторію.

## 6. Підсумковий вердикт одним абзацом

Поточна кодова база OMEN вже істотно сильніша і концептуально зібраніша, ніж це відображала попередня редакція `concept_conceptual_audit.md`: у ній уже є прямий structural anchor, world-graph grounding, trace-supervised WorldRNN, named semantic roles, formal benchmark/transfer protocol і eval-time symbolic cycle. Тому головний концептуальний борг зараз уже не у відсутності ключових механізмів, а в незакритому питанні канону, у частковій залежності world-model від hidden-state proxies і в ще недостатньо масштабному емпіричному доведенні заяв `concept.md`.
