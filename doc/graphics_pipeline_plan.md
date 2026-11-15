# План: графический конвейер (VS + PS) в kernel_slicer

Рабочий документ. Состояние на 2026-11-15:
- этап 1 (CPU) выполнен;
- этапы 2 и 3 (фронтенд и Vulkan/Slang, Vulkan 1.0) работают для тестов 080, 081 и демо 31, 32, 33 (`ktest.py` OK, без ошибок validation layers);
- шаги 4a.1 (текстуры в Slang) и 4a.2 (GLSL-бэкенд) выполнены: все GP-примеры проходят на Slang и GLSL;
- dynamic rendering (`-gp_dynamic_rendering 1`) выполнен, тесты проверяют оба варианта Vulkan;
- этап 4 (ISPC), первая версия, выполнен: все GP-примеры проходят на ISPC, результат почти побитово совпадает с C++.

Реализация в трансляторе: [graphics_pipeline.h](../graphics_pipeline.h), [graphics_pipeline.cpp](../graphics_pipeline.cpp),
шаблоны `templates/inc_gp_vulkan.h/.cpp`, `templates_slang/generated_gp.slang`. Остальной код транслятора затронут точечными хуками.

## 0. Принятые решения

| Вопрос | Решение |
|--------|---------|
| API | Вариант A: команды `LiteRS` в управляющей функции явно называют шейдеры |
| Пространство имён, файл | `LiteRS`, [apps/LiteMathAux/raster.h](../apps/LiteMathAux/raster.h) |
| Шейдеры | const-функции главного класса `kernelVS_*`, `kernelPS_*`, пишутся на одну вершину / один пиксель |
| Данные вершин | VS сам читает члены класса по номеру вершины (vertex pulling), описания формата вершин нет |
| Состояние конвейера | объект `LiteRS::Pipeline` — член класса, настраивается в обычных host-функциях |
| Соглашения | Vulkan: 0 ≤ z ≤ w, NDC y = -1 — верхняя строка, центр пикселя +0.5, `VkFrontFace`; GL-матрицы LiteMath домножаются на `OpenglToVulkanProjectionMatrixFix()` |
| Варьинги | только поля float, float2, float3, float4; все интерполируются с перспективной коррекцией |
| Лейаут изображений на CPU | `LiteImage::Layout::TILED4x4` в `Image2D` (одноуровневые тайлы 4x4, адрес — сдвиги и маски) |
| Индексированная отрисовка | есть (`DrawIndexedTriangles`) |
| Клиппинг | в CPU-эталоне есть (near/far/guard band), в ISPC откладывается |

### API в одном месте

```cpp
// host-функция (не управляющая): настройка
m_pipeline.SetColorTarget(m_color);        // Image2D<float4> или Image2D<uint32_t> (RGBA8), индекс цели 0..3
m_pipeline.SetDepthTarget(m_depth);        // Image2D<float>
m_pipeline.SetViewport(0, 0, w, h);
m_pipeline.cullMode = LiteRS::CullMode::BACK;

// управляющая функция
LiteRS::Clear(m_pipeline);
LiteRS::DrawTriangles(m_pipeline, this, &C::kernelVS_A, &C::kernelPS_A, vertNum, args...);               // VS + PS
LiteRS::DrawTriangles(m_pipeline, this, &C::kernelVS_Depth, vertNum, args...);                           // только глубина
LiteRS::DrawIndexedTriangles(m_pipeline, this, &C::kernelVS_A, &C::kernelPS_A, m_indices.data(), n, args...);
LiteRS::DrawIndexedTriangles(m_pipeline, this, &C::kernelVS_Depth, m_indices.data(), n, args...);
kernel2D_Post(w, h, a_out);                // обычный кернел читает цели как Image2D

// шейдеры
float4 C::kernelVS_A(uint32_t a_vertId, Args... a_args, Varyings* a_out) const; // возвращает clip-позицию
float4 C::kernelVS_Depth(uint32_t a_vertId, Args... a_args) const;
float4 C::kernelPS_A(const Varyings& a_in) const;                                // цвет цели 0
GBuf   C::kernelPS_B(const Varyings& a_in) const;                                // структура из float4: цели 0, 1, ...
```

Дополнительные аргументы `args...` команды отрисовки передаются в VS. PS получает данные только через варьинги и
члены класса. Шейдеры — const: они выполняются параллельно и не меняют данные класса.

## 1. Этап 1 — CPU-эталон, тесты, демо (выполнен)

### 1.1 LiteRS (CPU)

- Фиксированная точка (8 бит субпикселя, int64 для функций рёбер), правило top-left.
- Клиппинг Сазерленда–Ходжмана по near, far и guard band (±64w) с интерполяцией варьингов.
- Отбраковка граней по `VkFrontFace`.
- Два алгоритма растеризации: попиксельный half-space и блоки 4x4 (пустой/полный/частичный блок). Выбор AUTO по
  площади bbox (`Pipeline::cpu.smallTriArea`, по умолчанию 64).
- Многопоточность: sort-middle, корзины 64x64, OpenMP; порядок треугольников в пикселе сохраняется.
- Ранний тест глубины (PS не меняет глубину и не делает discard), смешивание NONE / ALPHA / ADD, до 4 целей цвета.
- Индексированная отрисовка: VS выполняется по одному разу для каждой вершины из диапазона [min, max] индексов.
- Цели могут быть линейными или тайловыми (все цели одного прохода — с одинаковыми размером и лейаутом).
- Под `KERNEL_SLICER` транслятор видит только типы и объявления команд, а реализация скрыта.

### 1.2 Тайловый лейаут Image2D (сабмодуль LiteMath, коммит делает автор сабмодуля)

- `enum class Layout { LINEAR, TILED4x4 }`, `resize(w, h, layout)`, `setLayout()`, `layout()`, `pitch()`, `offset(x,y)`.
- Свободная функция `PixelOffset(layout, pitch, x, y)`.
- `operator[]` и `sample()` (bilinear и nearest) учитывают лейаут.
- `SaveImage` для тайловых изображений сначала переводит их в линейный вид.
- Арифметические операторы сохраняют лейаут. `MSE` и новая `LinearPixels()` работают в линейном виде.
- По умолчанию LINEAR, поэтому старый код не меняется.

### 1.3 Примеры

| Пример | Что проверяет |
|--------|---------------|
| `apps/tests/080_gp_triangle` | один треугольник, неиндексированная отрисовка, аргумент VS, тайловые цели, обычный кернел после отрисовки |
| `apps/31_gp_textured_cube` | индексированная отрисовка, текстура (`Image2D<uint32_t>` + `Sampler`) в PS, цель RGBA8, отбраковка задних граней |
| `apps/32_gp_shadow_map` | два объекта `Pipeline`, проход только с глубиной в карту теней 2048², `CullMode::FRONT`, чтение карты теней в PS (PCF 3x3) |
| `apps/33_gp_deferred_ao` | 3 цели цвета (PS возвращает структуру), затем три обычных кернела: SSAO, размытие, освещение |
| `apps/LiteMathAux/raster_test` | юнит-тесты растеризатора и тайлового `Image2D`; `--bench` — замеры алгоритмов |

Общая процедурная геометрия: [apps/utils/simple_meshes.h](../apps/utils/simple_meshes.h).

Сборка CPU-версии: `mkdir build && cd build && cmake -DUSE_VULKAN=OFF .. && make`, запуск из папки примера,
результат `zout_cpu.bmp`. В `ktest.json` включены Slang- и GLSL-варианты.

## 2. Этап 2 — фронтенд транслятора (общий для всех бэкендов)

1. **Шейдеры как новый вид кернелов.**
   - Добавить префиксы `kernelVS_` и `kernelPS_` в `POSSIBLE_KERNEL_NAMES` ([kslicer_def.cpp](../kslicer_def.cpp)).
   - В `KernelInfo` добавить поле стадии (`NONE / VERTEX / PIXEL`) и описание сигнатуры: номер вершины, пользовательские аргументы, выход-варьинги, вход PS, тип результата PS.
   - Тела транслируются так же, как member-функции и RTV-кернелы без `tid` (циклов-потоков нет).
   - `IsKernel` и `RemoveKernelPrefix` должны работать без изменений в остальном коде.
2. **Распознавание команд в управляющей функции** ([ast_matchers.cpp](../ast_matchers.cpp), initial pass).
   - Матчеры `callExpr` на `LiteRS::Clear / DrawTriangles / DrawIndexedTriangles`.
   - Из аргументов извлекаются член-`Pipeline` (MemberExpr), указатели на методы (`&C::kernelVS_*` → CXXMethodDecl), буфер индексов (`m_vec.data()`), выражение количества и доп. аргументы.
   - Функция с командами LiteRS считается управляющей, даже если в ней нет обычных кернелов. VS и PS добавляются в `UsedKernels`.
   - В `MainFuncInfo` появляется список `DrawCallInfo` с диапазонами исходного кода, чтобы потом переписать вызов.
3. **Члены класса.**
   - `LiteRS::Pipeline` — host-only объект, его надо исключить из UBO. Сейчас он туда попадает: проверено, `Pipeline m_pipeline` оказывается в `*_UBO_Data`.
   - Цели находятся статически: матчер на `Pipeline::SetColorTarget / SetDepthTarget` в любых методах класса с аргументом — членом `Image2D`. Это даёт пары (pipeline, индекс цели) → член-изображение и его формат.
   - Сгенерированный код в `CommitDeviceData()` проверяет `assert`-ом, что указатели в `Pipeline` указывают на те же члены.
4. **Типы.** Структуры варьингов и выходов PS выгружаются в общий заголовок шейдеров. Проверки с понятными ошибками:
   - поля только float-типов;
   - результат PS — `float4` или структура из `float4`, не больше 4 целей;
   - последний аргумент VS с варьингами — указатель;
   - шейдеры — const;
   - для прохода только с глубиной VS без варьингов.
5. **Используемые данные шейдеров** (векторы, текстуры, POD-члены) собираются тем же механизмом, что и для кернелов. Разница в том, что набор дескрипторов виден стадиям VERTEX|FRAGMENT.
6. **TINYSTL.** Добавлять интерфейсы (без реализации), которых не хватает host-коду примеров: `std::move` (сейчас в `Image2D::setLayout` обходится копированием), `vector::insert(it, initializer_list)`, `<random>`. В примерах сейчас стоят обходы, после расширения TINYSTL их можно убрать. Сам `raster.h` остаётся разделённым по `KERNEL_SLICER`: транслятору нужны только объявления команд.

Критерий готовности этапа: для всех четырёх примеров в логе kslicer видны найденные VS/PS и команды, `Pipeline` не
попадает в UBO, структуры варьингов есть в общем заголовке.

## 3. Этап 3 — Vulkan (сначала Slang, затем GLSL)

1. **Шейдеры.**
   - Новый шаблон `templates_slang/generated_gp.slang` с точками входа `[shader("vertex")]` и `[shader("fragment")]` (или два шаблона).
   - Вход VS — `SV_VertexID`. Выход VS — структура `{ float4 pos : SV_Position; Varyings v : TEXCOORD0..N }`.
   - Выход PS — `SV_Target0..N`.
   - Те же объявления буферов, текстур и UBO, что у compute-кернелов. Доп. аргументы VS идут в push constants.
   - `build_slang.sh` компилирует с `-stage vertex/fragment -entry ...`.
   - GLSL: шаблоны `.vert` / `.frag`.
2. **Изображения-цели.**
   - Usage (`template_rendering_host.cpp`, сейчас `local["Usage"]`): `COLOR_ATTACHMENT` или `DEPTH_STENCIL_ATTACHMENT` + `SAMPLED`/`STORAGE` по использованию.
   - Глубина — `D32_SFLOAT`. Чтение глубины в шейдерах (карта теней в PS, `m_shadowMap[int2]`) — через sampled-вид с DEPTH aspect, т.к. storage для D32 обычно не поддерживается.
3. **Пайплайны.** По умолчанию только возможности Vulkan 1.0 (кроссплатформенность):
   - `VkRenderPass` + `VkFramebuffer` на каждый объект `LiteRS::Pipeline` (набор целей и их форматы известны статически);
   - одна `VkPipeline` на каждую команду отрисовки (VS, PS, render pass); состояние `cullMode`, `frontFace`, depth, blend
     запекается из `LiteRS::Pipeline` при создании; viewport и scissor — dynamic state (есть в 1.0);
   - пайплайны создаются в `CommitDeviceData()`, когда значения `LiteRS::Pipeline` уже заданы на host. Изменение
     состояния после этого требует повторного `CommitDeviceData()` (или отдельной функции пересоздания).
   - Опция `-gp_dynamic_rendering 1` (сделано, см. 4a): Vulkan 1.3, dynamic rendering и extended dynamic state для
     cull/depth, тогда render pass и framebuffer не нужны, а состояние можно менять без пересоздания пайплайнов.
4. **Команды.** Каждая команда LiteRS переписывается в `tr_host_vulkan.cpp` (по аналогии со служебными вызовами `memcpy`/`sort`, `VisitCallExpr`):
   - `Clear` — в первой версии `vkCmdClearColorImage` / `vkCmdClearDepthStencilImage` (render pass с `loadOp = LOAD`);
     оптимизация позже: слить с `loadOp = CLEAR` следующего прохода той же цели;
   - `Draw*`:
     - переходы лейаутов (`GENERAL` ↔ `COLOR_ATTACHMENT_OPTIMAL` / `DEPTH_ATTACHMENT_OPTIMAL`);
     - `vkCmdBeginRenderPass`, bind pipeline и набора дескрипторов (graphics bind point), push constants;
     - `vkCmdDraw` или `vkCmdBindIndexBuffer + vkCmdDrawIndexed` (буфер индексов получает usage `INDEX_BUFFER`);
     - `vkCmdEndRenderPass`.
   - Барьеры compute → vertex/fragment и attachment → compute: расширить существующую логику барьеров (`vk_class_init.cpp`, около строки 1550).
5. **Тайловые Image2D на GPU.** Генерированные копирования (`UpdateTextureMembers`, `vk_class.cpp:246` использует `data()`) и чтение изображений обратно должны линеаризовать тайловые данные (`LiteImage::LinearPixels`). То же — во всех местах бэкендов, где используется `Image2D::data()`.
6. **Таймстемпы** (`-timestamps 1`): время каждой отрисовки по имени `DrawTriangles_<VS>_<PS>` или по имени PS.
7. **Проверка.**
   - Снять `skip` в `ktest.json`. Сравнение CPU/GPU — по доле различающихся пикселей (< 0.5%), т.к. правило заполнения и точность растеризации GPU могут отличаться на рёбрах.
   - Без ошибок validation layers.
   - Slang и GLSL оба.
   - Добавить конфигурации в `.vscode/launch*.json`.
   - Строки в [doc/features.md](features.md): `gp_draw`, `gp_indexed`, `gp_depth_only`, `gp_mrt`, `image_layout_tiled`.

## 4. Этап 4 — ISPC

**Состояние (2026-11-15): первая версия сделана**, все GP-примеры проходят (`ktest.py -b ispc`), отличие от C++-эталона —
не больше нескольких пикселей на 1/255. Как устроено:
- ISPC-бэкенд не переписывает управляющие функции (выполняется C++-версия базового класса), поэтому команды LiteRS
  перехватываются в библиотеке: у `LiteRS::Pipeline` есть указатель `backend` (`LiteRS::DrawBackend`), генерированный
  класс `<Class>_ISPC` наследует `DrawBackend`, сопоставляет шейдеры команды по ключам (`LiteRS::MakeShaderKey` — байты
  указателя на функцию-член) и выполняет отрисовку сам; незнакомые пары шейдеров рисует C++-реализация;
- VS — генерированная `<Draw>_VS_ISPC` (`foreach` по вершинам, порции по 4096 вершин в OpenMP), результат —
  `LiteRS::Vertex<detail::FloatArray<N>>`, варьинги как массив float;
- сборка треугольников, клиппинг (полный, как в C++), setup и раскладка по корзинам — в C++
  (`detail::DrawVertexStreamByBins`), корзины обрабатываются параллельно OpenMP;
- `<Draw>_Raster_ISPC` растеризует треугольники корзины блоками 4x4 (ганг обрабатывает 16 пикселей блока), функции
  рёбер int64 с той же фиксированной точкой (покрытие совпадает с C++ побитово), тест глубины, интерполяция с
  перспективной коррекцией, вызов PS, смешивание, упаковка RGBA8 (`apps/LiteMathAux/raster_ispc.h`);
- для этого в ISPC-бэкенде добавлены `Image2D`-члены (`apps/LiteMathAux/LiteImageISPC.h`: чтение/запись `[]`, `sample`
  для `uint32_t` и `float4`, лейауты LINEAR и TILED4x4), недостающие функции `LiteMathISPC.h` (`float4x4*float4`,
  `rotate4x4*`, `normalize(float4)`, `lerp`, `cross(float4)`), переписывание тел локальных функций и удаление `std::`.
  Буферы `float2`/`float3` шейдерам передаются копией с шагом 16 байт (в ISPC `float<2>` и `float<3>` занимают 16 байт).

Осталось (оптимизации, исходный план ниже):
- попиксельный путь для мелких треугольников и выбор AUTO по порогу, пропуск пустых и полных блоков по углам;
- выровненные векторные load/store для TILED4x4 вместо gather/scatter;
- задачи ISPC вместо OpenMP; замеры (`raster_test --bench` для ISPC).
Сейчас ISPC быстрее C++ до 2 раз (app 33: 41 → 21 мс, app 32: 12.7 → 8.6 мс).

Исходный план:

1. **Структура.**
   - Общая часть растеризатора (setup, раскладка по корзинам, обход блоков) — библиотека на ISPC + C++ рядом с `raster.h` (например `apps/LiteMathAux/raster_ispc.h/.ispc`). Она параметризуется сгенерированным кодом.
   - VS — `foreach` по вершинам (как IPV 1D), результат во временный буфер вершин.
   - PS — varying-функция, которую генерированная функция `Draw_<VS>_<PS>_ISPC` вызывает на ганге пикселей.
2. **Блок 4x4 = тайл.**
   - Маска покрытия считается как varying по 16 пикселям тайла (AVX2 `i32x8` — 2 итерации, AVX-512 `x16` — 1).
   - Тест глубины маскированный, интерполяция варьингов varying, вызов PS на ганге, запись в 16 подряд идущих пикселей цели TILED4x4 (выровненные векторные load/store). Здесь окупается тайловый лейаут.
3. **Мелкие треугольники.**
   - Попиксельный путь: ганг по пикселям bbox (bbox ≤ 16 пикселей — одна итерация).
   - Порог AUTO подобрать бенчмарком на ISPC по аналогии с `raster_test --bench`. В скалярном C++ блоки выигрывают только с ~48–64 px, в ISPC порог должен быть заметно ниже.
4. **Потоки.** Сначала sort-middle по корзинам 64x64 через задачи ISPC (`ispc_tasksys.cpp`); setup и раскладку по корзинам тоже распараллелить. Позже можно сравнить с sort-last с блокировками тайлов (по статье).
5. **Клиппинг.** v1 — только тривиальное отбрасывание; треугольники, пересекающие near (w ≤ ε), отбрасываются с предупреждением в документации. v2 — порт клиппинга из `raster.h`.
6. **Изображения в ISPC** (текстуры в `doc/features.md` для ISPC сейчас ❔): `Image2D::sample`, `operator[]` с учётом лейаута, цели.
7. **Проверка.** Покрытие должно совпадать с C++-эталоном побитово (та же фиксированная точка). Цвет — с точностью до порядка операций float.

## 5. Этап 5 — документация

- Пользовательская документация (`doc/graphics_pipeline.md`): API, правила для шейдеров, ограничения.
- Справочник в skill: `.claude/skills/kernel-slicer/references/graphics.md`, ссылка в SKILL.md и в таблице образцов.
- Обновить `doc/features.md` и `README.md`.

## 4a. Ближайшие шаги и новые примеры

Порядок работ (решение от 2026-11-15):
1. ~~Текстуры в Slang-бэкенде~~ (выполнено 2026-11-15, сняты `skip` у 006, 008, 066, app 14, app 15, app 31, app 33):
   - вид текстуры в шейдере и дескриптор на хосте выбираются одной функцией `kslicer::ShaderTextureKind`: запись в кернеле —
     `RWTexture2D<T>` (storage image), `sample(sampler, uv)` — `Sampler2D<T>` (combined image sampler), остальное чтение —
     `Texture2D<T>` (sampled image); изображения, которые где-то пишутся, остаются в лейауте `GENERAL`;
   - `Image2D<uint32_t>`, который читается через `sample()`, на GPU — `R8G8B8A8_UNORM`, в шейдере `float4` (оба бэкенда).
     Чтение таких текстур на CPU после GPU не планируется: CPU пишет, GPU читает;
   - параметр-текстура функции: `Sampler2D<T>`, если функция делает `sample`, иначе `Texture2D<T>`, неконстантная ссылка —
     `RWTexture2D<T>`;
   - квалификаторы `LiteMath::`, `LiteImage::`, `LiteRS::` (и псевдонимы этих пространств) удаляются в шейдерах Slang/GLSL
     у вызовов, констант и типов локальных переменных; `LiteRS::Pipeline` исключается из UBO по каноническому типу, поэтому
     работают `using LiteRS::Pipeline`, `using LiteRS::DrawIndexedTriangles` и вызовы через ADL (тест 081);
   - `cross(float4, float4)` в Slang; `cross3(float4, float4)` в шейдерах теперь даёт `w = 0`, как в LiteMath (было `w = 1`).
2. ~~GLSL-бэкенд для графического конвейера~~ (выполнено 2026-11-15, сняты `skip` у 080, 081, app 31, 32, 33):
   - шаблон `templates_glsl/generated_gp.glsl`: файлы `kernelVS_*.vert` / `kernelPS_*.frag`, варьинги — структура с
     `layout(location = 0) out/in`, выходы PS — `layout(location = i) out vec4|uint`, доп. аргументы VS — push constants;
   - в `common_generated.glsl` для шейдеров графики набор дескрипторов `set = DSetId`, буферы и UBO `readonly`;
   - GLSL выбирает вид текстуры той же функцией `kslicer::ShaderTextureKind`, что и Slang: запись — `image2D`
     (`imageLoad/imageStore`), `sample` — `sampler2D`, остальное чтение — `texture2D` + `texelFetch`
     (`GL_EXT_samplerless_texture_functions`). Раньше чтение без сэмплера шло через storage image: это не работало для
     глубины D32 (карта теней) и для целей без `STORAGE` usage. Чтение `float`/`uint`-текстур берёт нужные компоненты (`.x`);
   - `cross(vec4, vec4)` в GLSL.
3. ~~Опциональный dynamic rendering~~ (выполнено 2026-11-15): опция `-gp_dynamic_rendering 1`, Vulkan 1.3:
   - нет `VkRenderPass`/`VkFramebuffer`, пайплайны создаются с `VkPipelineRenderingCreateInfo`, отрисовка —
     `vkCmdBeginRendering` с `loadOp = LOAD` для всех целей объекта `Pipeline` (в проходе без PS запись цвета выключена);
   - `cullMode`, `frontFace`, `depthTest`, `depthWrite`, `depthCompare` — динамическое состояние, читаются при каждой
     отрисовке без пересоздания пайплайнов; `blendMode` по-прежнему запекается (нужен extended dynamic state 3);
   - устройство создаётся с `<Class>_Generated_ListRequiredDeviceFeatures()` (Vulkan 1.3, `dynamicRendering`); `main.cpp`
     всех GP-примеров переведены на эту функцию;
   - в `ktest.json` каждого GP-примера четыре варианта: `slang`, `slang_dr`, `glsl`, `glsl_dr`.
4. ~~ISPC (этап 4), первая версия~~ (выполнено 2026-11-15): вариант `ispc` в `ktest.json` каждого GP-примера; кернелы
   примеров сделаны `virtual` (ISPC-класс их переопределяет), `main.cpp` и `CMakeLists.txt` поддерживают `--ispc` и `USE_ISPC`.

Новые тесты (`apps/tests`), по одной фиче, сравнение CPU и GPU (081 — пространства имён LiteMath/LiteRS — уже добавлен):

| # | Тест | Что проверяет |
|---|------|---------------|
| 1 | Перекрытие треугольников | `depthCompare` (`LESS`, `GREATER`, `ALWAYS`), `depthWrite = false` |
| 2 | Отсечение граней | `cullMode` NONE/BACK/FRONT, `frontFace` CW/CCW на сетке с обеими ориентациями |
| 3 | Смешивание | `BlendMode::ALPHA` и `ADD` на цели `float4`, порядок отрисовки |
| 4 | Клиппинг | треугольники через near/far и за guard band |
| 5 | Viewport | отрисовка в часть изображения, несколько viewport в одной цели |
| 6 | Несколько целей цвета | 2–4 цели (`float4` и RGBA8) без SSAO |
| 7 | Аргументы VS и uniform-данные | `float4x4` как аргумент отрисовки, члены класса через `UpdateMembersPlainData()` между кадрами |
| 8 | Отрисовки в цикле | разные объекты с разными аргументами VS, `Clear` между проходами, повторное использование наборов дескрипторов |
| 9 | Вершины от кернела | `kernel1D_` пишет вершины в `std::vector`, затем VS их читает (барьер compute → vertex) |

Тест тайлового и линейного лейаута отложен: тайловый лейаут ещё будет меняться (см. 5a).

Новые демо (`apps`):

| # | Демо | Что показывает |
|---|------|----------------|
| 11 | Wireframe | каркас поверх сплошной заливки; первый вариант — в PS по барицентрическим координатам (VS выдаёт барицентрики как варьинг): без новых фич Vulkan и без новых команд. Позже, при необходимости, `LiteRS::DrawLines` (`LINE_LIST`, Vulkan 1.0) с растеризатором отрезков на CPU; `polygonMode = LINE` требует `fillModeNonSolid` и в ядро 1.0 не входит |
| 14 | Постобработка | рендер во `float4`, затем bloom/тонмаппинг обычными кернелами (по мотивам app 05/28) |
| 16 | Instancing | много копий меша; нужно расширение API: номер экземпляра в VS (`DrawTrianglesInstanced` или аргумент) |
| 18 | Гибрид растеризации и трассировки лучей | G-буфер растеризацией, тени через `m_pAccelStruct->RayQuery_AnyHit` в обычном кернеле (ray query) |

## 5a. Позже: двухуровневый тайлинг

Корзины 64x64 (или 128x128), внутри них тайлы 4x4, как `FrameBufferTwoLvl` в SWGL. Корзины улучшают работу кэша
(в экспериментах SWGL выигрыш до 60% в одном потоке), а тайлы 4x4 позволяют обращаться к блоку одной векторной
командой в ISPC. Нужен новый `LiteImage::Layout` и исследование размера корзины (64 / 128) бенчмарком.

## 6. Риски и открытые вопросы

- Правило заполнения и субпиксельная точность GPU зависят от реализации: точного совпадения рёбер с CPU не будет, только допуск.
- Изменение `Image2D` в сабмодуле затрагивает все места, где `data()` трактуется как линейный буфер (копирование на GPU, ISPC, CUDA). Пока тайловый лейаут не включён явно, поведение прежнее.
- Отрисовка по частям буфера индексов (`m_indices.data() + first`) и цикл отрисовок по объектам в первую версию не входят. В примерах сцена собрана в один меш. Возможно расширение аргументом `firstIndex`.
- Чтение цели текущего прохода в его же PS (feedback loop) запрещено, транслятор должен выдавать ошибку.
- В управляющей функции используется `uint32_t(m_indices.size())`: это host-код, но надо проверить, что транслятор его не трогает.
