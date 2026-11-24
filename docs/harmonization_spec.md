# v2 跨库特征映射规范（harmonization spec）

- 日期：2026-10-18
- 代码：`src/icu/v2/harmonization.py`（只有映射表、换算和检查函数，**没有**接入 v1 的三个加载器）
- 测试：`tests/v2/test_harmonization.py`
- 相关：[ICDE2027_558_audit_and_revision_plan.md](ICDE2027_558_audit_and_revision_plan.md) 的 P1-1，[v2_engineering_brief.md](v2_engineering_brief.md)

## 依据与范围

本文的结论全部来自只读操作，没有扫描任何大表：

- 逐行读 v1 代码：`src/icu/mimic_iv_schema.py`、`mimic_iv_dataset.py`、`mimic_iv_io.py`、`eicu_dataset.py`、`hirid_dataset.py`、`shock_anchor_builder.py`，以及用 `git log`/`git show` 看 `hirid_dataset.py` 的历史版本。
- 小型字典文件：
  - MIMIC-IV 3.1 的 `icu/d_items.csv.gz`（58 KB）和 `hosp/d_labitems.csv.gz`（13 KB），在 zip 里直接读，没有解压。
  - HiRID 1.1.1 的 `hirid_variable_reference.csv`（31 KB）、`hirid_variable_reference_preprocessed.csv`、`ordinal_vars_ref.csv`。
  - eICU 目录下的 `eicu-code-master.zip` 和 `mimic-code-main.zip`：官方概念 SQL（`pivoted-lab.sql`、`pivoted-uo.sql`、`pivoted-infusion.sql`、`measurement/*.sql`、`medication/*.sql`、`treatment/ventilation.sql`）和 eICU 表文档（`website/content/eicutables/*.md`）。
- 极少量数据：
  - eICU `patient.csv.gz` 的前 2000 行，只统计 `uniquepid` 的格式，没有输出任何标识。
  - HiRID merged stage 一个 parquet 文件的 footer，只读列名。
  - `data/processed/hirid*` 里各 4 个 episode 的 json 和缺失掩码。
- 在内存里运行 v1 的 `_load_hirid_reference_feature_aliases`，以及它的两个历史版本（用 `git show` 导出到临时目录），输入是上面的参考 CSV。

**不在本文范围内**：没有改任何 v1 文件，没有重建任何数据集。需要扫原始大表才能确认的事项见第 8 节。

## 0. 结论

1. 审计列出的 eICU 和 HiRID 映射问题全部属实，只有一条需要修正措辞：harmonization manifest 并非「没有代码读取」，而是只有 `report_fields` 被读取，其中的映射内容（7 个共享特征、Lactate 单位）从未生效。
2. 另外发现几个审计没有列出、影响更大的问题，集中在 MIMIC-IV：
   - `urine_output` 漏掉了 Foley 和 Void 这两个主要尿量 itemid。
   - `ventilation_on` 实际只由 PEEP 和 FiO2 决定，两个文本型呼吸机条目被静默丢弃。
   - 血压只用了无创 itemid，有动脉置管的病人看起来像「没测血压」（第 3 节）。
3. 论文使用的 HiRID 已处理数据，生成时的映射代码比当前 HEAD 还要宽松：凝血因子会被算成 Platelets，中性粒细胞百分比会被算成 WBC，尿钠和尿素会被算成尿量。HiRID 必须重建。
4. HiRID 公开版的 merged stage 只有 18 个变量，其中唯一像升压药的是 dobutamine。v1 默认优先使用 merged stage，所以 v2 必须从 raw stage 构建。
5. v2 给三个源各建了一张显式映射表，覆盖全部 18 个特征：
   - 每一行都是带命名空间的确切标识，例如 HiRID 的 `("observation", 300)` 和 `("pharma", 300)` 是两行，不能用裸 ID 查表。
   - 每一行都写明源单位、置信度和相对 v1 的状态。
   - 已知的误映射都写成带理由的排除行。
   - `validate_mapping_tables()` 对三张表的检查结果为 0 个问题。

## 1. 审计说法逐条核实

| 编号 | 审计说法 | 结论 | 证据 |
|---|---|---|---|
| A0 | `docs/experiments/icde_transfer_harmonization_manifest.json`（7 个共享特征、Lactate 单位）没有被任何代码读取 | **部分成立** | `scripts/run_comparator_external_transfer.py:51` 会通过 `--transfer-manifest` 读这个文件，但只取 `report_fields`（:55-58）。`shared_feature_order` 和 `unit_normalization` 在所有 `.py`、`.ps1` 里都没有出现。`icde_comparator_execution_manifest.json` 里只记了文件名（`transfer_manifest_name`），`tests/test_external_transfer_config_contract.py:25` 也只检查这个键是否存在，没有代码按这个名字去加载文件。结论：映射内容从未生效。 |
| E1 | eICU 化验项按子串匹配，会混进尿肌酐、脑脊液葡萄糖、尿/体液白细胞 | **成立** | `src/icu/eicu_dataset.py:223-239`：`"creatinine" in text`、`"glucose" in text`、`"wbc" in text`。另外 `"glucose"` 子串还会收进 `bedside glucose`。据 eicu-code `notebooks/lab.ipynb` 的输出，它有 3,175,835 行，是 eICU 最多的化验项，而实验室 `glucose` 只有 1,319,496 行。具体有哪些 labname 被收进来，要扫 `lab.csv.gz` 才能列全（第 8 节 R-E1）。 |
| E2 | `urine_output` 有 `outputtotal` 时就取它（全部出量） | **成立，且更严重** | `eicu_dataset.py:421-426`：只要 `outputtotal` 非空就用它，只有为空时才回退到 `celllabel` 含 urine 的行。eICU 官方文档（`eicutables/intakeOutput.md`）写明：`outputTotal` 是截至该时刻的**累计**总出量，包含引流、失血等；同一时刻的多行会重复同一个值。v1 对每一行（包括入量行）都取它，矩阵构建时再按 bin 求和（`agg="sum"`），所以结果是累计值的若干倍。 |
| E3 | 升压药关键词包含 dobutamine | **成立** | `eicu_dataset.py:33`。 |
| E4 | `uniquepid` 是文本，病人 ID 退化成 stay ID | **成立** | `eicu_dataset.py:109` 用 `pd.to_numeric(..., errors="coerce")` 转 `uniquepid`，`:119` 再用 `stay_id` 填空。官方文档里 `uniquepid` 是 `varchar(10)`。抽查前 2000 行：可以转成数字的 0 行，格式全是 `ddd-d+`；2000 个 stay 只对应 1388 个病人。所以 v1 的 eICU「按病人划分」实际是按 stay 划分，同一病人会跨训练集和测试集。 |
| H1 | HiRID 别名表 | **成立** | `src/icu/hirid_dataset.py:23-68`。 |
| H2 | 单位检查 | **成立** | `hirid_dataset.py:195-209`。提示表 `:94-110` 里，WBC 允许 `g/l`（:107），Platelets 不允许（:108）。 |
| H3 | 「低置信」隔离（< 60，:413-416）永远不会触发，因为最低分是 65（:251-265） | **成立，且更强** | 来源分：observation 30、preprocessed 25、pharma 20。证据分：name_alias 70、name_keyword 60、description_infer 45。最低组合是 65 ≥ 60。实际上 `description_infer` 这一档根本到不了：`_canonical_feature_name` 自己在最后就会调用描述推断（:478），调用方（:376-378）一律把结果记成 `name_alias`。所以实际最低分是 80。在本机参考文件上运行 v1：低置信隔离 0 行，单位不符隔离 11 行。 |
| H4 | HiRID 没有通气映射 | **成立** | 没有任何别名、关键词或推断能得到 `ventilation_on`。实测 3845（Ventilator mode）、15001552（Ventilator Airway Code）、2600（PEEP setting）都映射为 None。 |
| H5 | Platelets 因单位 "G/l" 被隔离 | **成立** | 实测 20000110（`Platelets [#/volume] in Blood`，`G/l`）被判为 `unit_mismatch`，HEAD 下 Platelets 没有任何映射 ID。 |
| H6 | 尿肌酐、脑脊液葡萄糖/乳酸/pH、混合静脉血氧饱和度（当成 SpO2）被映射进来 | **成立，另有两项** | 实测：24000572、24000573 → Creatinine；24000400 → Glucose；24000564 → Lactate；24000608 → pH；4200 → SpO2。另外 20000800（动脉血气 SaO2）→ SpO2，24000741（引流液 pH）→ pH。推断逻辑在 `hirid_dataset.py:141-182`，比如 :154 只要同时有 oxygen 和 saturation 就判为 SpO2。 |
| H7 | dobutamine（pharmaid 426）算成升压药 | **成立** | 来源是 preprocessed 参考表的 `pm41,Dobutamine,...,426`：关键词命中在 :382-384，raw id 登记在 :427-432。raw 参考表里 426 的名字是 `Dobutrex 250 mg/20ml`，本身不含 dobutamine。实测 `426` 和 `pm41` 都映射为 `vasopressor_on`。 |
| H8 | propofol（Disoprivan）和 midazolam（Dormicum）没有映射到镇静 | **成立，另有 dexmedetomidine** | 关键词只有通用名（:84-92），而 HiRID 用商品名：Disoprivan、Dormicum、Dexdor。实测 v1 的 `sedation_on` 只有 7 个 pharmaid。其中 6 个是阿片类制剂：芬太尼注射液、推注、泵入、PCA 四种，`Sufentanyl Perfusor`（被子串 fentanyl 命中），`Morphine Inj`。剩下 1 个是误判：1000843 `PDA-Bupivacain 0.125% ohne Fentanyl`（硬膜外布比卡因，**不含**芬太尼，"ohne" 是德语「不含」）。 |
| H9 | ID 300 在 observation 是呼吸频率、在 pharma 是甲硝唑，共用查表会串 | **成立** | 参考表 147 个 observation ID 和 565 个 pharma ID 之间只有 300 这一个冲突。v1 把所有裸 ID 登记进同一个字典（:312-332），`_build_observation_rows` 对 `variableid` 和 `pharmaid` 用同一张表（:611-637）。只要处理 raw stage 的 pharma 表，pharmaid 300 的 `givendose` 就会变成 RR。 |
| M1 | MIMIC-IV 按 itemid 映射，只有 Temp 做了 F→C | **成立** | `mimic_iv_schema.py:17-36`。唯一的单位换算在 `mimic_iv_dataset.py:80-82`，全仓库都没有检查 `valueuom`。 |

## 2. 审计未列出的问题

### 2.1 MIMIC-IV

| 编号 | 问题 | 证据 |
|---|---|---|
| N1 | `urine_output` 的 4 个 itemid 是 226566 `Urine and GU Irrigant Out`、226627 `OR Urine`、226631 `PACU Urine`、227489 `GU Irrigant/Urine Volume Out`。最主要的 226559 `Foley`、226560 `Void` 和其余 8 个尿量条目都没有。227489 也没有扣除 227488（膀胱冲洗入量）。v1 的尿量很可能几乎全是零填充（待 R-M1 确认）。 | `mimic_iv_schema.py:35`；`d_items` 标签；mimic-code `measurement/urine_output.sql` |
| N2 | `ventilation_on` 列了 223848 `Ventilator Type` 和 223849 `Ventilator Mode`，但这两个是 `param_type=Text`，没有 `valuenum`。`normalize_chart_chunk` 只取 `valuenum` 并丢掉空值，所以这两个条目一行都进不来。剩下的 220339 `PEEP set` 和 223835 FiO2 中，FiO2 在高流量和普通吸氧时也会记录。 | `mimic_iv_schema.py:25`；`mimic_iv_io.py:118-119`；`d_items` 的 `param_type` |
| N3 | SBP、DBP、MAP 只用无创条目，见第 3 节。 | `mimic_iv_schema.py:21-23` |
| N4 | 输液条目：`rate` 为空的行（推注）被填成 1.0，也算「在用药」；只用 `starttime`，不用 `endtime`，所以「在用药」只出现在新行开始的那个 bin，不覆盖整个输注期。锚点用的是同一组 itemid。 | `mimic_iv_io.py:200-204`；`mimic_iv_dataset.py:83-84`；`shock_anchor_builder.py:10-14` |
| N5 | 镇静没有 dexmedetomidine（225150、229420），而 eICU v1 有。MIMIC-IV 3.1 新增的去氧肾上腺素条目 229630、229632、229631 和 229617 `Epinephrine.` 都不在 v1 里。 | `mimic_iv_schema.py:33-34`；`d_items` |

### 2.2 eICU

| 编号 | 问题 | 证据 |
|---|---|---|
| N6 | 通气关键词按子串匹配 `respChartTypeCat + label + value` 拼起来的文本。`"ett"` 会命中 `respFlowSettings`（se**tt**ings），`"vent"` 会命中 `respFlowPtVentData`，所以几乎所有呼吸设置行都会被判成通气，包括鼻导管吸氧。类别名来自官方文档的示例，各类行数待 R-E4 统计。 | `eicu_dataset.py:43-49, 384-391`；`eicutables/respiratoryCharting.md` |
| N7 | 升压药子串漏掉了 eicu-code 整理过的名称：`Levophed ...`、`NSS with LEVO`、`Neo-Synephrine ...`、`EPI (mcg/min)`、`Epinepherine (mcg/min)`。镇静关键词也没有商品名（Diprivan、Versed、Precedex，待核实）。 | `eicu_dataset.py:25-41, 268-272`；eicu-code `pivoted-infusion.sql` |
| N8 | 只要药名命中，不管 `drugrate` 多少都记为「在用药」，`drugrate=0` 的停药行也算。缺失的速率按 1.0 处理。锚点把不同药物、不同单位的速率直接相加。 | `eicu_dataset.py:347-359, 465-470` |
| N9 | Glucose 混进了床旁快速血糖，见 E1。MIMIC 的 Glucose 只有化验室值（50931），两库的测量方式和测量频率都不一样。 | 同 E1 |
| N10 | Temp 只来自 `vitalPeriodic.temperature`，按文档只有接了探头时才有。手工测量的体温在 `nurseCharting` 里，v1 没有用。 | `eicu_dataset.py:167`；`eicutables/vitalPeriodic.md` |

### 2.3 HiRID

| 编号 | 问题 | 证据 |
|---|---|---|
| N11 | `_resolve_hirid_stage_dir` 优先用 `merged_stage`。本机完整 HiRID 目录里 raw stage 还是 tar 包，所以对 `data/raw/hirid` 运行 v1 会落到 merged stage。公开版 merged stage 只有 18 个变量：`vm1, vm3, vm4, vm5, vm13, vm20, vm28, vm62, vm136, vm146, vm172, vm174, vm176, pm41, pm42, pm43, pm44, pm87`。没有去甲肾上腺素、肾上腺素、血管加压素，唯一像升压药的是 `pm41` dobutamine。用这个 stage 做出来的「升压药锚点」实际是 dobutamine 锚点，RR、Temp、pH、肌酐、WBC、血小板、尿量、镇静、通气全部为空。 | `hirid_dataset.py:18, 503-508`；merged parquet footer 列名 |
| N12 | 论文用的 HiRID 已处理数据早于 HEAD 的映射代码。各时间点：`data/processed/hirid` 的索引 2026-04-27 19:31，`hirid_rebuild` 的索引 2026-04-28 00:13；提交 4871154 在 04-27 19:40，2647c6d 在 04-28 14:03，03a3476（加入单位检查）在 04-28 15:28。抽查两份数据各 4 个 episode，都有 RR、pH、WBC、Platelets 观测。4871154 的逻辑映射不出 Platelets，HEAD 会把 Platelets 隔离，所以生成时用的是与 2647c6d 一致、当时尚未提交的代码。在参考表上模拟 2647c6d：除了第 1 节的全部问题，还会把纤维蛋白原、凝血酶原活动度、V/VII/X 因子（名字里有 "in Platelet poor plasma"）算成 Platelets，把中性粒细胞百分比算成 WBC，把尿钠、尿素算成 `urine_output`。生成时的工作区已无法还原，只能重建。 | `git log`；模拟输出；episode 缺失掩码 |
| N13 | 升压药 token 还会命中推注剂型（`Noradrenalin 10 µg/ml Bolus`、`Adrenalin 10/100 µg/ml Bolus`）和 1 mg/ml 复苏安瓿（`Adrenalin 1mg/ml`，ID 71）。 | `hirid_dataset.py:74-83`；实测映射 |
| N14 | `Hourly urine volume`（10020000）的单位是 ml/h，v1 按 bin 求和当作 mL，这只在每小时恰好一条记录时才成立（待 R-H2 核实）。 | 参考表；`mimic_iv_schema.py:35` 的 `agg="sum"` |
| N15 | `subject_id = stay_id`。HiRID 的 `patientid` 本来就是按住院编号的，影响不大，但论文里应如实说明。 | `hirid_dataset.py:730-731` |

## 3. 有创与无创血压（协调方追加核查）

**结论**：
- 映射层面已确认：MIMIC-IV v1 不含有创动脉血压。
- 由此导致泄漏审计里「最后一个 bin 有无创血压：正例 23%，对照 62%」这一差异，**尚未确认**。机制成立，但需要 chartevents 统计（R-M4）。

### 3.1 v1 各源用的是哪些条目

| 源 | v1 的 SBP/DBP/MAP | 含有创？ | 证据 |
|---|---|---|---|
| MIMIC-IV | 220179、220180、220181。`d_items` 标签依次是 `Non Invasive Blood Pressure systolic/diastolic/mean` | **否**。220050、220051、220052（`Arterial Blood Pressure systolic/diastolic/mean`）和 225309、225310、225312（`ART BP Systolic/Diastolic/Mean`）都不在 v1 里 | `mimic_iv_schema.py:21-23`；`d_items` |
| eICU | `vitalPeriodic.systemicsystolic/systemicdiastolic/systemicmean`（动脉线，监护仪 5 分钟中位数）＋ `vitalAperiodic.noninvasivesystolic/noninvasivediastolic/noninvasivemean` | 是。两种混在同一特征里，按 bin 取均值 | `eicu_dataset.py:164-166, 192-194`；`eicutables/vitalPeriodic.md`、`vitalAperiodic.md` |
| HiRID（HEAD） | 有创 100/120/110（别名 :35、:39、:42）＋ 无创 600/620（描述推断 :156-161）＋ 610（别名 :43） | 是，混在一起。04-27 的 4871154 版本只映射了有创和无创 MAP（600、620 为 None）。merged stage 的 vm3/vm4/vm5 只有有创 | `hirid_dataset.py`；实测 |

其余几个生命体征在 MIMIC-IV v1 里分别是：HR 220045（`Heart Rate`）、RR 220210（`Respiratory Rate`）、SpO2 220277（`O2 saturation pulseoxymetry`）、Temp 223762/223761（`Temperature Celsius/Fahrenheit`）。见 `mimic_iv_schema.py:18-20, 24`。

### 3.2 机制

动脉置管后，护士每小时记录有创血压，通常也就不再定时测袖带血压。v1 的 MIMIC 只认袖带血压，于是有动脉线的病人在血压通道上大量缺失。这类病人病情更重、更可能开始用升压药，所以「血压在最后一个 bin 有没有测」实际上成了「有没有动脉线」的代理变量。eICU 和 HiRID 的 v1 把两种血压混在一起，没有这个偏差，因此三个库的血压缺失含义并不一致，跨库迁移时也会有影响。

### 3.3 v2 的做法

- 映射表里每个 SBP/DBP/MAP 行都有 `modality` 字段（`invasive` 或 `noninvasive`）。校验函数要求血压的纳入行必须写明 modality，非血压行不得填写。
- 合并规则 `BP_MERGE_RULE`：对每个（stay, bin, 特征），只要该 bin 有有创值，就取有创值的均值；没有时才取无创值的均值。由 `merge_blood_pressure(frame, source, keys=("stay_id", "bin"))` 实现。
- 用了哪种来源记在 `bp_source` 里，属于 `AUXILIARY_INDICATORS`，**默认不作为模型输入**，只用于泄漏审计和分层分析。
- 合并之后，有动脉线的病人几乎每个 bin 都有血压，缺失率仍会随置管情况变化。建议泄漏审计在合并后重算「最后一个 bin 是否测了血压」，模型默认不使用血压通道的缺失掩码，或者同时报告使用与不使用的结果。

## 4. v2 标准定义

合理范围只用来筛掉明显的录入错误，不是临床阈值。来源是 mimic-code（`vitalsign.sql`、`chemistry.sql`）和 eicu-code（`pivoted-vital.sql`、`pivoted-lab.sql`、`pivoted-bg.sql`），个别值取了两者中较宽的一个。

| 特征 | 类型 | 标准单位 | 可接受的源单位 → 换算 | 合理范围 | 定义 |
|---|---|---|---|---|---|
| HR | measurement | bpm | bpm、/min（恒等） | 1–300 | 监护仪或护士记录的心率 |
| RR | measurement | insp/min | insp/min、/min（恒等） | 1–70 | 实测总呼吸频率，排除呼吸机设定频率 |
| SpO2 | measurement | % | % | 1–100 | 脉搏血氧，排除血气 SaO2、SvO2、ScvO2 |
| SBP / DBP / MAP | measurement | mmHg | mmHg | 1–400 / 1–300 / 1–300 | 体循环动脉压，按 `BP_MERGE_RULE` 优先有创；排除肺动脉压和 CVP |
| Temp | measurement | degC | degC；degF → (x−32)×5/9 | 25–46 | 体温（核心、直肠、口腔、腋下合并），排除设备温度 |
| ventilation_on | state | flag | 任意 → 由取值规则得到 0/1 | 0–1 | 有记录的正压通气（有创或无创）时为 1，高流量和普通吸氧为 0 |
| Lactate | measurement | mmol/L | mmol/L；mg/dL → ÷9.008 | 0.1–30 | 血乳酸（动脉或静脉） |
| pH | measurement | pH | pH、空、units（恒等） | 6.5–8.5 | 血气 pH |
| Creatinine | measurement | mg/dL | mg/dL；µmol/L → ÷88.4 | 0.1–30 | 化验室血清/血浆肌酐 |
| Glucose | measurement | mg/dL | mg/dL；mmol/L → ×18.016 | 10–1500 | 化验室血清/血浆葡萄糖，排除床旁血糖（待团队决定） |
| Bilirubin_Total | measurement | mg/dL | mg/dL；µmol/L → ÷17.104 | 0.1–70 | 血清/血浆总胆红素 |
| WBC | measurement | 10^9/L | 10^9/L、K/µL、K/mcL、G/l（恒等） | 0.1–500 | 血白细胞计数 |
| Platelets | measurement | 10^9/L | 同 WBC | 1–2000 | 血小板计数 |
| vasopressor_on | state | flag | 任意速率 → 速率 > 0 为 1 | 0–1 | 去甲肾上腺素、肾上腺素、血管加压素、去氧肾上腺素、多巴胺**持续输注**期间为 1；排除正性肌力药和推注 |
| sedation_on | state | flag | 同上 | 0–1 | 丙泊酚、咪达唑仑、右美托咪定、芬太尼**持续输注**期间为 1；排除推注、PCA/硬膜外、贴剂、口服和其他药物 |
| urine_output | volume | mL | mL；mL/h → mL（假定每小时一条记录） | 0–5000（单条） | 每条记录的尿量，按 bin 求和 |

单位识别（`normalize_unit`）不区分大小写，空格和 µ/μ/mc 也视为相同。唯一的例外是 `G/l`：大写 G 表示每升 10^9 个，小写 `g/l` 表示每升克数。v1 统一转小写，结果 HiRID 的血小板被隔离；而纤维蛋白原（`g/L`）如果被推断成 WBC，又会通过单位检查。

各源的取值规则（`VALUE_RULES`）概要：
- **状态量**：
  - 先把记录转成区间，再把与区间重叠的 bin 记为 1。MIMIC 输液用 `[starttime, endtime)` 且 `rate > 0`。eICU 输液用 `drugrate`（缺失时用 `infusionrate`）> 0 开始、= 0 停止，最长沿用多久在 R-E2 之后定。HiRID 按 `infusionid` 分段。
  - 通气按「同一状态的相邻记录之间连续，间隔 ≥ 14 小时则断开」处理，与 mimic-code `ventilation.sql` 相同。
  - MIMIC 的文本条目从 `value` 列读，不是 `valuenum`。
- **尿量**：MIMIC 中 227488 取负号。eICU 只用 `is_eicu_urine_cellpath` 判为尿量的行的 `cellValueNumeric`，永远不用 `outputTotal`。HiRID 用 10020000 的值。
- **血压**：见第 3 节。

## 5. 各源映射表

下表由 `src/icu/v2/harmonization.py` 的表格数据直接生成，与代码一致。只想看结构化数据时，可以用 `mapping_frame(source)` 或 `feature_summary(source)` 导出。标记说明：*（v2 新增）* 指 v1 没有；*（v2 修正）* 指 v1 想映射但处理错了；*（v1 曾映射）* 指 v1 会把它映射进特征，v2 排除。置信度为纳入行中的最高/最低档。

#### MIMIC-IV 3.1

| 特征 | 纳入的标识（命名空间:标识 = 名称 [源单位]） | 换算到标准单位 | 显式排除 | 置信度 |
|---|---|---|---|---|
| HR | chartevents:`220045` = Heart Rate [bpm] | 无（恒等） | — | high |
| RR | chartevents:`220210` = Respiratory Rate [insp/min]<br>chartevents:`224690` = Respiratory Rate (Total) [insp/min] *（v2 新增）* | 无（恒等） | chartevents:`224688` Respiratory Rate (Set)<br>chartevents:`224689` Respiratory Rate (spontaneous) | high/medium |
| SpO2 | chartevents:`220277` = O2 saturation pulseoxymetry [%] | 无（恒等） | chartevents:`220227` Arterial O2 Saturation<br>chartevents:`223772` SvO2<br>chartevents:`223769` O2 Saturation Pulseoxymetry Alarm - High<br>chartevents:`223770` O2 Saturation Pulseoxymetry Alarm - Low<br>chartevents:`226253` SpO2 Desat Limit<br>labevents:`50817` Oxygen Saturation (Blood Gas) | high |
| SBP | chartevents:`220179` = Non Invasive Blood Pressure systolic [mmHg] (noninvasive)<br>chartevents:`220050` = Arterial Blood Pressure systolic [mmHg] (invasive) *（v2 新增）*<br>chartevents:`225309` = ART BP Systolic [mmHg] (invasive) *（v2 新增）* | 无（恒等） | chartevents:`220056` Arterial Blood Pressure Alarm - Low<br>chartevents:`220058` Arterial Blood Pressure Alarm - High | high/medium |
| DBP | chartevents:`220180` = Non Invasive Blood Pressure diastolic [mmHg] (noninvasive)<br>chartevents:`220051` = Arterial Blood Pressure diastolic [mmHg] (invasive) *（v2 新增）*<br>chartevents:`225310` = ART BP Diastolic [mmHg] (invasive) *（v2 新增）* | 无（恒等） | — | high/medium |
| MAP | chartevents:`220181` = Non Invasive Blood Pressure mean [mmHg] (noninvasive)<br>chartevents:`220052` = Arterial Blood Pressure mean [mmHg] (invasive) *（v2 新增）*<br>chartevents:`225312` = ART BP Mean [mmHg] (invasive) *（v2 新增）* | 无（恒等） | — | high/medium |
| Temp | chartevents:`223762` = Temperature Celsius [°C]<br>chartevents:`223761` = Temperature Fahrenheit [°F] | degF->degC | chartevents:`224642` Temperature Site<br>chartevents:`226329` Blood Temperature CCO (C)<br>chartevents:`229236` Cerebral Temperature (C)<br>chartevents:`228242` Pt. Temperature (BG) (SOFT) | high |
| ventilation_on | chartevents:`223849` = Ventilator Mode *（v2 修正）*<br>chartevents:`229314` = Ventilator Mode (Hamilton) *（v2 新增）*<br>chartevents:`226732` = O2 Delivery Device(s) *（v2 新增）*<br>chartevents:`220339` = PEEP set [cmH2O] | 状态量，见取值规则 | chartevents:`223835` Inspired O2 Fraction *（v1 曾映射）*<br>chartevents:`223848` Ventilator Type *（v1 曾映射）* | medium |
| Lactate | labevents:`50813` = Lactate (Blood, Blood Gas) [mmol/L] | 无（恒等） | labevents:`52442` Lactate (Blood, Blood Gas)<br>labevents:`53154` Lactate (Blood, Chemistry)<br>labevents:`50954` Lactate Dehydrogenase (LD) | high |
| pH | labevents:`50820` = pH (Blood, Blood Gas) [units] | 无（恒等） | labevents:`50831` pH (Other Body Fluid)<br>labevents:`52041` pH (Fluid)<br>labevents:`51491` pH (Urine, Hematology)<br>labevents:`51094` pH (Urine, Chemistry)<br>labevents:`52730` pH (Urine, Chemistry)<br>labevents:`52045` pH, Urine (Blood Gas) | high |
| Creatinine | labevents:`50912` = Creatinine (Blood, Chemistry) [mg/dL] | 无（恒等） | labevents:`52546` Creatinine (Blood, Chemistry)<br>labevents:`52024` Creatinine, Whole Blood (Blood Gas)<br>labevents:`51082` Creatinine, Urine<br>labevents:`51081` Creatinine, Serum (fluid=Urine)<br>labevents:`51977` Creatinine, Blood (fluid=Urine)<br>labevents:`51080` Creatinine Clearance<br>labevents:`50841` Creatinine, Ascites<br>labevents:`51032` Creatinine, Body Fluid<br>labevents:`51787` Creatinine, CSF<br>labevents:`51021` Creatinine, Joint Fluid<br>labevents:`51052` Creatinine, Pleural<br>labevents:`51937` Creatinine, Stool<br>chartevents:`220615` Creatinine (serum)<br>chartevents:`229761` Creatinine (whole blood) | high |
| Glucose | labevents:`50931` = Glucose (Blood, Chemistry) [mg/dL] | 无（恒等） | labevents:`50809` Glucose (Blood, Blood Gas)<br>labevents:`52569` Glucose (Blood, Chemistry)<br>labevents:`52027` Glucose, Whole Blood (Blood Gas)<br>labevents:`51478` Glucose (Urine, Hematology)<br>labevents:`51981` Glucose (Urine, Chemistry)<br>labevents:`51084` Glucose, Urine<br>labevents:`51790` Glucose, CSF<br>labevents:`50842` Glucose, Ascites<br>labevents:`51034` Glucose, Body Fluid<br>labevents:`51022` Glucose, Joint Fluid<br>labevents:`51053` Glucose, Pleural<br>labevents:`51941` Glucose, Stool<br>chartevents:`225664` Glucose finger stick (range 70-100)<br>chartevents:`220621` Glucose (serum)<br>chartevents:`226537` Glucose (whole blood) | high |
| Bilirubin_Total | labevents:`50885` = Bilirubin, Total (Blood) [mg/dL] | 无（恒等） | labevents:`53089` Bilirubin, Total (Blood)<br>labevents:`50883` Bilirubin, Direct<br>labevents:`50838` Bilirubin, Total, Ascites<br>labevents:`51028` Bilirubin, Total, Body Fluid<br>labevents:`51783` Bilirubin, Total, CSF<br>labevents:`51812` Bilirubin, Total, Joint Fluid<br>labevents:`51049` Bilirubin, Total, Pleural<br>labevents:`51932` Bilirubin, Total, Stool<br>chartevents:`225690` Total Bilirubin | high |
| WBC | labevents:`51301` = White Blood Cells (Blood, Hematology) [K/uL]<br>labevents:`51300` = WBC Count (Blood, Hematology) [K/uL] | 无（恒等） | labevents:`51755` White Blood Cells (Blood, Chemistry)<br>labevents:`51756` White Blood Cells (Blood, Chemistry)<br>labevents:`51516` WBC (Urine)<br>labevents:`51517` WBC Casts (Urine)<br>labevents:`51518` WBC Clumps (Urine)<br>labevents:`52407` WBC (Stool)<br>chartevents:`220546` WBC | high |
| Platelets | labevents:`51265` = Platelet Count (Blood, Hematology) [K/uL] | 无（恒等） | labevents:`53189` Platelet Count (Blood, Chemistry)<br>chartevents:`227457` Platelet Count<br>chartevents:`225678` ZPlatelet Count<br>inputevents:`225170` Platelets<br>inputevents:`226369` OR Platelet Intake<br>inputevents:`227071` PACU Platelet Intake | high |
| vasopressor_on | inputevents:`221906` = Norepinephrine [mcg/kg/min]<br>inputevents:`221289` = Epinephrine [mcg/kg/min]<br>inputevents:`221662` = Dopamine [mcg/kg/min]<br>inputevents:`221749` = Phenylephrine [mcg/kg/min]<br>inputevents:`222315` = Vasopressin [units/hour]<br>inputevents:`229630` = Phenylephrine (50/250) *（v2 新增）*<br>inputevents:`229632` = Phenylephrine (200/250) *（v2 新增）*<br>inputevents:`229631` = Phenylephrine (200/250)_OLD_1 *（v2 新增）* | 状态量，见取值规则 | inputevents:`229617` Epinephrine.<br>inputevents:`221653` Dobutamine<br>inputevents:`221986` Milrinone<br>chartevents:`229789` Phenylephrine (Intubation) | high/medium/low |
| sedation_on | inputevents:`222168` = Propofol<br>inputevents:`221668` = Midazolam (Versed)<br>inputevents:`221744` = Fentanyl<br>inputevents:`225942` = Fentanyl (Concentrate) *（v2 新增）*<br>inputevents:`225150` = Dexmedetomidine (Precedex) *（v2 新增）*<br>inputevents:`229420` = Dexmedetomidine (Precedex) *（v2 新增）* | 状态量，见取值规则 | inputevents:`225972` Fentanyl (Push)<br>chartevents:`227210` Propofol (Intubation)<br>inputevents:`221385` Lorazepam (Ativan)<br>inputevents:`221833` Hydromorphone (Dilaudid)<br>inputevents:`225154` Morphine Sulfate | high/medium |
| urine_output | outputevents:`226559` = Foley [mL] *（v2 新增）*<br>outputevents:`226560` = Void [mL] *（v2 新增）*<br>outputevents:`226561` = Condom Cath [mL] *（v2 新增）*<br>outputevents:`226584` = Ileoconduit [mL] *（v2 新增）*<br>outputevents:`226563` = Suprapubic [mL] *（v2 新增）*<br>outputevents:`226564` = R Nephrostomy [mL] *（v2 新增）*<br>outputevents:`226565` = L Nephrostomy [mL] *（v2 新增）*<br>outputevents:`226567` = Straight Cath [mL] *（v2 新增）*<br>outputevents:`226557` = R Ureteral Stent [mL] *（v2 新增）*<br>outputevents:`226558` = L Ureteral Stent [mL] *（v2 新增）*<br>outputevents:`227489` = GU Irrigant/Urine Volume Out [mL]<br>outputevents:`227488` = GU Irrigant Volume In [mL] ×-1 *（v2 新增）* | 无（恒等） | outputevents:`226566` Urine and GU Irrigant Out *（v1 曾映射）*<br>outputevents:`226627` OR Urine *（v1 曾映射）*<br>outputevents:`226631` PACU Urine *（v1 曾映射）*<br>outputevents:`226713` Incontinent/voids (estimate) | high |

MIMIC 通气的取值常量：
- `MIMIC_VENT_MODES_ON`：223849 中算作「在通气」的 36 个模式，取自 mimic-code 的 InvasiveVent 列表。
- `MIMIC_HAMILTON_MODES_ON`：229314 的有创模式加 `DuoPaP`、`NIV`、`NIV-ST`。
- `MIMIC_O2_DEVICES_ON`：226732 中的 `Endotracheal tube`、`Bipap mask `、`CPAP mask `，原数据里后两个带尾随空格。

#### eICU-CRD 2.0

| 特征 | 纳入的标识（命名空间:标识 = 名称 [源单位]） | 换算到标准单位 | 显式排除 | 置信度 |
|---|---|---|---|---|
| HR | vitalPeriodic:`heartrate` = heartRate [bpm] | 无（恒等） | — | high |
| RR | vitalPeriodic:`respiration` = respiration [/min] | 无（恒等） | — | high |
| SpO2 | vitalPeriodic:`sao2` = saO2 [%] | 无（恒等） | lab:`O2 Sat (%)` O2 Sat (%) | medium |
| SBP | vitalPeriodic:`systemicsystolic` = systemicSystolic (arterial line) [mmHg] (invasive)<br>vitalAperiodic:`noninvasivesystolic` = nonInvasiveSystolic [mmHg] (noninvasive) | 无（恒等） | vitalPeriodic:`pasystolic` paSystolic | high |
| DBP | vitalPeriodic:`systemicdiastolic` = systemicDiastolic (arterial line) [mmHg] (invasive)<br>vitalAperiodic:`noninvasivediastolic` = nonInvasiveDiastolic [mmHg] (noninvasive) | 无（恒等） | vitalPeriodic:`padiastolic` paDiastolic | high |
| MAP | vitalPeriodic:`systemicmean` = systemicMean (arterial line) [mmHg] (invasive)<br>vitalAperiodic:`noninvasivemean` = nonInvasiveMean [mmHg] (noninvasive) | 无（恒等） | vitalPeriodic:`pamean` paMean<br>vitalPeriodic:`cvp` cvp | high |
| Temp | vitalPeriodic:`temperature` = temperature [°C] | 无（恒等） | nurseCharting:`Temperature (C)` nursingchartcelltypevalname 'Temperature (C)' | medium |
| ventilation_on | respiratoryCare:`ventstartoffset` = ventStartOffset/ventEndOffset interval *（v2 新增）*<br>respiratoryCharting:`RT Vent On/Off` = respChartValueLabel 'RT Vent On/Off' (on: EICU_VENT_ON_VALUES) *（v2 新增）* | 状态量，见取值规则 | respiratoryCharting:`/vent\|peep\|fio2\|ett\|endotracheal/` v1 keyword rule over respChartTypeCat + label + value *（v1 曾映射）*<br>respiratoryCharting:`FiO2` FiO2 *（v1 曾映射）* | low |
| Lactate | lab:`lactate` = lactate [mmol/L] | 无（恒等） | lab:`/lactate/` v1 rule: 'lactate' in name *（v1 曾映射）* | high |
| pH | lab:`pH` = pH | 无（恒等） | lab:`/^ph$\|^ph /` v1 rule: name == 'ph' or startswith 'ph ' *（v1 曾映射）* | high |
| Creatinine | lab:`creatinine` = creatinine [mg/dL] | 无（恒等） | lab:`urinary creatinine` urinary creatinine *（v1 曾映射）*<br>lab:`/creatinine/` v1 rule: 'creatinine' in name *（v1 曾映射）* | high |
| Glucose | lab:`glucose` = glucose (laboratory) [mg/dL] | 无（恒等） | lab:`bedside glucose` bedside glucose *（v1 曾映射）*<br>lab:`glucose - CSF` glucose - CSF *（v1 曾映射）*<br>lab:`/glucose/` v1 rule: 'glucose' in name *（v1 曾映射）* | high |
| Bilirubin_Total | lab:`total bilirubin` = total bilirubin [mg/dL] | 无（恒等） | lab:`direct bilirubin` direct bilirubin<br>lab:`/bilirubin.*total\|total.*bilirubin/` v1 rule: 'bilirubin' and 'total' in name *（v1 曾映射）* | high |
| WBC | lab:`WBC x 1000` = WBC x 1000 [K/mcL] | 无（恒等） | lab:`WBC's in urine` WBC's in urine *（v1 曾映射）*<br>lab:`WBC's in body fluid` WBC's in body fluid *（v1 曾映射）*<br>lab:`WBC's in cerebrospinal fluid` WBC's in cerebrospinal fluid *（v1 曾映射）*<br>lab:`WBC's in pleural fluid` WBC's in pleural fluid *（v1 曾映射）*<br>lab:`WBC's in peritoneal fluid` WBC's in peritoneal fluid *（v1 曾映射）*<br>lab:`WBC's in synovial fluid` WBC's in synovial fluid *（v1 曾映射）*<br>lab:`/wbc\|white blood/` v1 rule: 'wbc' or 'white blood' in name *（v1 曾映射）* | high |
| Platelets | lab:`platelets x 1000` = platelets x 1000 [K/mcL] | 无（恒等） | lab:`/platelet/` v1 rule: 'platelet' in name *（v1 曾映射）* | high |
| vasopressor_on | infusionDrug:`/\bnor-?epinephrine\b\|\bnorepi\b\|\blevophed\b\|\blevo\b/` = norepinephrine incl. Levophed and 'NSS with LEVO' *（v2 修正）*<br>infusionDrug:`/\bepinephrine\b\|\bepinepherine\b\|\bepi\b\|\badrenaline?\b/` = epinephrine incl. 'EPI (mcg/min)' and the misspelling 'Epinepherine' *（v2 修正）*<br>infusionDrug:`/\bphenylephrine\b\|\bneo-? ?synephrine\b\|\bneosynsprine\b/` = phenylephrine incl. Neo-Synephrine *（v2 修正）*<br>infusionDrug:`/\bvasopressin\b\|\bvasostrict\b\|\bvaso\b/` = vasopressin incl. 'levo/vaso' mixtures *（v2 修正）*<br>infusionDrug:`/\bdopamine\b\|\bino?tropin\b/` = dopamine | 状态量，见取值规则 | infusionDrug:`/dobu/` dobutamine incl. Dobutrex *（v1 曾映射）*<br>infusionDrug:`/milrinone\|milronone\|primacor/` milrinone<br>infusionDrug:`/isoproterenol\|isuprel/` isoproterenol<br>infusionDrug:`/terlipressin\|angiotensin\|giapreza\|methylene blue/` terlipressin, angiotensin II, methylene blue<br>infusionDrug:`/lidocaine\|racemic/` lidocaine-epinephrine or nebulised racemic epinephrine | medium |
| sedation_on | infusionDrug:`/\bpropofol\b\|\bdiprivan\b/` = propofol incl. Diprivan *（v2 修正）*<br>infusionDrug:`/\bmidazolam\b\|\bversed\b/` = midazolam incl. Versed *（v2 修正）*<br>infusionDrug:`/\bdexmedetomidine\b\|\bprecedex\b/` = dexmedetomidine incl. Precedex *（v2 修正）*<br>infusionDrug:`/\bfentanyl\b\|\bsublimaze\b/` = fentanyl incl. Sublimaze | 状态量，见取值规则 | infusionDrug:`/bupivacaine\|ropivacaine\|epidural\|\bpca\b\|patch\|transdermal/` epidural, PCA, local-anaesthetic and transdermal preparations<br>infusionDrug:`/lorazepam\|ativan\|ketamine\|ketalar\|morphine\|hydromorphone\|dilaudid\|sufentanil\|remifentanil\|ultiva\|clonidine\|pentobarbital/` other sedatives and opioids | medium |
| urine_output | intakeOutput:`eicu_urine_cellpath` = cellPath under Output (ml) with a urine label (eicu-code pivoted-uo.sql + v2 exclusions) [mL] *（v2 修正）* | 无（恒等） | intakeOutput:`outputtotal` outputTotal *（v1 曾映射）* | medium |

eICU 的匹配规则：
- **化验**：只接受确切名单。判定顺序是：确切纳入 > 确切排除 > 模式排除 > 模式纳入。v1 的子串规则保留为排除模式，凡是被它命中、但不在名单里的名称一律排除；如果名称里有非血液标本标记（urine、CSF、fluid 等），排除理由写成「非血液标本」。
- **尿量**：`is_eicu_urine_cellpath` 要求路径以 `flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|` 开头，标签在 eicu-code 的 93 个尿量标签里（大小写和空格合并后），或者符合 eicu-code 的前缀规则。v2 在此基础上另外排除计数类（count）、未计量类（unmeasured）、班次合计（7AM-7PM、NOC end shift）、冲洗和 PACU 的标签。
- **药物**：模式排除优先于模式纳入。所以硬膜外的芬太尼/布比卡因混合液、含肾上腺素的局麻药都会被排除。

#### HiRID 1.1.1

| 特征 | 纳入的标识（命名空间:标识 = 名称 [源单位]） | 换算到标准单位 | 显式排除 | 置信度 |
|---|---|---|---|---|
| HR | observation:`200` = Heart rate [/min] | 无（恒等） | — | high |
| RR | observation:`300` = Respiratory rate [/min]<br>observation:`310` = Respiratory rate [/min]<br>observation:`5685` = Respiratory rate [/min] | 无（恒等） | observation:`320` Ventilator rate<br>pharma:`300` Metronidazole tabl 200 mg *（v1 曾映射）* | high/medium |
| SpO2 | observation:`4000` = Peripheral oxygen saturation [%]<br>observation:`8280` = Peripheral oxygen saturation [%] | 无（恒等） | observation:`4200` Mixed venous oxygen saturation *（v1 曾映射）*<br>observation:`20000800` Oxygen saturation in Arterial blood *（v1 曾映射）*<br>observation:`20001000` Central venous oxygenation saturation<br>observation:`24000737` Central venous oxygenation saturation | high |
| SBP | observation:`100` = Invasive systolic arterial pressure [mmHg] (invasive)<br>observation:`600` = Non-invasive systolic arterial pressure [mmHg] (noninvasive) | 无（恒等） | observation:`150` Pulmonary artery systolic pressure | high |
| DBP | observation:`120` = Invasive diastolic arterial pressure [mmHg] (invasive)<br>observation:`620` = Non-invasive diastolic arterial pressure [mmHg] (noninvasive) | 无（恒等） | observation:`170` Pulmonary artery diastolic pressure | high |
| MAP | observation:`110` = Invasive mean arterial pressure [mmHg] (invasive)<br>observation:`610` = Non-invasive mean arterial pressure [mmHg] (noninvasive) | 无（恒等） | observation:`160` Pulmonary artery mean pressure<br>observation:`700` Central venous pressure<br>observation:`960` Central venous pressure<br>observation:`15001441` Central venous pressure | high |
| Temp | observation:`410` = Core body temperature [°C]<br>observation:`7100` = Rectal temperature [°C]<br>observation:`400` = Axillary temperature [°C] | 无（恒等） | — | high/medium |
| ventilation_on | observation:`3845` = Ventilator mode *（v2 新增）*<br>observation:`15001552` = Ventilator Airway Code *（v2 新增）*<br>observation:`2600` = Positive end expiratory pressure setting [cmH2O] *（v2 新增）* | 状态量，见取值规则 | observation:`2010` Inspired oxygen concentration<br>observation:`15001555` Oxygen administration by mask<br>observation:`15001525` Oxygen administration by nasal cannula | medium |
| Lactate | observation:`24000524` = Lactate [Mass/volume] in Arterial blood [mmol/l]<br>observation:`24000732` = Lactate [Moles/volume] in Venous blood [mmol/l]<br>observation:`24000485` = Lactate [Moles/volume] in Venous blood [mmol/l] | 无（恒等） | observation:`24000564` Lactate [Moles/volume] in Cerebral spinal fluid *（v1 曾映射）* | high/medium |
| pH | observation:`20000300` = pH of Arterial blood | 无（恒等） | observation:`24000608` pH of Cerebral spinal fluid *（v1 曾映射）*<br>observation:`24000741` pH of Body fluid *（v1 曾映射）* | high |
| Creatinine | observation:`20000600` = Creatinine [Moles/volume] in Blood [umol/l] | umol/L->mg/dL (creatinine, /88.4) | observation:`24000572` Creatinine [Moles/volume] in Urine *（v1 曾映射）*<br>observation:`24000573` Creatinine [Moles/volume] in Urine *（v1 曾映射）* | high |
| Glucose | observation:`20005110` = Glucose [Moles/volume] in Serum or Plasma [mmol/l]<br>observation:`24000523` = Glucose [Moles/volume] in Serum or Plasma [mmol/l]<br>observation:`24000585` = Glucose [Moles/volume] in Serum or Plasma [mmol/l] | mmol/L->mg/dL (glucose, x18.016) | observation:`24000400` Glucose [Moles/volume] in Cerebral spinal fluid *（v1 曾映射）*<br>pharma:`1000022` Glucose 10%<br>pharma:`1000690` Glucose 10%<br>pharma:`1000689` Glucose 20%<br>pharma:`1000544` Glucose 20% /100ml<br>pharma:`1000746` Glucose 20%/100ml Pflege<br>pharma:`1000835` Glucose 20% /500ml<br>pharma:`1000060` Glucose 30%<br>pharma:`1000545` Glucose 40%<br>pharma:`1000567` Glucose 50% | high/medium |
| Bilirubin_Total | observation:`20004300` = Bilirubin.total [Moles/volume] in Serum or Plasma [umol/l] | umol/L->mg/dL (bilirubin, /17.104) | observation:`24000560` Bilirubin.direct [Mass/volume] in Serum or Plasma | high |
| WBC | observation:`20000700` = Leukocytes [#/volume] in Blood [G/l] | 无（恒等） | observation:`24000550` Neutrophils/100 leukocytes in Blood<br>observation:`24000556` Segmented neutrophils/100 leukocytes in Blood<br>observation:`24000557` Band form neutrophils/100 leukocytes in Blood<br>observation:`24000480` Lymphocytes [#/volume] in Blood | high |
| Platelets | observation:`20000110` = Platelets [#/volume] in Blood [G/l] *（v2 修正）* | 无（恒等） | observation:`24000536` Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay<br>observation:`24000666` Prothrombin activity actual/normal in Platelet poor plasma by Coagulation assay<br>observation:`24000637` Coagulation factor V activity actual/normal in Platelet poor plasma by Coagulation assay<br>observation:`24000552` Coagulation factor VII activity actual/normal in Platelet poor plasma by Coagulation assay<br>observation:`24000667` Coagulation factor X activity actual/normal in Platelet poor plasma by Coagulation assay<br>pharma:`1000245` Platelet transfusion<br>pharma:`1000201` Platelet transfusion | high |
| vasopressor_on | pharma:`1000656` = Noradrenalin 100 µg/ml Perfusor<br>pharma:`1000657` = Noradrenalin 20 µg/ml Perfusor<br>pharma:`1000462` = Noradrenalin 1mg/ml<br>pharma:`1000649` = Adrenalin 20 µg/ml Perfusor<br>pharma:`1000650` = Adrenalin 100 µg/ml Perfusor<br>pharma:`113` = Vasopressin inf 0.4 U/ml<br>pharma:`112` = Vasopressin inj 20 U/ml | 状态量，见取值规则 | pharma:`1000658` Noradrenalin 10 µg/ml Bolus *（v1 曾映射）*<br>pharma:`1000750` Adrenalin 100 µg/ml Bolus *（v1 曾映射）*<br>pharma:`1000655` Adrenalin 10 µg/ml Bolus *（v1 曾映射）*<br>pharma:`71` Adrenalin  1mg/ml *（v1 曾映射）*<br>pharma:`426` Dobutrex 250 mg/20ml *（v1 曾映射）*<br>pharma:`1000441` Corotrop Inj Lsg 1mg/ml 10 ml<br>pharma:`1000606` Simdax inf<br>pharma:`1000793` Glypressin Inj Lsg<br>pharma:`1001061` Haemopressin TS Inj Lsg | high/medium |
| sedation_on | pharma:`1000491` = Disoprivan 1% *（v2 新增）*<br>pharma:`1001053` = Disoprivan 1% *（v2 新增）*<br>pharma:`1001050` = Disoprivan 2% *（v2 新增）*<br>pharma:`208` = Disoprivan 2% *（v2 新增）*<br>pharma:`252` = Dormicum Perfusor 5 mg/ml *（v2 新增）*<br>pharma:`1001051` = Dormicum Perfusor 5mg/ml *（v2 新增）*<br>pharma:`1001049` = Dexdor Inf. Lsg *（v2 新增）*<br>pharma:`1000847` = Dexdor Inf. Lsg *（v2 新增）*<br>pharma:`1000741` = Fentanyl Perf<br>pharma:`1000251` = Fentanyl inj 50 µg/ml | 状态量，见取值规则 | pharma:`1000692` Fentanyl Bolus inj 50 µg/ml *（v1 曾映射）*<br>pharma:`1000621` Fentanyl-PCA-Lösung *（v1 曾映射）*<br>pharma:`1000843` PDA-Bupivacain 0.125% ohne Fentanyl *（v1 曾映射）*<br>pharma:`1000350` Durogesic TTS 25 ug<br>pharma:`1000351` Durogesic TTS 50 ug<br>pharma:`1000627` Durogesic TTS Matrix 100 ug<br>pharma:`1000768` Durogesic Matrix 12 ug/h<br>pharma:`1001052` Disoprivan 2% BOLUS<br>pharma:`1000699` Disoprivan 2% BOLUS<br>pharma:`1000691` Disoprivan  BOLUS 2% 20mg/ml<br>pharma:`1000700` Dormicum 5mg/ml BOLUS<br>pharma:`1001054` Dormicum Bolus 1mg/ml<br>pharma:`251` Dormicum inj 1 mg/ml<br>pharma:`1000902` Dormicum Tbl 7.5 mg<br>pharma:`1000771` Sufentanyl Perfusor *（v1 曾映射）*<br>pharma:`1000705` Ultiva Inj Lsg<br>pharma:`1000862` Morphine Inj à 10mg *（v1 曾映射）*<br>pharma:`1000659` Morphin HCL Perfusor<br>pharma:`1000239` Temesta inj Amp 4mg/ml<br>pharma:`1000400` Ketalar Inj Lsg 500 mg/10 ml<br>pharma:`1000857` Ketalar Perfusor<br>pharma:`1001194` Ketalar 5mg/ml Perfusor Schmerzdienst<br>pharma:`1000728` Catapresan 25µg /ml Perfusor<br>pharma:`202` Pentothal 1g Inf Lsg<br>pharma:`1000607` Pentothal inj 0.5g<br>pharma:`1001215` Rivotril Perfusor | high/medium |
| urine_output | observation:`10020000` = Hourly urine volume [ml/h] | mL/h->mL (one record per hour) | observation:`20003200` Sodium [Moles/volume] in Urine<br>observation:`24000754` Urea [Moles/volume] in Urine<br>observation:`10010020` Measurement of output from drain<br>observation:`10010070` Measurement of output from drain<br>observation:`10010071` Measurement of output from drain<br>observation:`10010072` Measurement of output from drain<br>observation:`30005010` Fluid balance<br>observation:`30005110` Fluid balance | medium |

HiRID 的取值常量：
- `HIRID_VENT_MODE_CODES_ON`：3845 中除 1（Stand By）和 9（Optional，含义不明）以外的编码。
- `HIRID_AIRWAY_CODES_ON`：15001552 中的 1 intubiert、3 Maske、4 Helm、5 Mundstück。2 tracheotomiert 只有同时记录了呼吸机模式才算。
- HiRID 参考表里没有多巴胺和去氧肾上腺素，所以这两种药在 HiRID 里不会出现，这是数据本身的情况，不是映射遗漏。
- 所有 HiRID 查表都必须带命名空间：`lookup_identifier("hirid", "pharma", 300)` 返回甲硝唑的排除行，`lookup_identifier("hirid", "observation", 300)` 返回 RR。不带命名空间会抛 `ValueError`。

## 6. v2 相对 v1 的改动汇总

- **全局**：
  - 18 个特征在三个源里都有显式纳入行，`UNAVAILABLE_FEATURES` 为空，机制保留。
  - 每行都有命名空间。
  - 排除行必须写理由。
  - 单位按特征逐个检查，不再用一张宽松的提示表。
  - 状态量改成区间语义，不再「只在事件开始的 bin 记 1」。
- **MIMIC-IV**：
  - 尿量换成 mimic-code 的 12 个条目，含 227488 取负；去掉 226566、226627、226631。
  - 通气改成读文本值，只认呼吸机模式、气管插管或无创面罩、设定 PEEP；去掉 FiO2。
  - 血压加入有创条目，按 `BP_MERGE_RULE` 合并。
  - RR 加入 224690。
  - 镇静加入右美托咪定（225150、229420）和芬太尼浓缩液（225942）。
  - 升压药加入 229630、229632、229631（待核实）。
  - 推注（`rate` 为空）不再算「在用药」。
- **eICU**：
  - 化验改成确切名单；`bedside glucose` 默认排除。
  - 尿量改成按 cellPath 选尿量行，取 `cellValueNumeric`；`outputTotal` 永不使用。
  - 升压药去掉 dobutamine，补上 Levophed、Neo-Synephrine 等名称。
  - 镇静补上商品名。
  - 通气改用 `respiratoryCare` 的起止时间和 `RT Vent On/Off`（低置信，待核实），去掉 v1 的关键词规则。
  - 病人 ID 应改用字符串形式的 `uniquepid`。这属于任务和划分模块，本模块只记录这个问题。
- **HiRID**：
  - 只从 raw stage 构建。
  - 显式的 observation 和 pharma ID 表取代别名和描述推断。
  - Platelets 恢复。
  - 去掉尿肌酐、脑脊液指标、引流液指标、SvO2、SaO2。
  - 加入通气。
  - 镇静按商品名纳入丙泊酚、咪达唑仑、右美托咪定的持续输注剂型。
  - 升压药去掉 dobutamine、推注剂型和 1 mg/ml 复苏安瓿。
- **依赖关系**：v1 的锚点和 `vasopressor_on` 用的是同一组 itemid（`shock_anchor_builder.py:10-14`）。v2 的任务模块（`task.py`）定义锚点时，必须和本表的升压药集合保持一致，或者明确说明两者为什么不同。

## 7. 需要团队决定的事

1. **床旁血糖**：三个库要么都排除（当前默认），要么都纳入。纳入的话，MIMIC 要加 225664 等 chartevents 条目，eICU 要加 `bedside glucose`，HiRID 要先查清 24000523、24000585 是不是床旁值。
2. **镇静药集合**：当前是丙泊酚、咪达唑仑、右美托咪定、芬太尼。HiRID 常用舒芬太尼和瑞芬太尼，MIMIC 有吗啡和氢吗啡酮。是否扩成「镇静或镇痛输注」？
3. **通气定义**：是否包含无创通气（当前包含）？只有气切、没有呼吸机模式的病人算不算？
4. **pH**：只用动脉血，还是动静脉合并（当前 MIMIC 合并，HiRID 只有动脉）？
5. **eICU 体温**：要不要加入 `nurseCharting` 的手工体温？需要扫一遍 1.7 GB 的表。
6. **跨库共享特征集**：建议定为「三个源的纳入行都至少是 medium」的特征，即去掉 eICU 置信度为 low 的 `ventilation_on`，共 17 个。等 R-E4、R-E5 核实后再决定是否放回。
7. **`bp_source` 和缺失掩码**：是否默认不进模型？本规范建议不进，同时报告加入后的敏感性结果。

## 8. 机器空闲后需要用原始数据核实的清单

所有命令都在仓库根目录运行，输出写到 `artifacts/v2/harmonization/`（允许的新目录）。脚本只按需读取列（`usecols`）并分块读取，结果都是聚合统计，不导出任何病人标识。每项完成后，按结果修改 `harmonization.py` 里的表（调整 confidence、删掉 pending 理由、换成真实拼写），再运行 `validate_mapping_tables()` 和测试。

公共前缀（每条命令都先执行）：

```bash
cd "<repository root>"
mkdir -p artifacts/v2/harmonization
```

### eICU

**R-E1 化验名称**（`lab.csv.gz`，517 MB）。目的：列出实际的 labname 和单位，核实排除名单的确切拼写。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import collections, sys
sys.path.insert(0, ".")
import pandas as pd
from src.icu.v2.harmonization import classify_eicu_lab
counts = collections.Counter()
for chunk in pd.read_csv("data/raw/EICU 2.0数据/lab.csv.gz", usecols=["labname", "labmeasurenamesystem"], chunksize=2_000_000):
    counts.update(zip(chunk["labname"].fillna(""), chunk["labmeasurenamesystem"].fillna("")))
rows = [(name, unit, n, *classify_eicu_lab(name)) for (name, unit), n in counts.items()]
df = pd.DataFrame(rows, columns=["labname", "unit", "n", "feature", "decision"]).sort_values("n", ascending=False)
df.to_csv("artifacts/v2/harmonization/eicu_labnames.csv", index=False)
print(df[df["decision"] != "unmapped"].to_string(index=False))
EOF
```

通过标准：
- 7 个纳入名称都存在，单位分别是 mmol/L、mg/dL、K/mcL，pH 无单位。
- 被排除的都确实是非血液标本或其他指标。
- `glucose - CSF`、`urinary creatinine`、`WBC's in ...` 改成数据里的真实拼写。
- 没有「像目标指标但被判为 unmapped」的名称。

**R-E2 输液药名**（`infusionDrug.csv.gz`，49 MB）。目的：对照 v1 和 v2 的分类，统计停药行和缺失速率，确定沿用时长。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.icu.v2.harmonization import classify_eicu_drug
from src.icu.eicu_dataset import _classify_infusion
df = pd.read_csv("data/raw/EICU 2.0数据/infusionDrug.csv.gz", usecols=["drugname", "drugrate", "infusionrate"])
rate = pd.to_numeric(df["drugrate"], errors="coerce").fillna(pd.to_numeric(df["infusionrate"], errors="coerce"))
g = df.assign(rate=rate).groupby("drugname")["rate"].agg(
    n="size", n_pos=lambda s: int((s > 0).sum()), n_zero=lambda s: int((s == 0).sum()), n_nan=lambda s: int(s.isna().sum()))
g["v2_feature"] = [classify_eicu_drug(name)[0] for name in g.index]
g["v2_reason"] = [classify_eicu_drug(name)[1] for name in g.index]
g["v1_pressor"] = [_classify_infusion(name)[0] for name in g.index]
g["v1_sedation"] = [_classify_infusion(name)[1] for name in g.index]
g.sort_values("n", ascending=False).to_csv("artifacts/v2/harmonization/eicu_drugnames.csv")
mask = (g["v2_reason"] != "unmapped") | g["v1_pressor"] | g["v1_sedation"]
print(g[mask].sort_values("n", ascending=False).head(300).to_string())
EOF
```

通过标准：
- eicu-code 列出的升压药名称全部被纳入。
- 纳入的只有 9 种标准药。
- 记下 `rate == 0` 的停药行和缺失速率的比例。
- 按同一药物相邻两行的间隔中位数，确定状态沿用的上限。

**R-E3 出入量路径**（`intakeOutput.csv.gz`，229 MB）。目的：核实尿量标签，并量化 v1 用 `outputTotal` 造成的偏差。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import collections, sys
sys.path.insert(0, ".")
import pandas as pd
from src.icu.v2.harmonization import is_eicu_urine_cellpath
IO = "data/raw/EICU 2.0数据/intakeOutput.csv.gz"
stats = collections.defaultdict(lambda: [0, 0.0])
sample = []
for chunk in pd.read_csv(IO, usecols=["patientunitstayid", "intakeoutputoffset", "outputtotal", "cellpath", "cellvaluenumeric"], chunksize=2_000_000):
    out = chunk[chunk["cellpath"].str.contains("|Output", regex=False, na=False)]
    for path, s in out.groupby("cellpath")["cellvaluenumeric"]:
        stats[path][0] += int(s.size)
        stats[path][1] += float(s.sum())
    sample.append(chunk[chunk["patientunitstayid"] % 50 == 0])
paths = pd.DataFrame([(p, n, total / max(n, 1), is_eicu_urine_cellpath(p)) for p, (n, total) in stats.items()],
                     columns=["cellpath", "n", "mean_value", "v2_urine"]).sort_values("n", ascending=False)
paths.to_csv("artifacts/v2/harmonization/eicu_output_cellpaths.csv", index=False)
print(paths.head(150).to_string(index=False))
io = pd.concat(sample)
io["hour"] = io["intakeoutputoffset"] // 60
v1 = io.groupby(["patientunitstayid", "hour"])["outputtotal"].sum()
v2 = io[io["cellpath"].map(is_eicu_urine_cellpath)].groupby(["patientunitstayid", "hour"])["cellvaluenumeric"].sum()
print(pd.concat({"v1_outputtotal_sum": v1, "v2_urine_ml": v2}, axis=1).describe().to_string())
EOF
```

通过标准：
- 行数多的尿量路径都被接受，引流、粪便、呕吐、失血路径都不被接受。
- 各尿量标签的平均单次量合理，约 50–300 mL。
- 如果 incontinent 一类标签的数值像次数而不是毫升，就把它们加入排除模式。
- 第二张表能看出 v1 的值偏大多少。

**R-E4 呼吸记录**（`respiratoryCharting.csv.gz`，198 MB）。目的：确认类别名和标签，统计 v1 关键词规则多抓了多少，确认 `RT Vent On/Off` 的取值。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import collections, sys
sys.path.insert(0, ".")
import pandas as pd
from src.icu.eicu_dataset import VENTILATION_KEYWORDS
labels, v1_hits, vent_values = collections.Counter(), collections.Counter(), collections.Counter()
for chunk in pd.read_csv("data/raw/EICU 2.0数据/respiratoryCharting.csv.gz",
                         usecols=["respcharttypecat", "respchartvaluelabel", "respchartvalue"], chunksize=2_000_000):
    chunk = chunk.fillna("").astype(str)
    labels.update(zip(chunk["respcharttypecat"], chunk["respchartvaluelabel"]))
    text = (chunk["respcharttypecat"] + " " + chunk["respchartvaluelabel"] + " " + chunk["respchartvalue"]).str.lower()
    hit = text.map(lambda t: any(k in t for k in VENTILATION_KEYWORDS))
    v1_hits.update(chunk.loc[hit, "respcharttypecat"])
    vent_values.update(chunk.loc[chunk["respchartvaluelabel"] == "RT Vent On/Off", "respchartvalue"])
pd.Series(labels).sort_values(ascending=False).to_csv("artifacts/v2/harmonization/eicu_resp_labels.csv")
print("v1 keyword hits by category:", dict(v1_hits))
print("RT Vent On/Off values:", vent_values.most_common(30))
EOF
```

**R-E5 通气起止**（`respiratoryCare.csv.gz`，9 MB）：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import pandas as pd
rc = pd.read_csv("data/raw/EICU 2.0数据/respiratoryCare.csv.gz", usecols=["patientunitstayid", "ventstartoffset", "ventendoffset", "airwaytype"])
print("stays:", rc["patientunitstayid"].nunique(),
      "with ventstartoffset:", rc.loc[rc["ventstartoffset"].notna(), "patientunitstayid"].nunique(),
      "ventstartoffset == 0 rows:", int((rc["ventstartoffset"] == 0).sum()))
print(rc["airwaytype"].value_counts().head(20).to_string())
EOF
```

R-E4 和 R-E5 的通过标准：
- 确认 `RT Vent On/Off` 存在，以及它的取值。
- 找出真正的呼吸机模式标签。
- 估计 `respiratoryCare` 的覆盖率。
- 覆盖率太低时，eICU 的 `ventilation_on` 保持 low，不进入共享特征集。

**R-E6 病人 ID**（`patient.csv.gz`，10 MB）：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import pandas as pd
p = pd.read_csv("data/raw/EICU 2.0数据/patient.csv.gz", usecols=["patientunitstayid", "uniquepid"], dtype={"uniquepid": str})
per = p.groupby("uniquepid")["patientunitstayid"].nunique()
print("stays", len(p), "patients", per.size, "patients with >1 stay", int((per > 1).sum()), "stays of those patients", int(per[per > 1].sum()))
EOF
```

**R-E7 血压来源和体温覆盖**（`vitalPeriodic.csv.gz` 1.8 GB，`vitalAperiodic.csv.gz` 240 MB，较重）：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import pandas as pd
inv, temp, nibp = [], [], []
for chunk in pd.read_csv("data/raw/EICU 2.0数据/vitalPeriodic.csv.gz", usecols=["patientunitstayid", "observationoffset", "systemicmean", "temperature"], chunksize=5_000_000):
    chunk["hour"] = chunk["observationoffset"] // 60
    inv.append(chunk.loc[chunk["systemicmean"].notna(), ["patientunitstayid", "hour"]].drop_duplicates())
    temp.append(chunk.loc[chunk["temperature"].notna(), ["patientunitstayid"]].drop_duplicates())
for chunk in pd.read_csv("data/raw/EICU 2.0数据/vitalAperiodic.csv.gz", usecols=["patientunitstayid", "observationoffset", "noninvasivemean"], chunksize=5_000_000):
    chunk["hour"] = chunk["observationoffset"] // 60
    nibp.append(chunk.loc[chunk["noninvasivemean"].notna(), ["patientunitstayid", "hour"]].drop_duplicates())
inv = pd.concat(inv).drop_duplicates().assign(invasive=True)
nibp = pd.concat(nibp).drop_duplicates().assign(noninvasive=True)
hours = inv.merge(nibp, on=["patientunitstayid", "hour"], how="outer")
hours[["invasive", "noninvasive"]] = hours[["invasive", "noninvasive"]].eq(True)
hours.to_parquet("artifacts/v2/harmonization/eicu_bp_modality_hours.parquet", index=False)
print(hours[["invasive", "noninvasive"]].value_counts().to_string())
print("stays with any vitalPeriodic temperature:", pd.concat(temp)["patientunitstayid"].nunique())
EOF
```

### MIMIC-IV

**R-M1 尿量条目**（`outputevents.csv.gz`，49 MB）。目的：确认 N1。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.icu.v2.harmonization import MIMIC_IV_TABLE
oe = pd.read_csv("data/raw/mimic-iv/mimic-iv-3.1/icu/outputevents.csv.gz", usecols=["stay_id", "itemid", "value", "valueuom"])
ids = {r.identifier for r in MIMIC_IV_TABLE if r.feature == "urine_output"}
s = oe[oe["itemid"].isin(ids)].groupby("itemid").agg(n=("value", "size"), stays=("stay_id", "nunique"),
                                                      median=("value", "median"), p99=("value", lambda v: v.quantile(0.99)),
                                                      uom=("valueuom", lambda u: ",".join(sorted(set(u.dropna())))))
s.to_csv("artifacts/v2/harmonization/mimic_urine_items.csv")
print(s.to_string())
print("stays with any outputevents:", oe["stay_id"].nunique())
EOF
```

通过标准：Foley 和 Void 占绝大部分行数，v1 的四个条目只覆盖少数 stay，这就确认了 N1。

**R-M2 输液条目**（`inputevents.csv.gz`，401 MB）。目的：确定 229617、229630–229632、229420、225942 的去留，统计 v1 当成「在用药」的推注行。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import collections, sys
sys.path.insert(0, ".")
import pandas as pd
from src.icu.v2.harmonization import MIMIC_IV_TABLE
ids = {r.identifier for r in MIMIC_IV_TABLE if r.namespace == "inputevents"}
acc = collections.Counter()
for chunk in pd.read_csv("data/raw/mimic-iv/mimic-iv-3.1/icu/inputevents.csv.gz",
                         usecols=["itemid", "ordercategorydescription", "rateuom", "rate", "statusdescription"], chunksize=2_000_000):
    c = chunk[chunk["itemid"].isin(ids)]
    acc.update(zip(c["itemid"], c["ordercategorydescription"].fillna(""), c["rateuom"].fillna(""), c["rate"].isna(), c["statusdescription"].fillna("")))
df = pd.DataFrame([(*k, n) for k, n in acc.items()], columns=["itemid", "ordercategorydescription", "rateuom", "rate_missing", "statusdescription", "n"])
df.sort_values(["itemid", "n"], ascending=[True, False]).to_csv("artifacts/v2/harmonization/mimic_drug_items.csv", index=False)
print(df.groupby(["itemid", "ordercategorydescription", "rate_missing"])["n"].sum().to_string())
EOF
```

**R-M3 化验条目和单位**（`labevents.csv.gz`，2.6 GB，较重）。目的：核对纳入条目的 `valueuom`，确定 11 个 pending 条目的去留。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import collections, sys
sys.path.insert(0, ".")
import pandas as pd
from src.icu.v2.harmonization import MIMIC_IV_TABLE
ids = {r.identifier for r in MIMIC_IV_TABLE if r.namespace == "labevents"}
acc = collections.Counter()
for chunk in pd.read_csv("data/raw/mimic-iv/mimic-iv-3.1/hosp/labevents.csv.gz", usecols=["itemid", "valueuom", "valuenum"], chunksize=5_000_000):
    c = chunk[chunk["itemid"].isin(ids)]
    acc.update(zip(c["itemid"], c["valueuom"].fillna(""), c["valuenum"].isna()))
df = pd.DataFrame([(*k, n) for k, n in acc.items()], columns=["itemid", "valueuom", "valuenum_missing", "n"])
df.sort_values(["itemid", "n"], ascending=[True, False]).to_csv("artifacts/v2/harmonization/mimic_lab_items.csv", index=False)
print(df.groupby(["itemid", "valueuom"])["n"].sum().to_string())
EOF
```

**R-M4 血压来源和通气取值**（`chartevents.csv.gz`，3.5 GB，最重，放最后）。目的：检验第 3 节的假设，并核对通气模式名单。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import collections
import pandas as pd
BP_INV = {220050, 220051, 220052, 225309, 225310, 225312}
BP_NIBP = {220179, 220180, 220181}
VENT_TEXT = {223849, 229314, 226732, 223848}
OTHER = {220339, 223835, 224690}
hours, vent_values, other_counts = [], collections.Counter(), collections.Counter()
for chunk in pd.read_csv("data/raw/mimic-iv/mimic-iv-3.1/icu/chartevents.csv.gz",
                         usecols=["stay_id", "itemid", "charttime", "value", "valuenum"], chunksize=5_000_000):
    bp = chunk[chunk["itemid"].isin(BP_INV | BP_NIBP) & chunk["valuenum"].notna()]
    if len(bp):
        hours.append(pd.DataFrame({"stay_id": bp["stay_id"].to_numpy(),
                                   "hour": pd.to_datetime(bp["charttime"]).dt.floor("h").to_numpy(),
                                   "invasive": bp["itemid"].isin(BP_INV).to_numpy()}).drop_duplicates())
    v = chunk[chunk["itemid"].isin(VENT_TEXT)]
    vent_values.update(zip(v["itemid"], v["value"].astype(str)))
    other_counts.update(chunk.loc[chunk["itemid"].isin(OTHER), "itemid"])
h = pd.concat(hours).groupby(["stay_id", "hour"])["invasive"].agg(any_invasive="max", any_noninvasive=lambda s: bool((~s).any()))
h.reset_index().to_parquet("artifacts/v2/harmonization/mimic_bp_modality_hours.parquet", index=False)
print(h[["any_invasive", "any_noninvasive"]].value_counts().to_string())
pd.DataFrame([(*k, n) for k, n in vent_values.items()], columns=["itemid", "value", "n"]).sort_values("n", ascending=False).to_csv(
    "artifacts/v2/harmonization/mimic_vent_values.csv", index=False)
print(dict(other_counts))
EOF
```

通过标准：
- 有有创血压的小时里，大部分没有无创血压，这就支持第 3.2 节的机制。
- 泄漏审计把 `mimic_bp_modality_hours.parquet` 和 v2 的 episode 窗口按 `stay_id`、小时对齐，按标签比较「最后一个 bin 只有有创血压」的比例。正例明显更高，就确认了假设。
- 通气文本取值覆盖 `MIMIC_VENT_MODES_ON` 等名单，没有遗漏常见模式。

### HiRID

**R-H1 ID 覆盖和给药语义**：
- 数据：`data/raw/hirid_raw_subset/raw_stage`，已解压，200 个 observation 分片和 120 个 pharma 分片，共 6.9 GB。
- 完整数据需要先把 `raw_stage/*.tar.gz` 解压到一个新目录：observation 6.6 GB，pharma 0.5 GB。

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import collections, glob, sys
sys.path.insert(0, ".")
import pandas as pd
import pyarrow.parquet as pq
from src.icu.v2.harmonization import HIRID_TABLE
ROOT = "data/raw/hirid_raw_subset/raw_stage"
obs_paths = sorted(glob.glob(f"{ROOT}/observation_tables/parquet/*.parquet"))
ph_paths = sorted(glob.glob(f"{ROOT}/pharma_records/parquet/*.parquet"))
print("observation columns:", pq.read_schema(obs_paths[0]).names)
print("pharma columns:", pq.read_schema(ph_paths[0]).names)
ph_ids = {r.identifier for r in HIRID_TABLE if r.namespace == "pharma"} | {300}
obs = collections.Counter()
for path in obs_paths:
    obs.update(pq.read_table(path, columns=["variableid"]).column("variableid").to_pandas().value_counts().to_dict())
ph = collections.Counter()
for path in ph_paths:
    names = pq.read_schema(path).names
    cols = [c for c in ("pharmaid", "doseunit", "typeid", "recordstatus") if c in names]
    t = pq.read_table(path, columns=cols).to_pandas()
    t = t[t["pharmaid"].isin(ph_ids)]
    ph.update(map(tuple, t[cols].astype(str).to_numpy()))
obs_df = pd.Series(obs).rename("n")
obs_df.to_csv("artifacts/v2/harmonization/hirid_observation_variableids.csv")
pd.DataFrame([(*k, n) for k, n in ph.items()]).to_csv("artifacts/v2/harmonization/hirid_pharma_semantics.csv", index=False)
mapped = {r.identifier for r in HIRID_TABLE if r.namespace == "observation"}
print("observation ids in the table but absent from data:", sorted(i for i in mapped if obs.get(i, 0) == 0))
print("pharmaid 300 rows:", sum(n for k, n in ph.items() if k[0] == "300"))
EOF
```

**R-H2 数值尺度**（只读 20 个分片）：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import glob
import pandas as pd
import pyarrow.parquet as pq
want = [24000524, 10020000, 3845, 15001552, 20000110, 20000600, 20005110]
frames = []
for path in sorted(glob.glob("data/raw/hirid_raw_subset/raw_stage/observation_tables/parquet/*.parquet"))[:20]:
    frames.append(pq.read_table(path, columns=["patientid", "datetime", "variableid", "value"],
                                filters=[("variableid", "in", want)]).to_pandas())
d = pd.concat(frames)
print(d.groupby("variableid")["value"].describe(percentiles=[0.01, 0.5, 0.99]).to_string())
u = d[d["variableid"] == 10020000].sort_values(["patientid", "datetime"])
print("minutes between hourly urine records:", (u.groupby("patientid")["datetime"].diff().dt.total_seconds() / 60).describe().to_string())
print(d[d["variableid"].isin([3845, 15001552])].groupby("variableid")["value"].value_counts().to_string())
EOF
```

R-H1 和 R-H2 的通过标准：
- 纳入的 ID 在数据里都存在。
- 从 typeid、recordstatus 能区分推注和持续输注，据此确定 1000462、112、1000251、251 和丙泊酚 1%/2% 的去留。
- pharmaid 300 是否真有记录：有的话，v1 处理 raw stage 时会把甲硝唑剂量混进 RR。
- 乳酸中位数约 1–2（确认单位是 mmol/l），肌酐中位数约 60–120 µmol/l，血小板中位数约 150–250 G/l。
- 10020000 的记录间隔中位数约 60 分钟。不是的话，要按实际间隔把 ml/h 换算成 mL。
- 3845 和 15001552 的编码分布与 `ordinal_vars_ref.csv` 一致。

**R-H3 重建前的确认**：
- 统计已处理数据的特征覆盖时，用 `os.scandir` 惰性读取，抽样几百个 episode，**不要**对 `data/processed/hirid_rebuild/hirid` 做 `ls` 或 glob。
- merged stage 只有 18 个变量，已确认，v2 不使用。
- 重建时 stage 目录要显式指定为 `raw_stage`。

## 9. 对外接口（`src/icu/v2/harmonization.py`）

### 数据和常量

- **词表**：`CANONICAL_FEATURES`（18 个，顺序与 v1 的 `get_feature_names("core")` 一致）、`SOURCES`、`NAMESPACES`、`MATCH_KINDS`、`DECISIONS`、`CONFIDENCE_LEVELS`、`V1_STATUSES`、`FEATURE_KINDS`。
- **血压**：`BP_FEATURES`、`BP_MODALITIES`、`BP_MERGE_RULE`、`AUXILIARY_INDICATORS`（= `("bp_source",)`，默认不作为模型输入）。
- **标准定义**：`CANONICAL_SPECS: dict[str, CanonicalFeature]`。
- **映射表**：`MIMIC_IV_TABLE`、`EICU_TABLE`、`HIRID_TABLE`（都是 `tuple[MappingRow, ...]`）、`MAPPING_TABLES`、`UNAVAILABLE_FEATURES`、`VALUE_RULES`。
- **取值常量**：`MIMIC_VENT_MODES_ON`、`MIMIC_HAMILTON_MODES_ON`、`MIMIC_O2_DEVICES_ON`、`EICU_URINE_CELLPATH_PREFIX`、`EICU_CODE_URINE_LABELS`、`EICU_CODE_URINE_LABEL_PATTERNS`、`EICU_URINE_LABEL_EXCLUSION`、`EICU_VENT_ON_VALUES`、`EICU_VENT_OFF_VALUES`、`EICU_NON_BLOOD_SPECIMEN`、`HIRID_VENT_MODE_CODES_ON`、`HIRID_VENT_MODE_CODES_OFF`、`HIRID_AIRWAY_CODES_ON`、`RULES`。

### 类

- `UnitConversion(name, factor=1.0, offset=0.0, binarize=False)`：有 `.to_canonical(values)`、`.from_canonical(values)` 和 `.is_identity`。
- `CanonicalFeature(name, kind, unit, low, high, accepted_units, definition)`
- `MappingRow(source, feature, namespace, identifier, match, decision, label, unit="", confidence="high", v1="kept", note="", sign=1.0, modality="")`

### 函数

```python
normalize_unit(unit: object) -> str
conversion_for(feature: str, unit: object) -> UnitConversion | None
to_canonical(feature: str, values: object, unit: object) -> float | np.ndarray
from_canonical(feature: str, values: object, unit: object) -> float | np.ndarray
lookup_identifier(source: str, namespace: str, identifier: object) -> MappingRow | None
map_identifier(source: str, namespace: str, identifier: object) -> str | None
is_excluded_mimic_item(itemid: object) -> bool
is_excluded_hirid_variable(namespace: str, variable_id: object) -> bool
bp_modality(source: str, namespace: str, identifier: object) -> str | None
classify_eicu_lab(labname: object) -> tuple[str | None, str]
map_eicu_lab(labname: object) -> str | None
is_excluded_eicu_lab(labname: object) -> bool
classify_eicu_drug(drugname: object) -> tuple[str | None, str]
is_excluded_eicu_drug(drugname: object) -> bool
is_eicu_urine_cellpath(cellpath: object) -> bool
cross_namespace_identifiers(source: str, rows: Iterable[MappingRow] | None = None) -> dict[object, tuple[str, ...]]
validate_mapping_tables(tables: Mapping[str, Sequence[MappingRow]] | None = None,
                        unavailable: Mapping[str, Mapping[str, str]] | None = None) -> list[str]
declared_units(source: str, feature: str) -> tuple[str, ...]
to_canonical_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame
check_units(frame: pd.DataFrame, source: str) -> pd.DataFrame
implausible_mask(frame: pd.DataFrame, source: str) -> pd.Series
plausibility_report(frame: pd.DataFrame, source: str) -> pd.DataFrame
merge_blood_pressure(frame: pd.DataFrame, source: str, keys: Sequence[str] = ("stay_id", "bin")) -> pd.DataFrame
mapping_frame(source: str) -> pd.DataFrame
feature_summary(source: str) -> pd.DataFrame
```

### 小表的列约定

- `check_units` 需要 `feature`、`unit` 两列。
- `plausibility_report`、`implausible_mask`、`to_canonical_frame` 需要 `feature`、`value` 两列，可以再给 `unit`，或者 `namespace` 加 `identifier`。单位按以下顺序确定：显式的 `unit` 列；标识对应行的源单位；该源对这个特征只声明了一种单位时用那一种；否则视为已是标准单位。
- `merge_blood_pressure` 需要 `keys`、`feature`、`value` 三类列，再加 `modality` 列，或者 `namespace` 加 `identifier`。

## 10. 测试

```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/v2/test_harmonization.py
29 passed
```

覆盖内容：
- 特征顺序与 v1 一致。
- 三张表通过校验，并覆盖全部 18 个特征。
- 校验函数能发现以下问题：同一命名空间内重复的标识；跨命名空间的 ID 缺少标签；声明为不可得却没写理由；同一特征既有映射又声明为不可得；单位无法换算；正则写错；血压行缺 modality；排除行没写理由。
- 查表必须带命名空间，HiRID 300 能正确分开。
- eICU 化验：尿肌酐、脑脊液葡萄糖、体液白细胞被排除，血清名称被接受。
- eICU 药名和尿量路径的分类结果正确。
- HiRID 和 MIMIC 的排除与修正生效。
- 所有非恒等换算都能往返还原，几个已知锚点值换算正确。
- `check_units` 的各种状态都能正确给出。
- 越界值能被标记出来，缺少单位时会按源的声明单位换算。
- 血压合并时优先有创，没有有创值时回退到无创，modality 无法确定时报错。

## 9. 原始数据核查之后的实际映射（2026-10-21）

原始数据核查已经全部跑完。**抽取时实际使用的映射以三个适配器为准**：`src/icu/v2/extract/mimic_iv.py`、`eicu.py`、`hirid.py`。每个适配器的模块文档里都有逐项决策和对应的证据计数，核查的原始输出在 `artifacts/v2/harmonization/`。和上面映射表不一致的地方以适配器为准，主要差异如下：

| 项目 | 上面映射表的写法 | 适配器实际采用的做法与证据 |
|---|---|---|
| 血糖 | 不含床旁血糖 | **纳入床旁/POC 血糖**，三个库统一：eICU 的血糖主要来自 bedside glucose（4.47M 行里有 3.16M 行）。MIMIC 只纳入指尖血糖 225664，它只有 0.5% 与化验结果相同，是独立测量；220621 和 226537 分别有 96% 和 90% 与化验、血气重复，是抄录副本，排除 |
| 镇静 | 含芬太尼 | 只纳入 propofol、midazolam、dexmedetomidine、lorazepam 的持续输注，阿片类默认不纳入（可选开启） |
| MIMIC 通气 | chartevents 里的呼吸机模式 | procedureevents 225792（有创通气区间），覆盖 84–85% 有有创通气记录的小时；无创通气默认不纳入 |
| MIMIC 尿量 | v1 的 4 个 itemid | 10 个尿量 itemid（Foley 和 Void 占 96%）；GU 冲洗液按同一 charttime 取出量减入量 |
| MIMIC 输液 | 删除 Paused/Stopped | **保留 Paused/Stopped**：这类行 100% 有实际给药量，删掉会把 5,300 多个住院的锚点推迟；Bolus 状态排除 |
| HiRID 呼吸频率 | 300、5685、310 | 只用 300：5685 不是呼吸次数；310 是呼吸机的自主呼吸频率，与 300 重复 |
| HiRID 尿量 | 直接作为体积 | 原始值是 ml/h 的速率，乘以距上一条记录的时长换算成体积，单次间隔最多算 12 小时 |
| HiRID 升压药 | 含多巴胺、去氧肾上腺素 | HiRID 没有这两种药。只有去甲肾上腺素、肾上腺素、血管加压素的持续输注；推注专用的 ID 排除 |
| eICU 通气 | 关键词规则 | **不提供**：ventendoffset 几乎全空，"RT Vent On/Off" 只覆盖 2,741 个住院 |
| eICU 尿量 | 部分 cellpath | 另外纳入 25 个尿量标签，包括 Indwelling Catheter Output 和 Voided Amount；排除 incontinent urine（它是次数） |
| eICU 体温 | vitalPeriodic | 另外纳入 nurseCharting 的 Temperature (C)：vitalPeriodic 的体温只覆盖 19k 个住院 |
| eICU 队列 | — | 排除不记录输液的医院（75 家）、有升压药证据但没有输液记录的住院，以及首次输液前已有升压药证据的住院；对照在同一家医院内匹配 |

另外，共用代码 `src/icu/v2/extract/common.py` 里的任务规则：
- 每名病人只保留 1 个正例；
- 对照病人在任何一次住院中都没有用过升压药；
- 截断前最后一小时必须有心率记录，正例和对照用同样的标准；
- 数据源没有的特征在所有时段都标为缺失。
