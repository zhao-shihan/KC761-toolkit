# 任务：编写基于 Geant4 与 geant4_pybind 的 γ 谱仪模拟

## 1. 总体目标
使用 `geant4_pybind` 编写一个 γ 谱仪 Monte Carlo 模拟程序。几何包括 γ 探头和源，提供若干种固定几何与核素类型的源，用户通过命令行标志选择。模拟需保存逐脉冲能量沉积，支持多进程并行运行，并可选可视化。

## 2. 工作环境与依赖
- 创建一个 Python 虚拟环境，并在其中安装所需依赖，包括 `geant4_pybind`。
- 使用 pip 安装依赖。
- 主要参考材料：
  - Geant4 源代码与示例：`~/Public/build/geant4/11.4.2/geant4-11.4.2`
  - Geant4 在线用户手册：https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/index.html
  - 本项目目录下的 `.prompt/geant4_pybind` 项目源代码与示例（与具体编码无关，但可参考其用法）。
- **在开始编写代码前，必须充分阅读上述参考资料并理解相关 API 和示例。**

## 3. 模拟几何与材料
### 3.1 探测器
- 探头晶体：CsI(Tl)（Tl 掺杂浓度 0.1 mol%），尺寸 `10 mm × 10 mm × 25.4 mm`，密度 `4.51 g/cm³`。
- 晶体外壳：0.5 mm 厚 ABS 塑料，密度可取常用 ABS 密度（约1.05 g/cm³）。
- 世界体积：`30 cm × 30 cm × 30 cm` 的空气正方体。
- 坐标系：探测器晶体长轴沿 z 轴，10 mm × 10 mm 的面为探测端面；外壳前端面位于 z = +13.2 mm。源放置在探测器前方（+z），面到面间隔沿 z 轴计算（均为 1 mm）。容器管（Th-232 玻璃管、Ra-226 不锈钢管）与源球为**并列**关系（源位于管腔内，不嵌套）。

### 3.2 源选项
用户通过命令行 flag 选择以下固定源几何与核素：

- **`--k40`**
  - 几何体：无水碳酸钾，尺寸 `13 cm × 8 cm × 6 cm`，总质量 500 g；13×8 cm 面朝向探头。
  - 密度：在代码中根据质量和体积计算。
  - 与探头的面到面间隔：1 mm。
  - 源：K-40 均匀分布在该几何体中（子体稳定，仅母核衰变）。

- **`--lu176`**
  - 几何体：氧化镥，尺寸 `3 cm × 3 cm × 0.5 cm`。
  - 密度：按较密实粉剂设置（5.5 g/cm³，约为晶体密度 9.42 g/cm³ 的 58%，代码中注明）。
  - 与探头的面到面间隔：1 mm。
  - 源：Lu-176 均匀分布在该几何体中（子体稳定，仅母核衰变）。

- **`--am241`**
  - 几何体：圆形金片/金箔，直径 2 mm，厚度 3 μm。
  - 与探头的面到面间隔：1 mm。
  - 源：Am-241 均匀分布在该金几何体中；**仅模拟母核自身衰变**（用 `/process/had/rdm/nucleusLimits` 排除长寿命子体 Np-237 的假衰变链）。

- **`--th232`**
  - 玻璃管：外径 2.2 cm，壁厚 1 mm，长 5 cm；轴沿 y 方向放置。
  - 管外壁与探头的面到面间隔：1 mm。
  - 管内正中心放置一个直径 1.5 cm 的五水合硝酸钍球，总质量 10 g；球密度在代码中根据质量和球体积计算。
  - 源：Th-232 均匀分布在该五水合硝酸钍球中。
  - 假设球已达到长期平衡 → **模拟完整衰变链**。

- **`--ra226`**
  - 不锈钢管：外径 6 mm，壁厚 0.5 mm，长 5 mm；轴沿 z 方向放置。
  - 管末端（靠近探头的一端）与探头的面到面间隔：1 mm。
  - 管内正中心放置一个直径 5 mm 的玻璃球。
  - 源：Ra-226 均匀分布在该玻璃球中；模拟完整衰变链（长期平衡，含 Bi-214 等子体 γ 线）。

## 4. 物理过程
- 物理列表采用 **`QBBC_EMZ`** + 放射性衰变（`kc761sim/physics.py` 中的 `PhysicsList` 类，B3 风格子类化 `G4VModularPhysicsList` 手动组装：`G4EmStandardPhysics_option4` EM + QBBC 各强子分量）。
  - 关键限制：geant4_pybind 0.1.3 的物理列表工厂返回的列表无法交给 `SetUserInitialization`（smart_holder/disown 缺陷），且绑定无 `ReplacePhysics`/`RemovePhysics`，因此不能直接通过工厂创建 `QBBC_EMZ`，只能手动组装。
  - `QBBC` 默认**不**注册放射性衰变（只有 `G4DecayPhysics`），必须手动 `RegisterPhysics(G4RadioactiveDecayPhysics())` 启用。
- 源核素通过 Geant4 放射性衰变/离子方式定义：GPS（`/gps/ion Z A 0 0`）+ RDM，离子静止，由 RDM 产生 γ。
- 最近版本 Geant4 默认禁用长寿命核素的衰变：在 `Initialize()` 后通过 `/process/had/rdm/thresholdForVeryLongDecayTime <阈值> year` 按源提升时间阈值启用（Th-232/Ra-226/K-40/Lu-176 用 1e60 年，Am-241 用 1e5 年）。

## 5. 输出与保存
- 使用 `G4Analysis`（`G4RootAnalysisManager`）保存。ROOT ntuple 名 `kc761_data`，列：`event_id`(int)、`edep`(float，keV)、`time`(double，秒)；另有直方图 `kc761_spectrum`（0–4096 keV，4096 bin，即 1 keV/bin）。
- 能量沉积以 **float** 类型保存，只在存储时才转换；代码中的计算均使用双精度。
- **堆积（pile-up）处理**：逐事例记录每个能量沉积及其全局时间；事例模拟结束后按 10 µs 分辨时间合并为脉冲（自首个沉积起，一个分辨时间内的沉积合并为一脉冲，再迭代至序列耗尽）。每个脉冲输出一行，`time` 取该组首个沉积时刻。
- 能量为 0 的事例/脉冲不输出。
- 输出 ROOT 文件名由用户通过命令行传入，默认值为 `sim_output.root`。

## 6. 命令行接口与运行
- 命令行参数：源 flag（`--k40`/`--lu176`/`--am241`/`--th232`/`--ra226`，互斥必选）、`-o/--output`（默认 `sim_output.root`）、`-n/--events`、`-t/--threads`、`-s/--seed`（随机种子，worker i 用 seed+i+1）、`-v/--verbose`（batch 默认 0，interactive 默认 1；`/tracking/verbose` 保持默认 0 不设置）。
- 用户从命令行传入输出 ROOT 文件名与要模拟的事例数；若未传入事例数（无 `-n`），则启动 Geant4 可视化交互会话（`kc761sim/script/` 下的宏，几何半透明 + perspective 投影）。
- 并行：因 geant4_pybind 的 GIL 限制，Geant4 原生多线程（G4MTRunManager）会死锁，故采用 **multiprocessing 多进程**并行（`-t N`），每个进程独立跑 Serial 运行管理器并写分块 ROOT 文件，最后用 uproot 合并为最终输出（`event_id` 全局连续）。

## 7. 项目结构
- 主文件 `sim.py`；实现拆分为包 `kc761sim/`，保持项目结构清晰、可维护：
  - `config.py`：源配置（核素、几何、材料、密度、衰变链策略）
  - `detector.py`：世界/探测器/源几何
  - `materials.py`：材料定义
  - `physics.py`：`PhysicsList`（QBBC_EMZ 组装）、RDM 配置、GPS 配置
  - `actions.py`：用户动作与逐脉冲能量沉积评分
  - `runner.py`：单进程运行、multiprocessing 分块、uproot 合并
  - `script/`：可视化宏（init_vis.mac / vis.mac / gui.mac）
- 代码遵循 Python 项目的最佳实践（PEP 8 命名：类 CapWords、函数/变量 snake_case、常量 ALL_CAPS）。

## 8. 协作要求
- 编写代码前，请充分阅读 `geant4_pybind` 示例代码、Geant4 用户手册中的相关章节，以及本地 Geant4 示例。
- 遇到任何不确定、歧义或需要决策的问题，**不要自行假设，先向用户确认**。
