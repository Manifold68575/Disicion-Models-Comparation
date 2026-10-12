# 实验协议 v0.1

## 数据与版本

每个样本为 `sample_id`、`text`、`label` 三个非空字符串；目录为 `version` 与 label → description 映射。重复样本 ID、未知真值、空数据和非法目录在发请求前被拒绝。Adapter 仅拿到 ID 与文本，不拿到真值。

正式比较要求相同输入文件及目录哈希。BERT/classic 训练完成时冻结目录签名；新目录不匹配返回 409，不能把旧分类头当成支持新增类别。BERT 使用验证集检查训练，最终测试集不得用于调参。Checkpoint 的 300 条验证子集不是最终测试报告。

Controller 自动记录 Python、平台、依赖、配置/数据/目录/源码哈希与 Git 状态，保存输入和源码快照。Worker 的 SKU、CPU/GPU、驱动、CUDA、容器 digest、模型 commit、训练 artifact hash、功率限制、线程数与网络路径需填写在配置的 `environment` 中。远端环境无法从 Controller 自动推断。

## 质量

Quality Runner 顺序请求，预热单独记录。请求失败在 Accuracy 分母和该真值类别的 FN 中保留；同时给出成功请求条件下的 Accuracy。主 Macro-F1 对冻结目录全部类别平均，缺失支持类别置零，并另外提供有真值支持类别的 Macro-F1。混淆矩阵以 `__failure__` 表示失败，逐类统计便于独立复算。

目录可声明 `reject_label`，拒识率按所有请求计算。当前无自动阈值校准和 OOS detection 指标；CLINC150 尚未接入。`scores` 原样按类型保存：Jev provider probability、BERT uncalibrated softmax、SVM uncalibrated margin。BERT/SVM 不伪造统一置信度。

## 开放环负载

固定间隔到达，计划发送时间为 `i / offered_rps`，小于 `duration_s` 的到达均计数。样本按固定 seed 有放回抽取。客户端任务数受到 `max_concurrency` 限制，满时计入 `client_overload`，不创建无限队列，也不等待完成后调整到达率。

E2E 从计划到达到完成，包含 Controller 调度迟滞、HTTP 连接池、网络与服务延迟。服务调用延迟从实际 dispatch 计到完成，并不等同于纯模型计算耗时。Timeout 从计划到达开始计算。HTTP 重试关闭，HTTP 4xx/5xx、传输错误、超时和非法输出都有独立状态。

延迟分位数同时给出 attempted（包含超时/错误）和 successful。客户端拒绝没有完成一次服务调用，所以没有服务 E2E 延迟；它仍降低成功率并计入 offered 分母。调度 lag 另行报告；最大 lag 超过阈值时不把该行认作有效容量候选。

吞吐和 Goodput 使用完整发送窗口加排空时间作为分母。Correct Goodput 再加语义正确条件。候选容量需本轮成功率 ≥ `min_success_rate`，attempted p95 ≤ SLO，调度有效；最终还要满足独立质量基线，重复 3–5 轮与更长稳态窗口。120 秒配置是初测起点，不是研究级验收。

当前调度器是固定速率，Poisson 到达、持续监测客户端 CPU、稳态分段、置信区间是 P1。JSONL 同步写入和内存聚合可能限制极高 RPS；调度迟滞可发现一部分客户端瓶颈。单次最多允许 100 万计划请求。

## 动态回放

`dynamic` 使用开放环时间表，在 stage 的 `at_s` 切换冻结目录。阶段按计划到达归属，慢响应跨越阶段边界时仍属于原阶段。整个 run 成本只核算一次；各阶段耗时可能重叠，因此不逐阶段分摊实例成本。

本轮只验证目录版本回放和新增类别造成的兼容性变化，尚未实现模型重训、缓存失效、恢复时间、维护成本与旧类退化研究。研究前需要新旧类分组测试集和明确的更新事件协议。

## 成本

价格必须显式配置 `currency`，并写清账单来源、价格日期、区域与汇率。示例价格只用于程序验证。Instance：测量加排空秒数 × 小时价 / 3600。API：所有已知 usage（包含解析失败响应）按输入/输出 Token 单价计费；usage 缺失时标为下界。

单位成本分别除以 successful 和 SLO-valid 请求数；零分母输出 null。预热、空闲、训练、网络固定成本未纳入逐请求测量价。月度 TCO 默认 730 小时，按峰值与平均需求所需的常驻副本数向上取整，并通过利用率目标留出余量，再加训练及维护成本。使用的是经过独立验收的可持续容量，不能使用一次最大吞吐。

## 部署

GPU Worker 轮流运行 Qwen 和 BERT GPU，保持相同驱动、功率、背景负载和测试窗口。CPU 与 GPU 共机时明确成本分配；正式 CPU 对照使用独立实例或整机账单。Jev 单独记录 Controller/API 网络路径。Prometheus 服务内部指标补充客户端 E2E；本项目尚未做实际容器或 GPU 验证。
