## **DeepLink多元算力混合推理加速方案**

DeepLink 多元算力混合推理加速方案是面向多元国产芯片的全域异构推理解决方案。其由上海人工智能实验室自研，基于 PD 分离架构实现多品类国产芯片异构协同推理。方案打破单芯片集群部署局限，借助智能路由、高速互联与策略自适应调度，优化 TTFT 最高 34\.5%、推理吞吐提升 32%，可灵活支撑 AIGC、AI4S、具身仿真等场景。项目持续开源迭代，后续集成 MoonCake 实现跨机房跨地域全域混推，构建国产训推一体异构算力生态。

方案的四大底座包括：

- **推理中间件（**[**DLInfer**](https://github.com/DeepLink-org/dlinfer)**）**：以标准化融合算子接口打通上层框架与底层硬件壁垒，实现算法模型在多元硬件上的统一推理，降低应用门槛。

- **高速通信库（**[**DLSlime**](https://github.com/DeepLink-org/DLSlime)**）**：全面兼容各类主流物理连接协议，实现跨架构设备高速互联，核心场景带宽利用率突破97%；具有较强的异步处理能力，可实现计算与通信的重叠。

- **智能流量路由系统（**[**DLRouter**](https://github.com/DeepLink-org/DLRouter)**）**：支持KVCache感知的请求路由，最大限度减少重复或重叠请求，节省计算资源，实现分布式集群负载均衡分配。

- **策略求解器（DLSolver）**：自动获取异构芯片全方位评测数据，结合模型配置以及用户服务等级目标等输入，匹配最优PD分离配置策略，兼顾推理性能与成本。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDdhODFhYjcyNThjMzRhYTU5NjlkYjAyN2VlMjU5MDhfNzI3Mjc0YTZlMzAyYzlmZGU0NjhhOGUzNmRiM2QxMDVfSUQ6NzY0Nzc4NTEyNDA3MTU5MTEyMV8xNzgwNjQ4MjM0OjE3ODA3MzQ2MzRfVjM)

## **DeepLink混合推理方案**

上海AI实验室牵头携手主流国产芯片共同建设DeepLink国产芯片**混合推理方案**，整合三类互补的推理部署模式，全方位覆盖产业落地需求，包括：

- **异构集群间混合调度：**支持多异构芯片推理集群的统一调度，如用户数据中心内有A、B、C三类国产芯片，在各集群上部署独立的推理服务，通过DLRouter感知各集群硬件配置，在整体服务网关层面实现推理任务在多集群间负载均衡调度，提升整体资源利用率；

- **异构国产芯片混推：**支持多款异构国产芯片的PD分离推理，如用户数据中心内有A、B两类国产芯片，结合芯片特定确定计算密集、访存密集等请求特征，完成算力节点最优匹配，实现Prefill阶段跑在A集群上，Decode阶段跑在B集群上。突破单卡性能瓶颈，提升异构芯片集群整体吞吐；

- **国产芯片\+NV混推：**支持国产芯片和NV芯片的PD分离推理，当前实际产业需求里已大量存在。因部分国产芯片采用DSA架构，硬件设计侧重堆叠张量算力，更适合用于Prefill大批量并行矩阵运算阶段。而Decode 阶段需要高频随机读写 KV 缓存，依赖大访存带宽，N卡的高规格HBM特性能更好处理Decode阶段任务

#### 混合推理方案能力一览

<table>
  <tr>
    <th>类别</th>
    <th>支持架构</th>
    <th>厂商</th>
    <th>镜像</th>
  </tr>
  <tr>
    <td rowspan="7">异构集群间混合调度</td>
    <td rowspan="5">vLLM</td>
    <td>天数+NV</td>
    <td></td>
  </tr>
  <tr>
    <td>昇腾</td>
    <td></td>
  </tr>
  <tr>
    <td>海光</td>
    <td></td>
  </tr>
  <tr>
    <td>燧原</td>
    <td></td>
  </tr>
  <tr>
    <td>壁仞</td>
    <td></td>
  </tr>
  <tr>
    <td>LMDeploy</td>
    <td>沐曦+平头哥</td>
    <td></td>
  </tr>
  <tr>
    <td>SGLang</td>
    <td>摩尔</td>
    <td></td>
  </tr>
  <tr>
    <td>异构国产芯片混推<br>（PD分离）</td>
    <td>LMDeploy</td>
    <td>沐曦+平头哥</td>
    <td>
      <ul>
        <li>crpi-z3i6df2ze96lh4ld.cn-hangzhou.personal.cr.aliyuncs.com/deeplink_infer/deeplink_infer:maca-lmdeploy</li>
        <li>crpi-z3i6df2ze96lh4ld.cn-hangzhou.personal.cr.aliyuncs.com/deeplink_infer/deeplink_infer:v1.5.2-lmdeploy0.10.0</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>国产芯片+NV混推</td>
    <td>vLLM</td>
    <td>天数+NV</td>
    <td>
      <ul>
        <li>NV：ghcr.io/deeplink2026/vllm/vllm-openai:nv_nixl</li>
        <li>天数：ghcr.io/deeplink2026/vllm-openai:iluvatar_nixl（暂未公开）</li>
      </ul>
    </td>
  </tr>
</table>