# include/cutriton

本目录对应 `cutriton` C++ 命名空间，是公共 API 的根目录。

应用可以只包含：

```cpp
#include <cutriton/cutriton.h>
```

库内部和关注编译速度的使用方应直接包含所需模块头文件。

## 模块关系

```text
core
 ├─ ir
 ├─ backend
 └─ runtime
      ▲
      └─ compiler（汇合 ir、backend 和 runtime）
```

- `core` 提供所有层共享的基础类型。
- `ir` 只描述计算语义，不执行 Kernel。
- `backend` 判断节点能否由目标设备执行并创建 Kernel。
- `compiler` 优化 Graph、选择后端并生成内存计划。
- `runtime` 持有编译结果和设备资源，负责绑定与执行。

对应实现位于仓库根目录的 `src/`。当前 CMake 将本目录暴露为
`cutriton` target 的 public include 路径。
