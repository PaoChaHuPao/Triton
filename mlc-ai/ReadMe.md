[Relax: TVM 的下一代图层级 IR - 知乎](https://zhuanlan.zhihu.com/p/523395133)
这里给出的链接并非问题来源只是对TVM的一些补充,

### **TVM Relax IR 问题清单**

#### 基础概念

1. 什么是 TVM Relax IR？它的设计目标是什么？
2. Relax IR 和 TVM 原有的 Relay IR 有什么区别？
3. Relax IR 如何支持动态形状（Dynamic Shape）？
4. Relax IR 的主要组成部分有哪些？（如 `VarNode`、`CallNode` 等）

#### 动态计算图

5. 如何在 Relax IR 中定义一个动态计算图？
6. Relax IR 如何处理控制流操作（如 `if` 和 `for`）？
7. 什么是符号形状（Symbolic Shape）？它在 Relax IR 中如何表示？

#### 算子与优化

8. 如何在 Relax IR 中定义一个自定义算子？
9. Relax IR 如何与 TVM 的底层表示（如 TIR）进行交互？
10. Relax IR 的优化流程是怎样的？

---

### **TVM AutoScheduler 问题清单**

#### 基础概念

1. 什么是 TVM AutoScheduler？它的主要作用是什么？
2. AutoScheduler 和 Ansor 是什么关系？
3. AutoScheduler 的搜索空间（Search Space）是如何定义的？
4. AutoScheduler 如何搜索最优的算子调度策略？

#### 搜索与优化

5. 在 AutoScheduler 中，什么是 `measure` 阶段？它的作用是什么？
6. 如何通过 AutoScheduler 优化一个矩阵乘法（MatMul）算子？
7. AutoScheduler 如何处理多线程和并行化优化？
8. AutoScheduler 的搜索结果如何保存和复用？

#### 性能评估

9. AutoScheduler 如何评估生成的调度策略的性能？
10. AutoScheduler 的优化结果如何与手动调优的结果进行比较？

---

### **综合应用问题清单**

1. 如何将 Relax IR 和 AutoScheduler 结合使用来优化一个深度学习模型？
2. 在 Relax IR 中定义一个简单的算子（如 `add`），并使用 AutoScheduler 优化它，你会怎么做？
3. Relax IR 的动态形状支持如何影响 AutoScheduler 的优化过程？
4. 如何使用 Relax IR 和 AutoScheduler 优化一个包含动态形状的模型？
5. 在实际项目中，如何调试和验证 Relax IR 和 AutoScheduler 生成的代码？

