# Intro

一个能够对MT翻译后的句子进行批量化打分的程序

主要包含三个评价指标：scare-bleu、comet22、bert-score，可以根据个人选择输出

输出包含两个excel，`*out.excel`为纵向的各个模型和语言的结果，`*res.excel`为横向的每个模型，在所选指标上的各个语言的输出，方便复制粘贴到横向表格中

该脚本支持多个不同的LLM在不同的翻译方向上推断的结果进行打分，推断结果来源依靠LLama-Factory的输出，具体使用为指定推断结果所在的目录，文件结构为`目录/输出/generated_predictions.jsonl`，一个例子为：`exam_out/qwen1.5-1.8B-Chat-1560_wmt22/Qwen1.5-1.8B-Chat_wmt22_cs2en-epoch_1560/generated_predictions.jsonl`

该脚本自动获取模型名`Qwen1.5-1.8B-Chat`，翻译方向`cs2en`，具体依靠`_`进行分割。

`generated_predictions.jsonl`中一个样例为:

```json

{"label": "The big comeback of the Czech underdog is coming.", "predict": "The Czech potash minerals are getting close to a major return."}
{"label": "Pavel Francouz has been called up to the NHL", "predict": "Pavel Francouz was called up to the NHL"}

```

文件整体结构为：

```
MT_evaluate_batch
    - excelfiles
    - logs
    - datasets
        - wmt-testset
    - exam_out
    comet_batch.py

```

其中`excelfiles`存储输出的excel，`logs`存放过程日志，`datasets`存放测试集，`exam_out`存放用llama-factory的推断结果或者自定义的要打分的推断翻译结果。