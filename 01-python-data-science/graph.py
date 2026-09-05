graph LR
    A[用户问题] --> B[模型A生成答案]
    A --> C[模型B生成答案]
    A --> D[Ground Truth / 参考答案]
    B --> E[GPT-4评委]
    C --> E
    D --> E
    E --> F[综合打分与排名]