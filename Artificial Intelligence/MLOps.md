# MLOps | [Databricks](https://www.databricks.com/glossary/mlops) | [Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

MLOps stands for Machine Learning Operations. MLOps is a core function of Machine Learning engineering, focused on streamlining the process of taking machine learning models to production, and then maintaining and monitoring them. MLOps is a collaborative function, often comprising data scientists, devops engineers, and IT.

Continuous delivery and automation pipelines in machine learning

This document discusses techniques for implementing and automating continuous integration (CI), continuous delivery (CD), and continuous training (CT) for machine learning (ML) systems.

Data science and ML are becoming core capabilities for solving complex real-world problems, transforming industires, and delivering value in all domains. Currently, the ingredients for applying effective ML are available to you:

- Large datasets
- Inexpensive on-demand compute resources
- Specialized accelerators for ML on various cloud platforms
- Rapid advances in different ML research fields (such as computer vision, natural language understanding, and recommendations AI systems).

Therefore, many business are investing in their data science teams and ML capabilities to develop predictive models that can deliver business value to their users.

This document is for data scientists and ML engineers who want to apply DevOps principles to ML systems (MLOps). MLOps is an ML engineering culture and practice that aims at unifying ML system development (Dev) and ML system operation (Ops). Practicing MLOps means that you advocate for automation and monitoring at all steps of ML system construction, including integrations, testing, releasing, deployment and infrastructure management.

Data scientists can implement and train an ML model with predictive performance on an offline holdout dataset, given relevant training data for their use case. However, the real challenge isn't building an ML model, the challenge is building an integrated ML system and to continuously operate it in production. With the long history of production ML services at Google, we've learned that there can be many pitfalls in operating ML-based systems in production. Some of these pitfalls are summarized in Machine Learning: The high-interest credit card of technical debt.

Only a small fraction of a real-world ML system is composed of the ML code. The required surrounding elements are vast and complex.

The rest of the system is composed of configuration, automation, data collection, data verification, testing and debugging, resource management, model analysis, process and metadata management, serving infrastructure, and monitoring.

To develop and operate complex systems like these, you can apply DevOps principles to ML systems (MLOps). This document covers concepts to consider when setting up an MLOps environment for your data science practices, such as CI, CD, and CT in ML.

### Continous Machine Learning | [Blog](https://levity.ai/blog/what-is-continuous-machine-learning)

### MLOps vs. DevOps | [Blog KR](https://techscene.tistory.com/entry/MLOps-%EA%B6%81%EA%B7%B9-%EA%B0%80%EC%9D%B4%EB%93%9C-MLOps-%EB%B0%8F-DevOps%EC%9D%98-%EA%B0%9C%EB%85%90%EA%B3%BC-%EC%B0%A8%EC%9D%B4%EC%A0%90-%EC%9D%B4%ED%95%B4)

---

### Reference
- MLOps Databricks, https://www.databricks.com/glossary/mlops, 2022-10-04-Tue.
- MLOps Google Cloud, https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning, 2022-10-11-Tue.
- Continous Machine Learning Blog, https://levity.ai/blog/what-is-continuous-machine-learning, 2023-03-14-Tue.
- MLOps Pipeline Kakao, https://tech.kakaopay.com/post/ifkakao2022-mlops-model-training-pipeline/, 2023-09-11-Mon.
- MLOps vs. DevOps Blog KR, https://techscene.tistory.com/entry/MLOps-%EA%B6%81%EA%B7%B9-%EA%B0%80%EC%9D%B4%EB%93%9C-MLOps-%EB%B0%8F-DevOps%EC%9D%98-%EA%B0%9C%EB%85%90%EA%B3%BC-%EC%B0%A8%EC%9D%B4%EC%A0%90-%EC%9D%B4%ED%95%B4, 2023-09-11-Mon.
- MLOps Blog KR, https://seunghan96.github.io/mlops/mlops%EC%A0%95%EB%A6%AC/, 2023-09-11-Mon.
