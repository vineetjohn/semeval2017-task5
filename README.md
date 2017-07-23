# UW-FinSent

This project documents an attempt at Task 5 for Semeval 2017
The current implementation focuses on Subtask 2 - Headlines

http://alt.qcri.org/semeval2017/task5/

If referenced for usage in your project, please cite using

```
@InProceedings{john-vechtomova:2017:SemEval,
  author    = {John, Vineet  and  Vechtomova, Olga},
  title     = {UW-FinSent at SemEval-2017 Task 5: Sentiment Analysis on Financial News Headlines using Training Dataset Augmentation},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {872--876},
  abstract  = {This paper discusses the approach taken by the UWaterloo team to arrive at a
	solution for the Fine-Grained Sentiment Analysis problem posed by Task 5 of
	SemEval 2017. The paper describes the document vectorization and sentiment
	score prediction techniques used, as well as the design and implementation
	decisions taken while building the system for this task. The system uses text
	vectorization models, such as N-gram, TF-IDF and paragraph embeddings, coupled
	with regression model variants to predict the sentiment scores. Amongst the
	methods examined, unigrams and bigrams coupled with simple linear regression
	obtained the best baseline accuracy. The paper also explores data augmentation
	methods to supplement the training dataset. This system was designed for
	Subtask 2 (News Statements and Headlines).},
  url       = {http://www.aclweb.org/anthology/S17-2149}
}
```
