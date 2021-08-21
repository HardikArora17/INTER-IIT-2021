import pandas as pd
import datasets
from datasets import Dataset
from headliner.model.transformer_summarizer import TransformerSummarizer
from headliner.trainer import Trainer
import yake
import re
import nltk
from nltk.corpus import stopwords
import heapq
from summarizer import Summarizer
from num2words import num2words

df=pd.read_csv("/home/deekshapcs18/anushkha/inter__iit/mobile_articles.csv",encoding='utf-8')
# print(df.head())
# print(df.columns)
df.drop(['Unnamed: 0'],axis=1,inplace=True)
# df['Article']=df['Article'].apply(lambda x:str(x))
# df['Headline']=df['Headline'].apply(lambda x:str(x))
model = Summarizer()
def extractive_summarization(body):
	# print(body)
	result = model(body, num_sentences=7)
	# print(result)
	# full = ''.join(result)
	return result
def remove_end(text):
	text = text.split()
	text.remove('<end>')
	text = ' '.join(text)
	return text

def num2word(text):
  # text=int(text)
  text1 = []
  
  for word in text.split():
    print(word)
    if word.replace(',', '').isdigit():
      word=word.replace(',','')
      print(word)
      text1.append(','.join([num2words(int(i)) for i in word.split(',') if i!='']))
    elif word.replace('.', '').replace(',','').isdigit():
      word=word.replace(',','')
      print(word)
      text1.append('.'.join([num2words(int(i)) for i in word.split('.') if i!='']))
    elif word.replace('.', '').isdigit():
      text1.append('.'.join([num2words(int(i)) for i in word.split('.') if i!='']))
    elif word.replace('/', '').isdigit():
      text1.append('/'.join([num2words(int(i)) for i in word.split('/') if i!='']))
    else:
      text1.append(word)
      
  text1 = ' '.join(text1)
  return text1

def numtoint(x):
	for i in x:
		if i.isdigit():
			i=int(i)
	print(x)
	return x

df['Article']=df['Article'].apply(lambda x: num2word(x))
df['Headline']=df['Headline'].apply(lambda x: num2word(x))
df['Article']=df['Article'].apply(lambda x : extractive_summarization(x))

train_data=[]
df1=df[:358]
# print(df1['Article'].loc[0])
for i in range(358):
  train_data.append((df1['Article'].loc[i],df1['Headline'].loc[i]))

val_data=[]
df2=df[358:511]
for i in range(358,511):
  val_data.append((df2['Article'].loc[i],df2['Headline'].loc[i]))


summarizer = TransformerSummarizer(num_heads=1,
                                   feed_forward_dim=512,
                                   num_layers=1,
                                   embedding_size=64,
                                   max_prediction_len=50)

trainer = Trainer(batch_size=4,
                  steps_per_epoch=50,
                  max_vocab_size_encoder=10000,
                  max_vocab_size_decoder=10000,
                  steps_to_log=50)

trainer.train(summarizer, train_data, val_data=val_data, num_epochs=30)
summarizer.save('/tmp/summarizer')


summarizer = TransformerSummarizer.load('/tmp/summarizer') 
eval_df = pd.read_excel("/home/deekshapcs18/anushkha/inter__iit/ASPECT_NEW_FINAL_BKCHODI _KHTM .xlsx")
eval_df['ARTICLES'].fillna('',inplace=True)
print(eval_df)
print(eval_df.info())
pred_list=[]
for i in range(222):
	if eval_df.loc[i]['MOBILE TECH OR NOT']==1:
		prediction=summarizer.predict(eval_df['ARTICLES'].loc[i])
		prediction=remove_end(prediction)
		print(prediction)
		pred_list.append(str(prediction))
	else:
		pred_list.append('')
eval_df['Headline_Generated_Eng_Lang']=pred_list

eval_df.to_csv('sample_output_1.csv')
