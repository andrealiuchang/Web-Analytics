import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import wordcloud
from matplotlib import pyplot as plt
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

text = open('ns_controller.txt', 'r', encoding='utf-8').read().lower()

sw = stopwords.words('english')

#去除停用词
my_stopword = [r"n't", r"wii", r"gon", r"na", r"u"]      #自己看情况加
sw += my_stopword
tokens = [token for token in word_tokenize(text) if token not in sw]

#词性还原
stemmed = [WordNetLemmatizer().lemmatize(token, pos='v') for token in tokens]

#去除标点符号
sign = re.compile('^[-,.()?:;"\'!]')
stemmed = [word for word in stemmed if not sign.findall(word)]
f = open('ns_controller_dc.txt', 'w')
f.write(str(stemmed))
f.close()

#计算词频，返回的是字典形式
Freq_dist_nltk = nltk.FreqDist(stemmed)
data = pd.DataFrame([Freq_dist_nltk[key] for key in Freq_dist_nltk],
                    index=[key for key in Freq_dist_nltk],
                    columns=['frequency'])
data.to_csv('wordFre-ns_controller.csv')

#画图
print(Freq_dist_nltk.plot(50, cumulative=False))

#显示最多的项
print(Freq_dist_nltk.tabulate(50))

#生成词云
wordcloud = wordcloud.WordCloud(
background_color='white', #设置背景为白色，默认为黑色
width=1500, #设置图片的宽度
height=960, #设置图片的高度
margin=10 #设置图片的边缘
).generate(str(stemmed))

plt.imshow(wordcloud)
plt.show()