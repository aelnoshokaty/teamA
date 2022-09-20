# load twitter library - the rtweet library is recommended now over twitteR
install.packages("rtweet")
library(rtweet)
# plotting and pipes - tidyverse!
library(ggplot2)
library(dplyr)
install.packages("tidytext")
# text mining library
library(tidytext)
# plotting packages
library(igraph)
install.packages("ggraph")
library(ggraph)
install.packages("stringr")
library(stringr)
library("wordcloud")

# remove http elements manually
Vax$stripped_text <- gsub("http.*","",  Vax$Full.Text)
Vax$stripped_text <- gsub("https.*","", Vax$stripped_text)

Vax$stripped_text = gsub("&amp", "", Vax$stripped_text)
Vax$stripped_text = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", Vax$stripped_text)
Vax$stripped_text = gsub("@\\w+", "", Vax$stripped_text)
Vax$stripped_text = gsub("[[:punct:]]", "", Vax$stripped_text)
Vax$stripped_text = gsub("[[:digit:]]", "", Vax$stripped_text)
Vax$stripped_text = gsub("http\\w+", "", Vax$stripped_text)
Vax$stripped_text = gsub("[ \t]{2,}", "", Vax$stripped_text)
Vax$stripped_text = gsub("^\\s+|\\s+$", "", Vax$stripped_text) 

#get rid of unnecessary spaces
Vax$stripped_text <- str_replace_all(Vax$stripped_text," "," ")
# Get rid of URLs
Vax$stripped_text <- str_replace_all(Vax$stripped_text, "http://t.co/[a-z,A-Z,0-9]*{8}","")
# Take out retweet header, there is only one
Vax$stripped_text <- str_replace(Vax$stripped_text,"RT @[a-z,A-Z]*: ","")
# Get rid of hashtags
Vax$stripped_text <- str_replace_all(Vax$stripped_text,"#[a-z,A-Z]*","")
# Get rid of references to other screennames
Vax$stripped_text <- str_replace_all(Vax$stripped_text,"@[a-z,A-Z]*","")   
Vax$stripped_text <- tolower(Vax$stripped_text)

unnest_tokens(Vax$stripped_text, input = text, output = word, format="text", tokens="word", drop = TRUE, to_lower = TRUE)

Vax_tweets_clean <- Vax %>%
  dplyr::select(stripped_text) %>%
  unnest_tokens(word, stripped_text)

# save data for python script to get the word frequency for tweets
df=data.frame(Vax$stripped_text)
write.csv(df,"r.csv", row.names = FALSE)

# Get data with word frequency from python
Vax=read.csv("Tweetswords.csv")
VaxC=Vax[ -c(1,2) ]

#install.packages("caret")
library(caret)



#install.packages("cluster")
#install.packages("factoextra")
library(cluster)
library(factoextra)

# 100 iterations of k
kmax= 100
# wss for Kmeans clustering
fviz_nbclust(VaxC, FUN = kmeans, method = "wss", k.max = kmax)

# 100 iterations of k
kmax= 20
# wss for Kmeans clustering
fviz_nbclust(VaxC, FUN = kmeans, method = "wss", k.max = kmax)

# Kmeans Clustering
c=11
c=12
KmeansClustering = kmeans(VaxC, centers = c)
KM_cluster1=subset(VaxC,KmeansClustering$cluster==1)
KM_cluster2=subset(VaxC,KmeansClustering$cluster==2)
KM_cluster3=subset(VaxC,KmeansClustering$cluster==3)
KM_cluster4=subset(VaxC,KmeansClustering$cluster==4)
KM_cluster5=subset(VaxC,KmeansClustering$cluster==5)
KM_cluster6=subset(VaxC,KmeansClustering$cluster==6)
KM_cluster7=subset(VaxC,KmeansClustering$cluster==7)
KM_cluster8=subset(VaxC,KmeansClustering$cluster==8)
KM_cluster9=subset(VaxC,KmeansClustering$cluster==9)
KM_cluster10=subset(VaxC,KmeansClustering$cluster==10)
KM_cluster11=subset(VaxC,KmeansClustering$cluster==11)
KM_cluster12=subset(VaxC,KmeansClustering$cluster==12)

# Cluster 1
m <- as.matrix(KM_cluster1)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:20], col='tan', las=2, main = 'KM Cluster 1')
#c11: Civil liberties/ Freedom
#c12: Developmental disabilities

# Cluster 2
m <- as.matrix(KM_cluster2)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:30], col='tan', las=2, main = 'KM Cluster 2')
#c11: Developmental disabilities
#c12: Government and policies /Conspiracy Theory

# Cluster 3
m <- as.matrix(KM_cluster3)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:20], col='tan', las=2, main = 'KM Cluster 3')
#c11: Conspiracy Theory
#c12: Conspiracy Theory / Government and policies

# Cluster 4
m <- as.matrix(KM_cluster4)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:20], col='tan', las=2, main = 'KM Cluster 4')
# c11: Civil liberties/ Freedom
# c12: #Effectiveness and Efficacy

# Cluster 5
m <- as.matrix(KM_cluster5)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:20], col='tan', las=2, main = 'KM Cluster 5')
# c11: Conspiracy Theory
# c12: Developmental Disabilities

# Cluster 6
m <- as.matrix(KM_cluster6)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:20], col='tan', las=2, main = 'KM Cluster 6')
#Pharma industry

# Cluster 7
m <- as.matrix(KM_cluster7)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:25], col='tan', las=2, main = 'KM Cluster 7')
#c11:Effectiveness and Efficacy
#c12:Conspiracy Theory

# Cluster 8
m <- as.matrix(KM_cluster8)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:40], col='tan', las=2, main = 'KM Cluster 8')
#c11: Government and policies
#c12: Government and policies / Conspiracy Theory

# Cluster 9
m <- as.matrix(KM_cluster9)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:50], col='tan', las=2, main = 'KM Cluster 9')
#Government and policies

# Cluster 10
m <- as.matrix(KM_cluster10)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:30], col='tan', las=2, main = 'KM Cluster 10')
#C11: Effectiveness and Efficacy
#C12: Civil liberties/ Freedom

# Cluster 11
m <- as.matrix(KM_cluster11)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:25], col='tan', las=2, main = 'KM Cluster 11')
#C11: Conspiracy Theory
#C12: Civil liberties/ Freedom // Conspiracy Theory

# Cluster 12
m <- as.matrix(KM_cluster12)
v <- sort(colSums(m), decreasing = TRUE)
barplot(v[1:25], col='tan', las=2, main = 'KM Cluster 12')
#C12: Conspiracy Theory