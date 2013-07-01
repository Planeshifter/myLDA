getMDS <- function(LDAobj)
{
  require(vegan)
  D <- LDAobj$D
  phi <- LDAobj$phi_avg
  
  TermMatrix <- LDAobj$Terms(20)
  K <- nrow(TermMatrix)
  W <- ncol(TermMatrix)
  
  ret <- list()
  # Hellinger distances between the rows
  phiH <- decostand(phi, method = "hellinger")
  d <- dist(phiH)
  fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
  
  # calculate scale factor
  points = (fit$points-min(fit$points))/(max(fit$points-min(fit$points)))
  
  x=points[,1]
  y=points[,2] # get topic coordinates 
  topic.docs <- table(as.factor(temp$Topics(1)))
  
  for (k in 1:K)
  {
  topic = list()
  topic$x <- x[k]
  topic$y <- y[k]
  topic$no_docs <- as.numeric(topic.docs[k]) 
  topic$words <- list()
    for(w in 1:W)
    {
    list_elem <- list()
    list_elem$prob <- phi[k,w]
    list_elem$word <- TermMatrix[k,w]
    topic$words[[length(topic$words)+1]] <- list_elem
    }
  
  top_assignments <- LDAobj$Topics(1)
  
  title <- LDAobj$title
  url <- LDAobj$url
  article_preview <- LDAobj$article_preview
  
  title_in_k <- title[top_assignments==k]
  url_in_k <- url[top_assignments==k]
  preview_in_k <- article_preview[top_assignments==k]
  
  topic$docs <- list()
  for (d in 1:length(title_in_k))
    {
    doc_elem <- list()
    doc_elem$url = url_in_k[d]
    doc_elem$title = title_in_k[d]
    doc_elem$preview = preview_in_k[d]
    topic$docs[[length(topic$docs)+1]] <- doc_elem
    }
  ret[[length(ret)+1]] <- topic
  }
  return(ret)  
}



getMDSdataNicholls <- function(TermMatrix,phi)
{
  require(vegan)
  K <- nrow(TermMatrix)
  W <- ncol(TermMatrix)
  ret <- list()
  # Hellinger distances between the rows:
  phiH <- decostand(phi, method = "hellinger")
  d <- dist(phiH)
  fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
  topic_coordinates <- data.frame(id=1:K,x=fit$points[,1],y=fit$points[,2]) # get topic coordinates 
  words <<- unique(as.vector(TermMatrix))
  words_df <- data.frame(id=rep(1:K,each=W),word=as.vector(TermMatrix))
  merged_df <- merge(words_df,topic_coordinates,by="id")
  for (i in 1:length(words))
    {
    list_elem <- list()
    list_elem$id <- merged_df$id[merged_df$word==words[i]]
    list_elem$prob <- mean(phi[list_elem$id,Vocabulary==words[i]])
    list_elem$x <- mean(merged_df$x[merged_df$word==words[i]])
    list_elem$y <- mean(merged_df$y[merged_df$word==words[i]])
    list_elem$word <- words[i]
    ret[[length(ret)+1]] <- list_elem
    }
return(ret)  
}

plot_mds <- function(ret)
{
require(ggplot2)
x <- unlist(lapply(ret,function(x) x$x))
y <- unlist(lapply(ret,function(x) x$y))
prob <- unlist(lapply(ret,function(x) x$prob))
word <- unlist(lapply(ret,function(x) x$word))
df <- data.frame(x=x,y=y,label=word,prob=prob)

q <- ggplot(df,aes(x=x,y=y,label=label))+geom_text()
q
}

getDocAssignments <- function()
{
docs <- list()
tops <- temp$Topics(1)
cont <- temp$content_original
for (i in 1:400)
  {
  docs[[i]]$content <- cont[i]
  docs[[i]]$topic < tops[i] 
  }
return(docs)
}
