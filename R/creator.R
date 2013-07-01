myLDA_class <- setRefClass("myLDA_class",
                           fields=list(
                             corpus = "ANY",
                             url = "character",
                             title ="character",
                             w = "list",  # list of length d which contains the words for each doc as a list
                             w_num = "list", # list of length d which contains the word indices instead of the actual words
                             z = "list", # list of the hidden topic labels for each word in each doc
                             z_res = "list", 
                             z_list = "list", # list of sampled z's as returned from Gibbs sampler
                             alpha = "numeric", # hyper-parameter for Dirichlet distr of theta
                             beta = "numeric", # hyper-parameter of Dirichlet distr of phi
                             K = "numeric",   # K number of Topics 
                             W = "numeric",  # W number of unique Words
                             D = "numeric", # D number of Documents
                             Vocabulary = "character", # Character Vector of all unique words
                             nw = "matrix", # nw_ij number of word i assigned to topic j
                             nd = "matrix", # nd_dj number of times topic j appears in doc d
                             nw_sum = "numeric", # nw_sum_j total number of words assigned to topic j
                             nd_sum = "numeric", # doc length of doc d,
                             stop_de_path = "character", # path to german stopwords
                             stop_en_path = "character" # path to english stopwords
                             
                           ),
                           methods=list(																												
                             initialize = function(corpus, alpha, beta, K)
                             {
                               
                               .self$corpus <- corpus$content
                               .self$url <- corpus$url
                               .self$title <- corpus$title
                               .self$K <- K
                               .self$alpha <- alpha;
                               .self$beta <- beta;
                               .self$stop_de_path <- system.file("german.stop",package="myLDA2")
                               .self$stop_en_path <- system.file("english.stop",package="myLDA2")
                               
                             }
                           )
)

myLDA <- function(artList, alpha=NULL, beta=0.1, K=20)
{
  
  if (is.null(alpha)) alpha <- 50 / K	
  intermediate = new("myLDA_class", corpus=artList, alpha=alpha, beta=beta, K=K);
  lda <- new(LDA,intermediate)
  return(lda)
}