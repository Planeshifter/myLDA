findPhiMAP <- function(LDAobj)
  {
  D <- LDAobj$D
  K <- LDAobj$K
  W <- LDAobj$W
  n <- LDAobj$n_wd
  beta <- LDAobj$beta
  
  posterior_density <- function(x) 
    {
      psi <- matrix(x,nrow=K,ncol=W-1)
      psi <- cbind(0,psi)
      
      phi <- matrix(0, nrow = K, ncol = W)
      for (z in 1:K) {
        phi[z, ] = exp(psi[z,])
        phi[z, ] = phi[z, ]/sum(phi[z, ])
      }
      print(rowSums(phi))
      lPhi <- log(phi)
      d_sum = 0
      for (d in 1:D) {
        k_sum = 0
        k_sum_vec <- rep(0,times=K)
        for (k in 1:K)
        {
          k_sum_vec[k] <- lPhi[z, ] %*% n[ ,d]
        }
        b = max(k_sum_vec);
        for (k in 1:K)
        {
          k_sum = k_sum + exp(k_sum_vec[k]-b);
        }
        
        d_sum = d_sum + log(k_sum)
      }
      prior_sum = sum(lPhi)
      res = - (d_sum + beta * prior_sum)
      return(res)
    }
  
  psiInit <- rnorm(mean=5,sd=3,n=W*K-K)
  require(nloptwrap)
  sol <- newuoa(x0=psiInit, fn=posterior_density) 
  return(sol)
  }

