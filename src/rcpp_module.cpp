#include "RegExp.h"

class LDA {
public:
  LDA(Reference Obj);
  int K; // K: number of topics
  int W; // W: size of Vocabulary
  int D; // D: number of documents
  double alpha; // hyper-parameter for Dirichlet prior on theta
  double beta; //  hyper-parameter for Dirichlet prior on phi
  vector<string> content; // vector of article body texts
  vector<string> article_preview; 
  vector<string> url; // vector of article links
  vector<string> title; // vector of article title
  vector< vector<int> >  w_num;
  vector< vector<string> > w;
  vector< vector<int> >  z;
  vector<int> nd_sum;
  vector<int> nw_sum;
  arma::Mat<int> nd;
  arma::Mat<int> nw;
  NumericMatrix phi_avg;
  arma::mat PhiProdMat;
  arma::mat n_wd;
  vector< vector < vector<int> > > z_list;
  vector< NumericMatrix > phi_list;
  vector< NumericMatrix > theta_list;
  NumericMatrix theta_avg;
  vector<string> Vocabulary; // vector storing all (unique) words of vocabulary
  boost::mt19937 rng; // seed for random sampling
  vector<string> stopwords_en;
  vector<string> stopwords_de;
  string stop_de_path;
  string stop_en_path;
  
  enum language 
  {
  ENGLISH = 0, 
  GERMAN = 1
  };
  vector<std::string> read_stopwords(int lang);
  vector<std::string> eliminate_stopwords(vector<std::string> input);
  
  void PreProcessText();
  void ExtractWords();
  void ConstructVocabulary();
  void InitSampling();
  vector<string> remove_empty_articles(vector<string> input, vector<int> empty);
  vector<int> mark_empty_articles(vector<string> input);
  int SampleZ();
  
  void collapsedGibbs(int iter, int burnin, int thin);
  void NichollsMH(int iter, int burnin, int thin);
  void LangevinMHSampling(int iter, int burnin, int thin);
  arma::mat ExponentiadedGradient();

  NumericVector DocTopics(int d, int k); 
  NumericMatrix Topics(int k);
  CharacterVector TopicTerms(int k, int no);
  CharacterMatrix Terms(int k);
  arma::rowvec rDirichlet(arma::rowvec param, int length);
  arma::rowvec rDirichlet2(arma::rowvec param, int length);
  arma::mat DrawFromProposal(arma::mat phi_current);
  arma::mat InitPhiMat();
  List getPhiList();
  List getZList();
  
  double PhiDensity2(NumericMatrix phi);
  double PhiDensity(arma::mat phi);
  
  arma::mat getPhiGradient(arma::mat phi);
  double getPhiGradient_elem(int z, int w, arma::mat phi);
  
  double rgamma_cpp(double alpha);
  double rbeta_cpp(double shape1, double shape2);
  double rnorm_cpp(double mean, double sd);

  double LogPhiProd(arma::mat phi); 
  vector<double> LogPhiProd_vec(arma::mat phi);
  arma::mat DrawLangevinProposal(arma::mat phi_current);
  double EvalLangevinProposal(arma::mat PhiFrom, arma::mat PhiTo);
  arma::mat ProjectProposalToSimplex(arma::mat PhiProposal);

  
private:
  vector< vector<int> > CreateIntMatrix(List input);
  double sigma; // for Langevin sampler
  
  NumericMatrix get_phis();
  NumericMatrix get_thetas();
  NumericMatrix MatrixToR(NumericMatrix input);
  NumericMatrix avgMatrix(NumericMatrix A, NumericMatrix B, int weight);
  arma::mat getTDM(int W, int D, vector< vector<int> > w_num);
  
  double ProposalDensity(arma::mat phi);
  double ArrayMax(double array[], int numElements);
  double ArrayMin(double array[], int numElements);
  
};

LDA::LDA(Reference Obj)
  {  
  content = as<vector<string> >(Obj.field("corpus"));
  vector<int> empty_articles = mark_empty_articles(content);
  content = remove_empty_articles(content, empty_articles);
  D = content.size(); 

  for (int d=0;d<D;d++)
    {
    string current_preview = content[d].substr(0,250);
    article_preview.push_back(current_preview);   
    }
    
  url = as<vector<string> >(Obj.field("url"));
  url = remove_empty_articles(url, empty_articles);
  
  title = as<vector<string> >(Obj.field("title"));
  title = remove_empty_articles(title, empty_articles);
  
  K = as<int>(Obj.field("K"));
  
  
  stop_de_path = as<string>(Obj.field("stop_de_path"));
  stop_en_path = as<string>(Obj.field("stop_en_path"));
  stopwords_en =  read_stopwords(ENGLISH);
  stopwords_de =  read_stopwords(GERMAN);
  
  alpha = Obj.field("alpha");
  beta = Obj.field("beta");
  sigma = 0.00001;
  
  PreProcessText();
  ExtractWords();
  ConstructVocabulary();
  W = Vocabulary.size();
  InitSampling();
  n_wd = getTDM(W, D, w_num);
  
  };
  
void LDA::PreProcessText()
  {
  for (int d=0;d<D;d++)
    {
    boost::algorithm::to_lower(content[d]);
    content[d] = remove_numbers(content[d]);
    content[d] = remove_punctuation(content[d]);
    content[d] = remove_special_characters(content[d]);
    content[d] = remove_whitespace(content[d]);
    }
  };

void LDA::ExtractWords()
  {
  for (int d=0;d<D;d++)
    {
    vector<string> wd = isolate_words(content[d]);
    wd = eliminate_empty_words(wd);
    wd = eliminate_stopwords(wd);
    w.push_back(wd);
    }
  };

void LDA::ConstructVocabulary()
  {
  int pos = 0;
  vector<vector<int> > w_num_temp(D);
  for (int d=0;d<D;d++)
    {
      int nd = w[d].size();
      for (int i=0;i<nd;i++)
      {
      string target = w[d][i]; 
      vector<string>::iterator it = std::find(Vocabulary.begin(), Vocabulary.end(), target);
      bool isPresent = (it != Vocabulary.end());
      if (isPresent==false) 
        {
        Vocabulary.push_back(target);
        w_num_temp[d].push_back(pos);
        pos += 1;
        }
      else
        {
        int index = std::distance(Vocabulary.begin(), it);
        w_num_temp[d].push_back(index); 
        }
      }
    }
  w_num = w_num_temp;
  }
  
vector<int> LDA::mark_empty_articles(vector<string> input)
  {
  int len = input.size();
  int empty_counter = 0;
  vector<int> nonempty_articles;
  for (int i = 0; i < len; i++)
    {
    if (input[i] == "") empty_counter++;
    else nonempty_articles.push_back(i);    
    }
  Rcout << "Articles with empty content were marked. There are" << empty_counter << " empty articles.";
  return nonempty_articles;
  }
  
vector<string> LDA::remove_empty_articles(vector<string> input, vector<int> nonempty)
  {
  int len = input.size();
  int empty_counter = 0;
  vector<string> output;
  
  for(std::vector<int>::iterator it = nonempty.begin(); it != nonempty.end(); ++it) 
    {
    Rcout << *it; 
    output.push_back(input[*it]);
    }
  return output;
  }

int LDA::SampleZ()  
  {
  boost::uniform_int<> dist(0, K-1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > sample(rng, dist);
  return sample();
  }
  
void LDA::InitSampling()
  {
  vector<int> nd_sum_temp(D,0);
  vector<int> nw_sum_temp(K,0);
  arma::Mat<int> nw_temp(W,K);
  nw_temp.zeros();
  arma::Mat<int> nd_temp(D,K); 
  nd_temp.zeros();
  vector< vector<int> >  z_temp(D);
					
	for (int d=0; d<D; d++)
		{		
      int len = w_num[d].size();
      
      // initialize random topics
      for (int i=0; i<len;i++)
        {
        int sampled_z = SampleZ();
        z_temp[d].push_back(sampled_z); 
        // Rcout << z[d][i] << "-";  
        }
        
			nd_sum_temp[d] = len;
			
			for (int i=0; i<len; i++)
			{
				int     wtemp = w_num[d][i];
				int     topic = z_temp[d][i];
				
				// number of instances of word i assigned to topic j
				nw_temp(wtemp,topic) = nw_temp(wtemp,topic) + 1;
				// number of words in document i assigned to topic j
				nd_temp(d,topic)  = nd_temp(d,topic) + 1;
				//  total number of words assigned to topic j           
				nw_sum_temp[topic] = nw_sum_temp[topic] + 1;
			}
		}
    
  nd_sum = nd_sum_temp;
  nw_sum = nw_sum_temp; 
  nd = nd_temp;
  nw = nw_temp;
  z = z_temp;
  }
  
vector<std::string> LDA::read_stopwords(int lang)
  {
  vector<std::string> output;
  string line;
  
  const char * stop_en_char = stop_en_path.c_str();
  const char * stop_de_char = stop_de_path.c_str();
    
  ifstream myfile(stop_en_char);
  ifstream myfile2(stop_de_char);
  
  switch(lang){
  case ENGLISH:   
  // Rcout << "Das ist Englisch";
    if (myfile.is_open())
    {
      while ( myfile.good() )
      {
        getline (myfile,line);
        output.push_back(line);
      }
      myfile.close();
    }    
    
  break;
  case GERMAN:
      if (myfile2.is_open())
    {
      while ( myfile2.good() )
      {
        getline (myfile2,line);
        output.push_back(line);
      }
      myfile2.close();
    }    
  
  break; 
  default: 
  
  break;
  }  
  return output;  
  }

vector<string> LDA::eliminate_stopwords(vector<string> input)
  {
  // Rcout << "Laenge des Inputs" << input.size();
  vector<string> output;
  
  int input_length = input.size();
  for (int i = 0; i<input_length; i++)
    {
      
     string target = input[i];
     vector<string>::iterator it = std::find(stopwords_en.begin(), stopwords_en.end(), target);
     if (it == stopwords_en.end()) 
     {
     vector<string>::iterator it_de = std::find(stopwords_de.begin(), stopwords_de.end(), target);
     if (it_de == stopwords_de.end()) output.push_back(target);
     }
          
    }
    
  // Rcout << "Lange des Output" << output.size();
  return output;  
  }

List LDA::getPhiList()
  {
  int iter = phi_list.size();
  List ret(iter);
  
  for (int i = 0; i<iter; i++)
    {
    ret[i] = phi_list[i];
    }
  return ret;
  }

List LDA::getZList()
  {
  int length = z_list.size();
  List ret(length);
  
  for (int i = 0; i<length; i++)
    {
    ret[i] = wrap(z_list[i]);
    }
  return ret;
  }

vector< vector<int> > LDA::CreateIntMatrix(List input)
  {
  int inputLength = input.size();
  vector< vector<int> > output;

  for(int i=0; i<inputLength; i++) {
            vector<int> test = as<vector<int> > (input[i]);
            output.push_back(test);
            }

  return output;
  }

NumericVector LDA::DocTopics(int d, int k)
  {
  vector<double> d_theta(K);
  NumericVector d_theta_R = theta_avg(d,_);
  d_theta = as<vector<double> > (d_theta_R);
  NumericVector ret_vector(k);

  for (int i=0;i<k;i++)
    {
    std::vector<double>::iterator result;
    result = std::max_element(d_theta.begin(),d_theta.end());
    int biggest_id = std::distance(d_theta.begin(), result);
    ret_vector[i] = biggest_id;
    d_theta[biggest_id] = 0;
    }

  return ret_vector;
  }

NumericMatrix LDA::Topics(int k)
  {
  NumericMatrix ret(D,k);
  for (int i = 0; i<D; i++)
    {
    NumericVector temp = DocTopics(i,k);
    ret(i,_) = temp;
    }
  ret = MatrixToR(ret);
  return ret;
  }

CharacterVector LDA::TopicTerms(int k, int no)
  {
  vector<double> k_phi(W);
  NumericVector k_phi_R = phi_avg(k,_);
  k_phi = as<vector<double> > (k_phi_R);
  NumericVector ret_vector(no);

  for (int i=0;i<no;i++)
    {
    std::vector<double>::iterator result;
    result = std::max_element(k_phi.begin(),k_phi.end());
    int biggest_id = std::distance(k_phi.begin(), result);
    ret_vector[i] = biggest_id;
    k_phi[biggest_id] = 0;
    }

  CharacterVector ret_char_vector(no);

  for (int i=0;i<no;i++)
    {
    ret_char_vector[i] = Vocabulary[ret_vector[i]];
    }

  return ret_char_vector;
  }

CharacterMatrix LDA::Terms(int k)
  {
  CharacterMatrix ret(K,k);
  for (int i = 0; i < K; i++)
    {
    CharacterVector temp = TopicTerms(i,k);
    ret(i,_) =  temp;
    }
  return ret;
  }

NumericMatrix LDA::MatrixToR(NumericMatrix input)
  {
  int n = input.nrow(), k = input.ncol();
  NumericMatrix output(n,k);
  for (int i = 0; i<n; i++)
    {
    for (int j = 0; j<k; j++)
      {
      output(i,j) = input(i,j) + 1;
      }
    }
  return output;
  }

void LDA::collapsedGibbs(int iter, int burnin, int thin)
  {

  double Kd = (double) K;
  double Wd = (double) W;
  double W_Beta  = Wd * beta;
  double K_Alpha = Kd * alpha;

   for (int i = 0; i < iter; ++i)
          {
            for (int d = 0; d < D; ++d)
            {
              
              for (int w = 0; w < nd_sum[d]; ++w)
              {
              int word = w_num[d][w];
              int topic = z[d][w];

              nw(word,topic) -= 1;
              nd(d,topic) -= 1;
              nw_sum[topic] -= 1;
              nd_sum[d] -=  1;

              vector<double>  prob(K);

              for(int j=0; j<K; j++)
                {
                double nw_ij = nw(word,j);
                double nd_dj = nd(d,j);
                prob[j] = (nw_ij + beta) / (nw_sum[j] + W_Beta) *
                                (nd_dj + alpha) / (nd_sum[d] + K_Alpha);
                }

              for (int r = 1; r < K; ++r)
              {
              prob[r] = prob[r] + prob[r - 1];
              }

              double u  = prob[K-1] * rand() / double(RAND_MAX);

              int new_topic = 0; // set up new topic

              for (int nt = 0 ; nt < K; ++nt)
                {
                if (prob[nt] > u)
                  {
                  new_topic = nt;
                  break;
                  }
                }
                
               //  assign new z_i to counts
                 nw(word,new_topic) +=  1;
                 nd(d,new_topic) += 1;
                 nw_sum[new_topic] += 1;
                 nd_sum[d] += 1;

                 z[d][w] = new_topic;

              }

            }



          if (i % thin == 0 && i > burnin)
            {
              z_list.push_back(z);

              NumericMatrix current_phi = get_phis();
              NumericMatrix current_theta = get_thetas();
              phi_list.push_back(current_phi);
              theta_list.push_back(current_theta);

              if(phi_list.size()==1) phi_avg = current_phi;
              else phi_avg =  avgMatrix(phi_avg, current_phi, phi_list.size());

              if(theta_list.size()==1) theta_avg = current_theta;
              else theta_avg =  avgMatrix(theta_avg, current_theta, theta_list.size());

            }

          }
  }

NumericMatrix LDA::avgMatrix(NumericMatrix A, NumericMatrix B, int weight)
  {
  int nrow = A.nrow();
  int ncol = A.ncol();
  NumericMatrix C(nrow,ncol);

  float wf = (float) weight;
  float propA = (wf-1) / wf;
  float propB = 1 / wf;

  for (int i=0; i<nrow;i++)
  {
    for (int j=0; j<ncol;j++)
    {
    C(i,j) =  propA * A(i,j) + propB * B(i,j);
    }
  }

  return C;
  }


NumericMatrix LDA::get_phis()
    {

      NumericMatrix phi(K,W);

       for (int k = 0; k < K; k++) {
         for (int w = 0; w < W; w++) {
           phi(k,w) = (nw(w,k) + beta) / (nw_sum[k] + W * beta);
         }
       }

      return phi;
    }

NumericMatrix LDA::get_thetas()
   {

    NumericMatrix theta(D,K);

     for (int d = 0; d<D; d++) {
       for (int k = 0; k<K; k++) {
         theta(d,k) = (nd(d,k) + alpha) / (nd_sum[d] + K * alpha);
       }
     }
   return theta;
   }

arma::mat LDA::getTDM(int W, int D, vector< vector<int> > w_num) 
  {
  arma::mat tdm(W,D);
  for (int d=0; d<D; ++d)
   for (int w=0; w<W; ++w)
    {
    int freq = 0;
    vector<int> current_w = w_num[d];
    int wlen = current_w.size();
    for (int l=0; l<wlen; ++l)
      {
      if(current_w[l] == w) freq += 1;
      }
  
     tdm(w,d) = freq;
    }
  return tdm;
  }

// using R: (unfortunately too slow to call R and convert objects back)
//NumericMatrix LDA::DrawFromProposal()
//  {
//    Environment MCMCpack("package:MCMCpack");
//    Function rdirichlet = MCMCpack["rdirichlet"];
//    return rdirichlet(K,rep(beta,W));
//  }


double LDA::rbeta_cpp(double shape1, double shape2)
  {
    double u  = rand() / double(RAND_MAX);
    beta_distribution<> beta_dist(shape1, shape2);
    return quantile(beta_dist, u);  
  }

double LDA::rgamma_cpp(double alpha)
  {
    boost::gamma_distribution<> dgamma(alpha);
    boost::variate_generator<boost::mt19937&,boost::gamma_distribution<> > ret_gamma( rng, dgamma);
    return ret_gamma();
  }
  
double LDA::rnorm_cpp(double mean, double sd)
  {
  boost::normal_distribution<> nd(mean, sd);
  boost::variate_generator<boost::mt19937&, 
                           boost::normal_distribution<> > ret_norm(rng, nd);
  return ret_norm();  
  }

arma::rowvec LDA::rDirichlet(arma::rowvec param, int length)
  {
  rowvec ret(length);
  for (int l = 0; l<length; l++)
    {
    double beta = param[l];
    ret[l] = rgamma_cpp(beta);
    }
  ret = ret / sum(ret);
  return ret;
  }
  
arma::rowvec LDA::rDirichlet2(arma::rowvec param, int length)
  {
  vector<double> ret;
  param *= 10000;
  vector<double> param_vec = conv_to<vector<double> >::from(param);
  Rcout << param_vec[0] << "-";
  int len = length - 1;
  
  double paramSum = std::accumulate(param_vec.begin()+1,param_vec.end(),(double)0);
  Rcout << paramSum;
  ret.push_back(rbeta_cpp(param_vec[0], paramSum));
  for (int i=1; i<len;i++)
    {
    double paramSum = std::accumulate(param_vec.begin()+i+1,param_vec.end(),(double)0); 
    double phi = rbeta_cpp(param_vec[i], paramSum);
    double sumRet = std::accumulate(ret.begin(),ret.end(),(double)0);  
    ret.push_back((1-sumRet) * phi);
    }   
  double sumRet = std::accumulate(ret.begin(),ret.end(),(double)0); 
  ret.push_back(1-sumRet);
  return ret;
  }  
  

mat LDA::DrawFromProposal(arma::mat phi_current)
    {
    arma::mat phi_sampled(K,W);
    for (int k=0;k<K;k++)
      {
      arma::rowvec phi_current_row = phi_current.row(k);
      arma::rowvec new_row = rDirichlet2(phi_current_row, W);
      // Rcout << new_row;
      phi_sampled.row(k) = new_row;
      }
    return phi_sampled;
    }
    
mat LDA::InitPhiMat()
    {
    arma::mat phi(K,W);
  
    for (int k=0; k<K; k++)
      {
        for (int w=0; w<W; w++)
        {
        phi(k,w) = beta / (W*beta);  
        }
      }
    return phi;
    }    
     

void LDA::NichollsMH(int iter, int burnin, int thin)
  {

    arma::mat phi_current = InitPhiMat();
    
    for (int t=1;t<iter;t++)
    {

    // Metropolis-Hastings Algorithm:
    // 1. draw from proposal density:
    arma::mat hyperParams = beta + 0.1 * (phi_current - beta);
    arma::mat phi_new = DrawFromProposal(hyperParams);

    // 2. Calculate acceptance probability
    double pi_new = PhiDensity(phi_new);
    double pi_old = PhiDensity(phi_current);
    double q_new = ProposalDensity(phi_new);
    double q_old = ProposalDensity(phi_current);

    double acceptanceMH = exp(pi_new + q_old - pi_old - q_new);
    double alphaMH = min((double)1,acceptanceMH);
    Rcout << "Acceptance Prob:" << alphaMH;

    // draw U[0,1] random variable
    double u  = rand() / double(RAND_MAX);
    if (u<=alphaMH) phi_current = phi_new;
    else phi_current = phi_current;

    if (t % thin == 0 && t > burnin) {
     NumericMatrix phi_add = wrap(phi_current);
     phi_list.push_back(phi_add);
      if(phi_list.size()==1) phi_avg = phi_add;
              else phi_avg =  avgMatrix(phi_avg, phi_add, phi_list.size());
    };

    // Rcout << pi_new;
    // Rcout << pi_old;
    // Rcout << q_new;
    // Rcout << q_old;


    }

  }
  
double LDA::LogPhiProd(arma::mat phi)
  {
  arma::mat logPhi = log(phi);
  double sumLik_vec[K];
  double logPhiProd = 0;
    
    for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       sumLik_vec[k] = dot(logPhi_k,nd);
       }
     double b = ArrayMax(sumLik_vec,K);
     
     for (int k=0; k<K; k++)
       {
       sumLik += exp(sumLik_vec[k]-b);
       }
     
     logPhiProd += b + log(sumLik);
     }  
     
  return logPhiProd;  
  }
  
vector<double> LDA::LogPhiProd_vec(arma::mat phi)
  {   
  vector<double> ret_vec;
  arma::mat logPhi = log(phi);
  arma::mat PhiProdMat_Pointer(K,D);
  double sumLik_vec[K];
    
    for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       sumLik_vec[k] = dot(logPhi_k,nd);    
       PhiProdMat_Pointer(k,d) = sumLik_vec[k];
       }
     double b = ArrayMax(sumLik_vec,K);
     
     for (int k=0; k<K; k++)
       {
       sumLik += exp(sumLik_vec[k]-b);
       }
     
     double ret_vec_d = b + log(sumLik);
     ret_vec.push_back(ret_vec_d);
     }  
  PhiProdMat = PhiProdMat_Pointer;
  return ret_vec;  
  }
  
// function adapted from Yunmei Chen and Xiaojing Ye (2011)

vector<double> ProjectOntoSimplex (vector<double> y)
  {
  int m = y.size();
  bool bget = false;

  vector<double> s = y;
  std::sort(s.rbegin(), s.rend());
  
  double tmpsum = 0;
  double tmax = 0;
  
  for (int i = 0; i<m-1; i++)
    {
    tmpsum = tmpsum + s[i];
    tmax = (tmpsum - 1)/(i+1);
    if (tmax >= s[i+1]) 
      {
      bget = true;
      break;
      }
    }
    
   if (bget==false) 
     {
     tmax = (tmpsum + s[m-1] - 1)/m;
     }

   vector<double> x;
   for (int j = 0; j<m;j++)
     {
     double elem1 = y[j] - tmax;
     double ret = max(elem1,0.0);
     x.push_back(ret);
     }
   
   return x;    
   }
     
arma::mat LDA::getPhiGradient(arma::mat phi)
  {
    arma::mat gradient(K,W);
    
    for (int z=0;z<K;z++)
      {
      for(int w=0;w<W;w++)
        {
        gradient(z,w) = getPhiGradient_elem(z, w, phi);  
        }
      }
       
    return gradient;   
  }  
  
double LDA::getPhiGradient_elem(int z, int w, arma::mat phi) 
  {
  arma::mat logPhi = log(phi*1000)-log(1000);
  vector<double> denom_vec = LogPhiProd_vec(phi);
  double dSum = 0;  
        
        for (int d = 0; d<D;d++)
          {  
          double nwd = n_wd(w,d);
          if (nwd==0) dSum += 0;
          else 
            {
            // Rcout << "nwd:" << nwd;
            arma::colvec nd = n_wd.col(d);
            arma::rowvec logPhi_k = logPhi.row(z);        
          
            double dotProd = PhiProdMat(z,d);
            // Rcout << "Dot Product: " << dotProd;
            // Rcout <<  nd[w]*logPhi_k[w];
            double Numerator = log(nwd) + (nwd - 1)*logPhi(z,w) + dotProd - nd[w]*logPhi_k[w]; 
            // Rcout << Numerator;
            double Denominator = denom_vec[d];
            // Rcout << "Denominator:" << Denominator;
            dSum += Numerator - Denominator;
            }
          }
  // Rcout << "dSum:" << dSum; 
  double ret = exp(dSum) + (beta - 1) / phi(z,w);
  // Rcout << "Prior Influence:" <<  (beta - 1) / phi(z,w);     
  return ret;
  }

arma::mat LDA::DrawLangevinProposal(arma::mat phi_current)  
  {
  arma::mat PhiProposal(K,W);
  arma::mat PhiGradient = getPhiGradient(phi_current);
  for (int z=0; z<K; z++)
    {
    vector<double> prop_vec; 
    for (int w=0; w<W; w++)
      { 
      double error = rnorm_cpp(0,sigma);
      double sigma_squared = pow(sigma,2);
      PhiProposal(z,w) = phi_current(z,w) + 0.5 * sigma_squared * PhiGradient(z,w) + error;
      }
    }   
  return PhiProposal;
  }
  
double LDA::EvalLangevinProposal(arma::mat PhiFrom, arma::mat PhiTo)  
  {
  double sigma_squared = pow(sigma,2);
  double logDensity = 0;
  arma::mat PhiGradient = getPhiGradient(PhiFrom);
  for (int z=0; z<K; z++)
    {
    for (int w=0;w<W;w++)
      {
       double gradient_zw = PhiGradient(z,w);
       double mean = PhiFrom(z,w) + 0.5 * sigma_squared * gradient_zw;
       double PhiMeanDiff = PhiTo(z,w) - mean; 
       logDensity -= (1/(2*sigma_squared))*pow(PhiMeanDiff,2);
      }
    }   
  return logDensity;
  }  
  
arma::mat LDA::ProjectProposalToSimplex(arma::mat PhiProposal)
  {
    for (int z=0; z<K; z++)
    {
    vector<double> prop_vec = conv_to<vector<double> >::from(PhiProposal.row(z));
    vector<double> Phi_proj_vec = ProjectOntoSimplex(prop_vec);
    PhiProposal.row(z) = conv_to<rowvec>::from(Phi_proj_vec); 
    }
  return PhiProposal;
  }
  
void LDA::LangevinMHSampling(int iter, int burnin, int thin)
  {
    arma::mat phi_current = InitPhiMat();
    arma::mat phi_current_projected = ProjectProposalToSimplex(phi_current);
    
    for (int t=1;t<iter;t++)
    {

    // Metropolis Algorithm:
    // 1. draw from Langevin proposal density:
    arma::mat phi_new = DrawLangevinProposal(phi_current_projected);
    arma::mat phi_new_projected = ProjectProposalToSimplex(phi_new);

    // 2. Calculate acceptance probability
    double pi_new = PhiDensity(phi_new_projected);
    double pi_old = PhiDensity(phi_current_projected);
    double q_num = EvalLangevinProposal(phi_new_projected,phi_current);
    double q_denom = EvalLangevinProposal(phi_current_projected,phi_new);
   
    // Rcout << "Pi_new:" << pi_new;
    // Rcout << "Pi_old:" << pi_old;
    // Rcout << "Q_numerator:" << q_num;
    // Rcout << "Q_denominator:" << q_denom;
    
    double acceptanceMH = exp(pi_new + q_num - pi_old - q_denom);
    double alphaMH = min((double)1,acceptanceMH);
    // Rcout << "Acceptance Prob:" << alphaMH;

    // draw U[0,1] random variable
    double u  = rand() / double(RAND_MAX);
    if (u<=alphaMH) 
      {
      phi_current = phi_new;
      phi_current_projected = phi_new_projected;
      }
    else 
      {
      phi_current = phi_current;
      phi_current_projected = phi_current_projected;
      }

    if (t % thin == 0 && t > burnin) {
     NumericMatrix phi_add = wrap(phi_current_projected);
     phi_list.push_back(phi_add);
      if(phi_list.size()==1) phi_avg = phi_add;
              else phi_avg =  avgMatrix(phi_avg, phi_add, phi_list.size());
    };

    }

  }
  

arma::mat LDA::ExponentiadedGradient()
  {
  arma::mat phi_current = InitPhiMat();
  Rcout << phi_current(1,1);
  
  for (int i=0;i<30;i++)  
    {
    phi_current = phi_current + 1e-10;  
    arma::mat PhiGradient = getPhiGradient(phi_current);
    double k = 0.0001;    
    arma::mat ExpGrad = exp(k*PhiGradient);
    arma::mat phi_new = phi_current % ExpGrad;
    // Rcout << phi_current(1,1);
    // Rcout << "\n" << ExpGrad(1,1) << "\n";
    // Rcout << phi_new(1,1);
    
    for (int z=0;z<K;z++)
      {
      double row_sum = norm(phi_new.row(z),2);
      Rcout << row_sum;
      phi_new.row(z) = phi_new.row(z) / row_sum; 
      }
    
    phi_current = phi_new;
    }
    
  return phi_current;
  }
      

double LDA::ProposalDensity(arma::mat phi)
  {
    double logBetaFun = 0;
    double betaSum = 0;
    for (int k=0; k<K;k++)
      {
      for (int w=0;w<W;w++)
        {
        double phi_scalar = phi(k,w);
        logBetaFun += lgamma(phi_scalar);
        betaSum  += phi_scalar;
        }
          
      }
    // double logBetaFun = K*(W*lgamma(beta)-lgamma(W*beta));
    logBetaFun -= lgamma(betaSum);
    
    arma::mat logPhi = log(phi);
    arma::mat temp = logPhi * (beta-1);
    double logPhiSum = accu(temp);

    double logDensity = logPhiSum - logBetaFun;
    return logDensity;
  }

double LDA::PhiDensity(arma::mat phi)
  {
  arma::mat logPhi = log(phi);
  double sumLik_vec[K];
  double logLikelihood = 0;

   for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       sumLik_vec[k] = dot(logPhi_k,nd);
       }
     double b = ArrayMax(sumLik_vec,K);
     
     for (int k=0; k<K; k++)
       {
       sumLik += exp(sumLik_vec[k]-b);
       }
     
     logLikelihood +=  b + log(sumLik);
     }
     
      logLikelihood += D * log(alpha);
      logLikelihood -= D * log(K*alpha);
      // Rcout << "logLikelihood: " << logLikelihood;

      double logBetaFun = K*(lgamma(W*beta)-W*lgamma(beta));
      // Rcout << "logBetaFun: " << logBetaFun;

      double logPhiSum = 0;

      arma::mat temp = logPhi * (beta-1);
      logPhiSum = accu(temp);

      // Rcout << "LogPhiSum: " << logPhiSum;

    double logProb = logLikelihood + logBetaFun + logPhiSum;
    return logProb;
   }

double LDA::PhiDensity2(NumericMatrix phi)
  {
  arma::mat phi2 = as<arma::mat>(phi);
  arma::mat logPhi = log(phi2);
  double logLikelihood_vec[D];
  double logLikelihood = 0;

   for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       double inProd_k = 0;

       for (int w=0; w<W; w++)
       {
    	   inProd_k += logPhi_k[w] * nd[w];
		   }
       // Rcout << inProd_k;
       double sumLik_k = exp(inProd_k) * alpha;
       sumLik += sumLik_k;
       }
     logLikelihood_vec[d] = log(sumLik);
     logLikelihood += logLikelihood_vec[d];
     }

      logLikelihood -= D * log(K*alpha);
      //Rcout << "logLikelihood: " << logLikelihood;

      double logBetaFun = K*(lgamma(W*beta)-W*lgamma(beta));
      //Rcout << "logBetaFun: " << logBetaFun;

      double logPhiSum = 0;

      arma::mat temp = logPhi * (beta-1);
      logPhiSum = accu(temp);

      //Rcout << "LogPhiSum: " << logPhiSum;

    double logProb = logLikelihood + logBetaFun + logPhiSum;
    double Prob = exp(logProb);
    return Prob;
   }


double LDA::ArrayMax(double array[], int numElements)
{
     double max = array[0];       // start with max = first element

     for(int i = 1; i<numElements; i++)
     {
          if(array[i] > max)
                max = array[i];
     }
     return max;                // return highest value in array
}

double LDA::ArrayMin(double array[], int numElements)
{
     double min = array[0];       // start with min = first element

     for(int i = 1; i<numElements; i++)
     {
          if(array[i] < min)
                min = array[i];
     }
     return min;                // return smallest value in array
}

RCPP_MODULE(yada){
class_<LDA>( "LDA" )
.constructor<Reference>()
.field("content",&LDA::content)
.field("url",&LDA::url)
.field("title",&LDA::title)
.field( "n_wd", &LDA::n_wd)
.field( "nd_sum", &LDA::nd_sum)
.field("nd",&LDA::nd)
.field( "nw_sum", &LDA::nw_sum)
.field("nw",&LDA::nw)
.field("K", &LDA::K)
.field("D",&LDA::D)
.field("W",&LDA::W)
.field("alpha",&LDA::alpha)
.field("beta",&LDA::beta)
.field("Vocabulary",&LDA::Vocabulary)
.field("stop_en_path",&LDA::stop_en_path)
.field("phi_avg",&LDA::phi_avg)
.field("theta_avg",&LDA::theta_avg)
.field("PhiProdMat",&LDA::PhiProdMat)
.field("article_preview",&LDA::article_preview)
.method("collapsedGibbs",&LDA::collapsedGibbs)
.method("Topics",&LDA::Topics)
.method("Terms",&LDA::Terms)
//.method("NichollsMH",&LDA::NichollsMH)
//.method("DrawFromProposal",&LDA::DrawFromProposal)
//.method("getPhiList",&LDA::getPhiList)
//.method("getZList",&LDA::getZList)
.method("getPhiGradient",&LDA::getPhiGradient)
.method("getPhiGradient_elem",&LDA::getPhiGradient_elem)
//.method("rgamma_cpp",&LDA::rgamma_cpp)
//.method("rbeta_cpp",&LDA::rbeta_cpp)
.method("InitPhiMat",&LDA::InitPhiMat)
//.method("rDirichlet2",&LDA::rDirichlet2)
.method("LogPhiProd_vec",&LDA::LogPhiProd_vec)
.method("LogPhiProd",&LDA::LogPhiProd)
//.method("DrawLangevinProposal",&LDA::DrawLangevinProposal)
.method("LangevinMHSampling",&LDA::LangevinMHSampling)
.method("ExponentiadedGradient",&LDA::ExponentiadedGradient)
//.method("PhiDensity",&LDA::PhiDensity)
//.method("ProjectProposalToSimplex",&LDA::ProjectProposalToSimplex)
;
}                     

class TwitterNews{
public:
  TwitterNews(Reference Obj);
  int K; // K: number of topics
  int W; // W: size of Vocabulary
  int D; // D: number of documents
  int S; // S: number of news sources
  double alpha; // hyper-parameter for Dirichlet prior on theta
  double beta; //  hyper-parameter for Dirichlet prior on phi
  double tau;  // hyper-parameter for Dirichlet prior on kappa
  double gamma; // hyper-parameter for Gamma prior on lambda (shape)
  double delta; // hyper-parameter for Gamma prior on lambda (rate)
  vector<string> content; // vector of article body texts
  vector<string> article_preview; 
  vector<string> url; // vector of article links
  vector<string> title; // vector of article title
  vector< vector<int> >  w_num;
  vector< vector<string> > w;
  vector<int> z; // topic assignments for each document
  vector<int> s; // newspaper source (int)
  vector<int> c; // number of citations
  vector<string> s_string; // newspaper source (string)
  vector<int> nd_sum;
  vector<int> nk;
  vector<int> ns;
  vector<int> nw_sum;
  arma::Mat<int> nd;
  arma::Mat<int> nw;
  arma::Mat<int> N_dw;
  arma::Mat<int> nsk;
  arma::Mat<int> csk;
  arma::mat PhiProdMat;
  NumericMatrix phi_avg;
  NumericMatrix theta_avg;
  NumericMatrix lambda_avg;
  NumericVector kappa_avg;
  NumericMatrix prob_s;
  arma::mat n_wd;
  arma::mat z_mat;
  vector< vector<int> >  z_list;
  vector< NumericMatrix > phi_list;
  vector< NumericMatrix > theta_list;
  vector<string> Vocabulary; // vector storing all (unique) words of vocabulary
  vector<string> Newspapers; // vector storing all newspapers
  boost::mt19937 rng; // seed for random sampling
  vector<string> stopwords_en;
  vector<string> stopwords_de;
  string stop_de_path;
  string stop_en_path;
  
  enum language 
  {
  ENGLISH = 0, 
  GERMAN = 1
  };
  vector<std::string> read_stopwords(int lang);
  vector<std::string> eliminate_stopwords(vector<std::string> input);
  
  void PreProcessText();
  void ExtractWords();
  void ConstructVocabulary();
  void CreateNewspapers();
  void InitSampling();
  vector<string> remove_empty_articles(vector<string> input, vector<int> empty);
  vector<int> mark_empty_articles(vector<string> input);
  int SampleZ();
  
  void collapsedGibbs(int iter, int burnin, int thin);
  void getParameterEstimates();
  vector<double> get_kappa_estimates();
  arma::mat get_prob_s();
  arma::mat get_lambda_estimates();
  arma::mat get_theta_estimates();
  arma::mat get_phi_estimates();

  CharacterVector TopicTerms(int k, int no);
  CharacterMatrix Terms(int k);
  arma::rowvec rDirichlet(arma::rowvec param, int length);
  arma::rowvec rDirichlet2(arma::rowvec param, int length);
  arma::mat DrawFromProposal(arma::mat phi_current);
  arma::mat InitPhiMat();
  List getPhiList();
  List getZList();
  
  double rgamma_cpp(double alpha);
  double rbeta_cpp(double shape1, double shape2);
  double rnorm_cpp(double mean, double sd);

  double LogPhiProd(arma::mat phi); 
  vector<double> LogPhiProd_vec(arma::mat phi);
  
private:
  vector< vector<int> > CreateIntMatrix(List input);
  double sigma; // for Langevin sampler
  
  NumericMatrix get_phis();
  NumericMatrix get_thetas();
  NumericMatrix MatrixToR(NumericMatrix input);
  NumericMatrix avgMatrix(NumericMatrix A, NumericMatrix B, int weight);
  arma::mat getTDM(int W, int D, vector< vector<int> > w_num);
  
  double ProposalDensity(arma::mat phi);
  double ArrayMax(double array[], int numElements);
  double ArrayMin(double array[], int numElements);
  
};

TwitterNews::TwitterNews(Reference Obj)
  {     
  content = as<vector<string> >(Obj.field("corpus"));
  s_string = as<vector<string> >(Obj.field("published_in"));
  c = as<vector<int> >(Obj.field("citations"));
  
  vector<int> empty_articles = mark_empty_articles(content);
  content = remove_empty_articles(content, empty_articles);
  D = content.size(); 
  
  for (int d=0;d<D;d++)
    {
    string current_preview = content[d].substr(0,250);
    article_preview.push_back(current_preview);   
    }
    
  url = as<vector<string> >(Obj.field("url"));
  url = remove_empty_articles(url, empty_articles);
  
  title = as<vector<string> >(Obj.field("title"));
  title = remove_empty_articles(title, empty_articles);
  
  K = as<int>(Obj.field("K"));
  
  stop_de_path = as<string>(Obj.field("stop_de_path"));
  stop_en_path = as<string>(Obj.field("stop_en_path"));
  stopwords_en =  read_stopwords(ENGLISH);
  stopwords_de =  read_stopwords(GERMAN);
  
  alpha = Obj.field("alpha");
  beta = Obj.field("beta");
  tau = Obj.field("tau");
  gamma = Obj.field("gamma");
  delta = Obj.field("delta");
  sigma = 0.00001;
  
  PreProcessText();
  ExtractWords();
  ConstructVocabulary();
  CreateNewspapers();
  S = Newspapers.size();
  W = Vocabulary.size();
  InitSampling();
  n_wd = getTDM(W, D, w_num);
  
  };
  
  void TwitterNews::PreProcessText()
  {
  for (int d=0;d<D;d++)
    {
    boost::algorithm::to_lower(content[d]);
    content[d] = remove_numbers(content[d]);
    content[d] = remove_punctuation(content[d]);
    content[d] = remove_special_characters(content[d]);
    content[d] = remove_whitespace(content[d]);
    }
  };

void TwitterNews::ExtractWords()
  {
  for (int d=0;d<D;d++)
    {
    vector<string> wd = isolate_words(content[d]);
    wd = eliminate_empty_words(wd);
    wd = eliminate_stopwords(wd);
    w.push_back(wd);
    }
  };

void TwitterNews::ConstructVocabulary()
  {
  int pos = 0;
  vector<vector<int> > w_num_temp(D);
  for (int d=0;d<D;d++)
    {
      int nd = w[d].size();
      for (int i=0;i<nd;i++)
      {
      string target = w[d][i]; 
      vector<string>::iterator it = std::find(Vocabulary.begin(), Vocabulary.end(), target);
      bool isPresent = (it != Vocabulary.end());
      if (isPresent==false) 
        {
        Vocabulary.push_back(target);
        w_num_temp[d].push_back(pos);
        pos += 1;
        }
      else
        {
        int index = std::distance(Vocabulary.begin(), it);
        w_num_temp[d].push_back(index); 
        }
      }
    }
  w_num = w_num_temp;
  }
  
void TwitterNews::CreateNewspapers()
  {
  int pos = 0;
  vector<int> s_temp(D);
  for (int d=0;d<D;d++)
    {
      string target = s_string[d];
      vector<string>::iterator it = std::find(Newspapers.begin(), Newspapers.end(), target);
      bool isPresent = (it != Newspapers.end());
      if (isPresent==false) 
        {
        Newspapers.push_back(target);
        s_temp[d] = pos;
        pos += 1;
        }
      else
        {
        int index = std::distance(Newspapers.begin(), it);
        s_temp[d] = index; 
        }
      }
  s = s_temp;
  }
  
  
vector<int> TwitterNews::mark_empty_articles(vector<string> input)
  {
  int len = input.size();
  int empty_counter = 0;
  vector<int> nonempty_articles;
  for (int i = 0; i < len; i++)
    {
    if (input[i] == "") empty_counter++;
    else nonempty_articles.push_back(i);    
    }
  Rcout << "Articles with empty content were marked. There are" << empty_counter << " empty articles.";
  return nonempty_articles;
  }
  
vector<string> TwitterNews::remove_empty_articles(vector<string> input, vector<int> nonempty)
  {
  int len = input.size();
  int empty_counter = 0;
  vector<string> output;
  
  for(std::vector<int>::iterator it = nonempty.begin(); it != nonempty.end(); ++it) 
    {
    Rcout << *it; 
    output.push_back(input[*it]);
    }
  return output;
  }

int TwitterNews::SampleZ()  
  {
  boost::uniform_int<> dist(0, K-1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > sample(rng, dist);
  return sample();
  }
  
void TwitterNews::InitSampling()
  {
    vector<int> nk_temp(K,0);
    arma::Mat<int> nsk_temp(S,K);
    nsk_temp.zeros();
    arma::Mat<int> csk_temp(S,K);
    csk_temp.zeros();
    vector<int> ns_temp(S,0);
    arma::Mat<int> nw_temp(W,K);
    nw_temp.zeros();
    arma::Mat<int> N_dw_temp(D,W);
    N_dw_temp.zeros();
    vector<int> nw_sum_temp(K,0);
    vector<int> z_temp(D);
      		
	  for (int d=0; d<D; d++)
    {		
    int topic = SampleZ();
    int source = s[d];
    
    nk_temp[topic] = nk_temp[topic] + 1;
    z_temp[d] = topic;
    ns_temp[source] = ns_temp[source] + 1;
    nsk_temp(source,topic) = nsk_temp(source,topic) + 1;
    csk_temp(source,topic) = csk_temp(source,topic) + c[d];
    
    int len = w_num[d].size();			
			for (int i=0; i<len; i++)
			{
  				int wtemp = w_num[d][i];
				  // number of instances of word i appears in documents of topic j
  				nw_temp(wtemp,topic) += 1;
				  // total number of words assigned to topic j           
  				nw_sum_temp[topic] += 1;
          // number of times word i appears in document d
          N_dw_temp(d,wtemp) += 1;
			}
    }
    
  nk = nk_temp;
  nsk = nsk_temp;
  csk = csk_temp;
  ns = ns_temp;
  nw_sum = nw_sum_temp; 
  nw = nw_temp;
  z = z_temp;
  N_dw = N_dw_temp;
  }
  
vector<std::string> TwitterNews::read_stopwords(int lang)
  {
  vector<std::string> output;
  string line;
  
  const char * stop_en_char = stop_en_path.c_str();
  const char * stop_de_char = stop_de_path.c_str();
    
  ifstream myfile(stop_en_char);
  ifstream myfile2(stop_de_char);
  
  switch(lang){
  case ENGLISH:   
  // Rcout << "Das ist Englisch";
    if (myfile.is_open())
    {
      while ( myfile.good() )
      {
        getline (myfile,line);
        output.push_back(line);
      }
      myfile.close();
    }    
    
  break;
  case GERMAN:
      if (myfile2.is_open())
    {
      while ( myfile2.good() )
      {
        getline (myfile2,line);
        output.push_back(line);
      }
      myfile2.close();
    }    
  
  break; 
  default: 
  
  break;
  }  
  return output;  
  }

vector<string> TwitterNews::eliminate_stopwords(vector<string> input)
  {
  // Rcout << "Laenge des Inputs" << input.size();
  vector<string> output;
  
  int input_length = input.size();
  for (int i = 0; i<input_length; i++)
    {
      
     string target = input[i];
     vector<string>::iterator it = std::find(stopwords_en.begin(), stopwords_en.end(), target);
     if (it == stopwords_en.end()) 
     {
     vector<string>::iterator it_de = std::find(stopwords_de.begin(), stopwords_de.end(), target);
     if (it_de == stopwords_de.end()) output.push_back(target);
     }
          
    }
    
  // Rcout << "Lange des Output" << output.size();
  return output;  
  }
  
void TwitterNews::collapsedGibbs(int iter, int burnin, int thin)
 {
  
  for (int i = 0; i < iter; ++i)
  {
    for (int d = 0; d < D; ++d)
      {        
      int sd = s[d];
      int old_topic = z[d];
      vector<double>  prob(K);
      
      for(int j=0; j<K; j++)
        {
        double nsk_sj = nsk(sd,j);
        double csj = csk(sd,j);
        double citation_term = 
        (csj + gamma - c[d]) * log(nsk_sj + delta - 1) -
                  (csj + gamma) * log(nsk_sj + delta);    
        
        double source_gamma_term = lgamma(csj + gamma) - lgamma(csj + gamma -c[d]);
        
        double source_term = log(nsk_sj + alpha - 1) - log(ns[sd]+ K * alpha - 1);
        
        int len = w_num[d].size();
        double topic_term = lgamma(nw_sum[j] + W * beta - len) - lgamma(nw_sum[j] + W * beta);      
        
        double w_sum = 0;
        for(int w=0;w<W;w++)
          {
          double nw_wj = nw(w,j);
          double Ndw = N_dw(d,w);
          double summand = lgamma(nw_wj + beta) - lgamma(nw_wj + beta - Ndw);
          // Rcout << summand;
          w_sum += summand;
          }
        
        prob[j] = citation_term + source_gamma_term +  source_term + topic_term + w_sum;
        prob[j] = prob[j] / 1000;
        //Rcout << "w_sum:" << w_sum << "\n";
        // Rcout << "Prob for class" << j << ":" << prob[j] << " ";
        }
        
      // Normalize prob vector
      int sum = std::accumulate(prob.begin(), prob.end(), 0);
      for (int j = 0; j < K; j++)
        {
        prob[j] = prob[j] / sum;
        }
               
      for (int r = 1; r < K; ++r)
        {
        prob[r] = prob[r] + prob[r - 1];
        }

      double u  = prob[K-1] * rand() / double(RAND_MAX);

      int new_topic = 0; // set up new topic

      for (int nt = 0 ; nt < K; ++nt)
          {
          if (prob[nt] > u)
            {
            new_topic = nt;
            break;
            }
          }
          
    // remove z_d from counts
    nsk(sd,old_topic) -= 1;
    csk(sd,old_topic) -= c[d];
    nk[old_topic] -= 1;

    //assign new z_d to counts
    nsk(sd,new_topic) += 1;
    csk(sd,new_topic) += c[d];
    nk[new_topic] += 1;
    
    for(int w=0;w<W;w++)
          {
          nw(w,old_topic)  -= n_wd(w,d);
          nw(w,new_topic)  += n_wd(w,d);
          }
     
    for(int k=0;k<K;k++)
        {
        arma::Col<int> nw_col = nw.col(k);
        nw_sum[k] = arma::accu(nw_col);
        }
    
     z[d] = new_topic;

      }
         
     Rcout << "--- END OF DOCUMENTS --- ";
     if (i % thin == 0 && i > burnin)
           {
           z_list.push_back(z);  
           getParameterEstimates();
           }

    }
    Rcout << "--- END OF ITERATIONS ---";  
  }
  
NumericMatrix TwitterNews::avgMatrix(NumericMatrix A, NumericMatrix B, int weight)
  {
  int nrow = A.nrow();
  int ncol = A.ncol();
  NumericMatrix C(nrow,ncol);

  float wf = (float) weight;
  float propA = (wf-1) / wf;
  float propB = 1 / wf;

  for (int i=0; i<nrow;i++)
  {
    for (int j=0; j<ncol;j++)
    {
    C(i,j) =  propA * A(i,j) + propB * B(i,j);
    }
  }

  return C;
  }
  
  
void TwitterNews::getParameterEstimates()
 {
  int weight = z_list.size();
  for (int i=0;i<weight;i++)
    {
    arma::mat phi_current = get_phi_estimates();
    arma::mat theta_current = get_theta_estimates();
    arma::mat lambda_current = get_lambda_estimates();
    arma::mat prob_s_current = get_prob_s();
    vector<double> kappa_current = get_kappa_estimates();
    
    NumericMatrix prob_s_add = wrap(prob_s_current);
    NumericMatrix phi_add = wrap(phi_current);  
    NumericMatrix theta_add = wrap(theta_current);  
    NumericMatrix lambda_add = wrap(lambda_current);  
    NumericVector kappa_add = wrap(get_kappa_estimates());
  
    if (i == 0) 
     {
     phi_avg = phi_add;
     theta_avg = theta_add;
     lambda_avg = lambda_add;
     kappa_avg = kappa_add;
     prob_s = prob_s_add;
     }
    else 
      {
      phi_avg = avgMatrix(phi_avg, phi_add, weight);
      theta_avg = avgMatrix(theta_avg, theta_add, weight);
      lambda_avg = avgMatrix(lambda_avg, lambda_add, weight);
      prob_s = avgMatrix(prob_s, prob_s_add, weight);
      }
    }   
 }
  
vector<double> TwitterNews::get_kappa_estimates()
  {
  vector<double> kappa(S);
  for (int s=0;s<S;s++)
    {
    double res = (ns[s] + tau) / (D + S * tau);
    kappa[s] = res;  
    }
  return kappa;
  }
  
arma::mat TwitterNews::get_prob_s()
  {
  arma::mat prob_mat(D,S);
  prob_mat.zeros();
  
  for (int d=0;d<D;d++)
    {
    arma::rowvec prob_vec(S);
    for (int s=0;s<S;s++)
      {
      double nsk_sz = nsk(s,z[d]);
      double T1 = log(nsk_sz + tau - 1) - log(D + K * tau - 1);
      double T2 = log(nsk_sz + alpha -1) - log(ns[s] + K*alpha - 1);
      double csk_sz = csk(s,z[d]);
      double T3 = lgamma(csk_sz + gamma) - lgamma(csk_sz + gamma - c[d]);
      double T4 = (csk_sz + gamma - c[d]) * log(nsk_sz + delta - 1) - (csk_sz + gamma) * log(nsk_sz + delta);
      prob_vec[s] = exp(T1 + T2 + T3 + T4);
      Rcout << prob_vec[s];
      } 
    double prob_sum = norm(prob_vec, 2);  
    prob_mat.row(d) = prob_vec / prob_sum;  
    }  
  return prob_mat;
  }
    
arma::mat TwitterNews::get_lambda_estimates()
  {
  arma::mat lambda(S,K);
  for (int s=0;s<S;s++)
    for (int k=0;k<K;k++)
      {
      double csk_sk = csk(s,k);
      double nsk_sk = nsk(s,k);
      lambda(s,k) = (csk_sk + gamma) / (nsk_sk + delta); 
      }
  return lambda;
  }
  
arma::mat TwitterNews::get_phi_estimates()
  {
  arma::mat phi(K,W);
  phi.zeros();
  for (int k=0;k<K;k++)
    for (int w=0;w<W;w++)
    {
    double n_kw = nw(w,k);
    phi(k,w) = (n_kw + beta) / (nw_sum[k] + W * beta);
    }
  return phi;
  }
  
arma::mat TwitterNews::get_theta_estimates()
  {
  arma::mat theta(S,K);  
  for (int s=0;s<S;s++)
    for (int k=0;k<K;k++)
      {
      double nsk_sk = nsk(s,k);
      theta(s,k) = (nsk_sk + alpha) / (ns[s] + K * alpha);
      }
  return theta;
  }
  
arma::mat TwitterNews::getTDM(int W, int D, vector< vector<int> > w_num) 
  {
  arma::mat tdm(W,D);
  for (int d=0; d<D; ++d)
   for (int w=0; w<W; ++w)
    {
    int freq = 0;
    vector<int> current_w = w_num[d];
    int wlen = current_w.size();
    for (int l=0; l<wlen; ++l)
      {
      if(current_w[l] == w) freq += 1;
      }
  
     tdm(w,d) = freq;
    }
  return tdm;
  }
  
CharacterVector TwitterNews::TopicTerms(int k, int no)
  {
  vector<double> k_phi(W);
  NumericVector k_phi_R = phi_avg(k,_);
  k_phi = as<vector<double> > (k_phi_R);
  NumericVector ret_vector(no);

  for (int i=0;i<no;i++)
    {
    std::vector<double>::iterator result;
    result = std::max_element(k_phi.begin(),k_phi.end());
    int biggest_id = std::distance(k_phi.begin(), result);
    ret_vector[i] = biggest_id;
    k_phi[biggest_id] = 0;
    }

  CharacterVector ret_char_vector(no);

  for (int i=0;i<no;i++)
    {
    ret_char_vector[i] = Vocabulary[ret_vector[i]];
    }

  return ret_char_vector;
  }

CharacterMatrix TwitterNews::Terms(int k)
  {
  CharacterMatrix ret(K,k);
  for (int i = 0; i < K; i++)
    {
    CharacterVector temp = TopicTerms(i,k);
    ret(i,_) =  temp;
    }
  return ret;
  }
  
RCPP_MODULE(yada2){
class_<TwitterNews>( "TwitterNews" )
.constructor<Reference>()
.field("content",&TwitterNews::content)
.field("url",&TwitterNews::url)
.field("title",&TwitterNews::title)
.field( "n_wd", &TwitterNews::n_wd)
.field( "z", &TwitterNews::z)
.field( "c", &TwitterNews::c)
.field( "s", &TwitterNews::s)
.field("nk",&TwitterNews::nk)
.field( "nw_sum", &TwitterNews::nw_sum)
.field("nw",&TwitterNews::nw)
.field("nsk",&TwitterNews::nsk)
.field( "csk", &TwitterNews::csk)
.field("ns",&TwitterNews::ns)
.field("K", &TwitterNews::K)
.field("D",&TwitterNews::D)
.field("W",&TwitterNews::W)
.field("N_dw",&TwitterNews::N_dw)
.field("S",&TwitterNews::S)
.field("alpha",&TwitterNews::alpha)
.field("beta",&TwitterNews::beta)
.field("z_mat",&TwitterNews::z_mat)
.field("Vocabulary",&TwitterNews::Vocabulary)
.field("Newspapers",&TwitterNews::Newspapers)
.field("stop_en_path",&TwitterNews::stop_en_path)
.field("phi_avg",&TwitterNews::phi_avg)
.field("theta_avg",&TwitterNews::theta_avg)
.field("lambda_avg",&TwitterNews::lambda_avg)
.field("kappa_avg",&TwitterNews::kappa_avg)
.field("prob_s",&TwitterNews::prob_s)
.field("PhiProdMat",&TwitterNews::PhiProdMat)
.field("article_preview",&TwitterNews::article_preview)
.method("collapsedGibbs",&TwitterNews::collapsedGibbs)
.method("get_phi_estimates",&TwitterNews::get_phi_estimates)
.method("Terms",&TwitterNews::Terms)
;
}     
