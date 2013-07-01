#include <RcppArmadillo.h>
// [[Rcpp::depends("RcppArmadillo")]]

#include <string>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>

using namespace Rcpp;
using namespace std;
using namespace boost;

std::string remove_numbers(std::string input)
{
boost::regex digit("\\d");
std::string fmt = "";
std::string output = boost::regex_replace(input, digit, fmt);
return output;
}

std::string remove_whitespace(std::string input)
{
boost::regex white("\\s+");
std::string fmt = " ";
std::string output = boost::regex_replace(input, white, fmt);
return output;
}

std::string remove_punctuation(std::string input)
{
boost::regex punct("[[:punct:]]");
std::string fmt = " ";
std::string output = boost::regex_replace(input,punct,fmt);
return output;
}

vector<std::string> isolate_words(std::string input)
  {
  vector<std::string> output_vector;
  boost::split(output_vector, input, boost::is_any_of("\t "));
  return output_vector;
  }

vector<std::string> eliminate_empty_words(vector<std::string> input)
  {
  vector<std::string> output_vector;
  int length = input.size();
  for (int i=0; i<length; i++)
    {
    if (input[i]!="\0") 
      output_vector.push_back(input[i]);  
    }
  return output_vector;
  }
  
std::string remove_special_characters(std::string input)
{
boost::regex only_words("[^\\wäöüÄÖÜß]");
std::string fmt = " ";
std::string output = boost::regex_replace(input,only_words,fmt);
return output;
}

vector<std::string> get_actors(std::string input)
  {
  
  vector<string> actor_names;
  vector<string> output;
  
  // get five capitalized words when appearing in a row 
  boost::regex sentence5("(?<!\\.\\s)([[:upper:]][A-Za-z]+\\s){4}([[:upper:]][A-Za-z]+)");
  
  boost::sregex_token_iterator iter5(input.begin(), input.end(), sentence5, 0);
  boost::sregex_token_iterator end;
  
      for( ; iter5 != end; iter5++) {
          actor_names.push_back(*iter5);
      }
      
  input = boost::regex_replace(input,sentence5," ");
  
   // get four capitalized words when appearing in a row 
  boost::regex sentence4("(?<!\\.\\s)([[:upper:]][A-Za-z]+\\s){3}([[:upper:]][A-Za-z]+)");
  
  boost::sregex_token_iterator iter4(input.begin(), input.end(), sentence4, 0);
  
      for( ; iter4 != end; iter4++) {
          actor_names.push_back(*iter4);
      }
      
  input = boost::regex_replace(input,sentence4," ");
  
  // get three capitalized words when appearing in a row 
  boost::regex sentence3("(?<!\\.\\s)([[:upper:]][A-Za-z]+\\s){3}([[:upper:]][A-Za-z]+)");
  
  boost::sregex_token_iterator iter3(input.begin(), input.end(), sentence3, 0);
  
      for( ; iter3 != end; iter3++) {
          actor_names.push_back(*iter3);
      }
      
  input = boost::regex_replace(input,sentence3," ");
  
  // get capitalized words when two appear in a row 
  boost::regex sentence2("(?<!\\.\\s)([[:upper:]][A-Za-z]+\\s[[:upper:]][A-Za-z]+)");
  
  boost::sregex_token_iterator iter2(input.begin(), input.end(), sentence2, 0);
      for( ; iter2 != end; iter2++) {
          actor_names.push_back(*iter2);
      }
      
  input = boost::regex_replace(input,sentence2," ");
  
  
  // get capitalized words when one appears
  boost::regex sentence1("(?<=[A-Za-z]\\s)([[:upper:]][A-Za-z]+)");
  
  boost::sregex_token_iterator iter1(input.begin(), input.end(), sentence1, 0);
      for( ; iter1 != end; iter1++) {
            actor_names.push_back(*iter1);
      }
      
  input = boost::regex_replace(input,sentence1," ");
  
  for (int i=0; i<actor_names.size();i++)
    {
    string target = actor_names[i];
    vector<string>::iterator it = std::find(output.begin(), output.end(), target);
    bool isPresent = (it != output.end());
    if (isPresent==false) output.push_back(target);    
    }
  
  return output;
  }

void print_vector(vector<std::string> input)
{
for( std::vector<string>::iterator i = input.begin(); i != input.end(); ++i)
	    Rcout << *i << '\n';
}