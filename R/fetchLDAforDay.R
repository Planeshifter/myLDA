fetchLDAforDay <- function(day)
  {
  day <- as.Date(day)
  require(newscrapeR)
  artList <- fetch_SQL(db_name="newscrapeR.db",from=day,to=day+1)
  LDA_model <- myLDA(artList, K = 12)
  LDA_model$collapsedGibbs(1000,100,100)
  
  mds.results <- getMDS(LDA_model)
  require(RJSONIO)
  mds.results.json <- toJSON(mds.results)
  file_name <- paste("json_data/", gsub(pattern="-",replacement="_",x=day), ".json", sep="")
  writeLines(mds.results.json,con=file_name)
  }