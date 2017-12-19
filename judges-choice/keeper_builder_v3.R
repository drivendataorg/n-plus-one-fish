# Clean environment and load required packages
rm(list=ls())
cat("\014")

#open keeper data
#Read training file
setwd("~/darknet-master/build/darknet/x64/results/")
file_name = "keepers.csv"
keeper <- fread(file_name)
colnames(keeper) <- c("filename","score","x1","y1","x2","y2")

confidence = 0.25

df <- data.frame()

frame <- gsub(".*_", "", keeper$filename)
keeper$frame <- as.integer(frame)
newdata <- subset(keeper, (score >= confidence))
frame <- gsub(".*_", "", newdata$filename)

keeper_length = dim(newdata)[1]
for (i in 1:keeper_length)
{
      x <- cbind(newdata$filename[i],newdata$score[i],newdata$x1[i],newdata$y1[i],newdata$x2[i],newdata$y2[i])
      df <- rbind(df,x)
print(i)
}


colnames(df) <- c("filename","score","x1","y1","x2","y2")
out_file <- paste("~/darknet-master/build/darknet/x64/results/","keeper","_",100*confidence,"_","confidence_v3",".csv",sep="")

# Write out dataset
write.csv(df, out_file, row.names = FALSE)